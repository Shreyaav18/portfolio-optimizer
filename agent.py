import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import LSTMActorCritic, ActorCritic

class EnhancedPPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, 
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                 use_lstm=True, use_attention=True, risk_aversion=0.5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.risk_aversion = risk_aversion
        self.use_lstm = use_lstm
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_lstm:
            self.policy = LSTMActorCritic(obs_dim, action_dim, use_attention=use_attention).to(self.device)
        else:
            self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.portfolio_values = []
        self.attention_weights = []
        
        self.episode_returns = []
        self.episode_lengths = []
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.use_lstm:
                if deterministic:
                    action, value, _, attention = self.policy.get_action(state, deterministic=True)
                    return action.cpu().numpy()[0], None, value.cpu().item(), attention
                else:
                    action, log_prob, value, _, attention = self.policy.get_action(state, deterministic=False)
                    return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item(), attention
            else:
                if deterministic:
                    action, value = self.policy.get_action(state, deterministic)
                    return action.cpu().numpy()[0], None, value.cpu().item(), None
                else:
                    action, log_prob, value = self.policy.get_action(state, deterministic)
                    return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item(), None
    
    def store_transition(self, state, action, reward, log_prob, value, done, portfolio_value=None, attention=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        if portfolio_value is not None:
            self.portfolio_values.append(portfolio_value)
        if attention is not None:
            self.attention_weights.append(attention)
    
    def calculate_risk_adjusted_reward(self, returns_history, current_reward):
        if len(returns_history) < 10:
            return current_reward
        
        recent_returns = np.array(returns_history[-30:])
        volatility = np.std(recent_returns)
        
        volatility_penalty = volatility * self.risk_aversion * 10
        
        if len(self.portfolio_values) >= 2:
            portfolio_history = np.array(self.portfolio_values)
            peak = np.maximum.accumulate(portfolio_history)
            drawdown = (portfolio_history - peak) / peak
            max_drawdown = abs(np.min(drawdown))
            
            if max_drawdown > 0.15:
                drawdown_penalty = (max_drawdown - 0.15) * 100
            else:
                drawdown_penalty = 0
        else:
            drawdown_penalty = 0
        
        risk_adjusted = current_reward - volatility_penalty - drawdown_penalty
        
        return risk_adjusted
    
    def compute_returns(self, next_value):
        returns = []
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * 0.95 * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        return returns, advantages
    
    def update(self, next_state):
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.use_lstm:
                _, next_value, _, _ = self.policy(next_state)
            else:
                _, next_value = self.policy(next_state)
            next_value = next_value.cpu().item()
        
        returns, advantages = self.compute_returns(next_value)
        
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        for _ in range(10):
            values, log_probs, entropy = self.policy.evaluate_actions(states, actions)
            values = values.squeeze()
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            entropy_loss = -entropy.mean()
            
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        
        episode_return = sum(self.rewards)
        episode_length = len(self.rewards)
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.portfolio_values = []
        self.attention_weights = []
        
        return total_loss / 10
    
    def get_statistics(self):
        if len(self.episode_returns) == 0:
            return {}
        
        return {
            'mean_return': np.mean(self.episode_returns[-100:]),
            'std_return': np.std(self.episode_returns[-100:]),
            'mean_length': np.mean(self.episode_lengths[-100:]),
            'max_return': np.max(self.episode_returns[-100:]),
            'min_return': np.min(self.episode_returns[-100:])
        }
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'episode_returns' in checkpoint:
            self.episode_returns = checkpoint['episode_returns']
        if 'episode_lengths' in checkpoint:
            self.episode_lengths = checkpoint['episode_lengths']

class MultiAgentEnsemble:
    def __init__(self, obs_dim, action_dim, num_agents=3):
        self.agents = [
            EnhancedPPOAgent(obs_dim, action_dim, use_lstm=True, use_attention=True, risk_aversion=0.3),
            EnhancedPPOAgent(obs_dim, action_dim, use_lstm=True, use_attention=True, risk_aversion=0.5),
            EnhancedPPOAgent(obs_dim, action_dim, use_lstm=True, use_attention=True, risk_aversion=0.7)
        ]
        self.num_agents = num_agents
    
    def get_action(self, state, deterministic=False):
        actions = []
        values = []
        
        for agent in self.agents:
            action, _, value, _ = agent.get_action(state, deterministic)
            actions.append(action)
            values.append(value)
        
        ensemble_action = np.mean(actions, axis=0)
        ensemble_value = np.mean(values)
        
        return ensemble_action, None, ensemble_value, None
    
    def save(self, path_prefix):
        for i, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent{i}.pth")
    
    def load(self, path_prefix):
        for i, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent{i}.pth")