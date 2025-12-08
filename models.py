import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value
    
    def get_action(self, state, deterministic=False):
        action_mean, value = self.forward(state)
        
        if deterministic:
            return action_mean, value
        
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        action = torch.tanh(action)
        
        return action, log_prob, value
    
    def evaluate_actions(self, state, action):
        action_mean, value = self.forward(state)
        
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_prob, entropy