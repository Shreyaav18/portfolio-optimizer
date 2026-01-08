import torch
import torch.nn as nn
from torch.distributions import Normal
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class StockAttention(nn.Module):
    def __init__(self, n_stocks, feature_dim):
        super().__init__()
        self.n_stocks = n_stocks
        self.feature_dim = feature_dim
        
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        self.scale = math.sqrt(feature_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, self.n_stocks, self.feature_dim)
        
        Q = self.query(x_reshaped)
        K = self.key(x_reshaped)
        V = self.value(x_reshaped)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.view(batch_size, -1)
        
        return attended, attention_weights

class LSTMActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, lstm_layers=2, use_attention=True):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )
        
        if use_attention:
            self.attention = StockAttention(n_stocks=action_dim, feature_dim=hidden_dim // action_dim)
            attention_output_dim = hidden_dim
        else:
            attention_output_dim = hidden_dim
        
        self.actor_fc1 = nn.Linear(attention_output_dim if use_attention else hidden_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        
        self.critic_fc1 = nn.Linear(attention_output_dim if use_attention else hidden_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_value = nn.Linear(hidden_dim, 1)
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, state, hidden=None):
        if len(state.shape) == 2:
            state = state.unsqueeze(1)
        
        lstm_out, hidden = self.lstm(state, hidden)
        features = lstm_out[:, -1, :]
        features = self.layer_norm(features)
        
        if self.use_attention:
            attended, attention_weights = self.attention(features)
            actor_input = attended
            critic_input = attended
        else:
            actor_input = features
            critic_input = features
            attention_weights = None
        
        actor_h = torch.relu(self.actor_fc1(actor_input))
        actor_h = self.dropout(actor_h)
        actor_h = torch.relu(self.actor_fc2(actor_h))
        action_mean = torch.tanh(self.actor_mean(actor_h))
        
        critic_h = torch.relu(self.critic_fc1(critic_input))
        critic_h = self.dropout(critic_h)
        critic_h = torch.relu(self.critic_fc2(critic_h))
        value = self.critic_value(critic_h)
        
        return action_mean, value, hidden, attention_weights
    
    def get_action(self, state, hidden=None, deterministic=False):
        action_mean, value, hidden, attention_weights = self.forward(state, hidden)
        
        if deterministic:
            return action_mean, value, hidden, attention_weights
        
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        action = torch.tanh(action)
        
        return action, log_prob, value, hidden, attention_weights
    
    def evaluate_actions(self, state, action, hidden=None):
        action_mean, value, hidden, _ = self.forward(state, hidden)
        
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return value, log_prob, entropy

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