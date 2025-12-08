import numpy as np
import gymnasium
from gymnasium import spaces

class PortfolioEnv(gymnasium.Env):
    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, lookback=30):
        super(PortfolioEnv, self).__init__()
        
        self.df = df
        self.n_stocks = len(df.columns) // 5
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback = lookback
        
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32
        )
        
        obs_shape = (self.n_stocks * 5 + self.n_stocks + 2,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_stocks)
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        
        return self._get_observation()
    
    def _get_observation(self):
        start = self.current_step - self.lookback
        end = self.current_step
        
        prices = []
        for i in range(self.n_stocks):
            price = self.df.iloc[self.current_step, i * 5]
            prices.append(price)
        
        features = self.df.iloc[self.current_step].values
        
        portfolio_weights = self._get_portfolio_weights()
        
        obs = np.concatenate([
            features,
            portfolio_weights,
            [self.balance / self.initial_balance],
            [self.portfolio_value / self.initial_balance]
        ])
        
        return obs.astype(np.float32)
    
    def _get_portfolio_weights(self):
        prices = []
        for i in range(self.n_stocks):
            price = self.df.iloc[self.current_step, i * 5]
            prices.append(price)
        
        holdings_value = self.shares_held * np.array(prices)
        total_value = self.balance + holdings_value.sum()
        
        if total_value > 0:
            weights = holdings_value / total_value
        else:
            weights = np.zeros(self.n_stocks)
        
        return weights
    
    def step(self, action):
        self.previous_portfolio_value = self.portfolio_value
        
        prices = []
        for i in range(self.n_stocks):
            price = self.df.iloc[self.current_step, i * 5]
            prices.append(price)
        prices = np.array(prices)
        
        current_value = self.balance + (self.shares_held * prices).sum()
        
        target_weights = (action + 1) / 2
        target_weights = target_weights / (target_weights.sum() + 1e-8)
        
        target_values = target_weights * current_value
        current_holdings_value = self.shares_held * prices
        
        for i in range(self.n_stocks):
            diff = target_values[i] - current_holdings_value[i]
            
            if diff > 0:
                shares_to_buy = diff / prices[i]
                cost = shares_to_buy * prices[i] * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.shares_held[i] += shares_to_buy
                    self.balance -= cost
            elif diff < 0:
                shares_to_sell = -diff / prices[i]
                shares_to_sell = min(shares_to_sell, self.shares_held[i])
                revenue = shares_to_sell * prices[i] * (1 - self.transaction_cost)
                self.shares_held[i] -= shares_to_sell
                self.balance += revenue
        
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            done = True
        else:
            done = False
        
        next_prices = []
        for i in range(self.n_stocks):
            price = self.df.iloc[self.current_step, i * 5]
            next_prices.append(price)
        next_prices = np.array(next_prices)
        
        self.portfolio_value = self.balance + (self.shares_held * next_prices).sum()
        
        reward = self._calculate_reward()
        
        obs = self._get_observation()
        
        return obs, reward, done, {}
    
    def _calculate_reward(self):
        returns = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        
        return returns * 100
    
    def render(self, mode='human'):
        profit = self.portfolio_value - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: ${self.balance:.2f}')
        print(f'Portfolio Value: ${self.portfolio_value:.2f}')
        print(f'Profit: ${profit:.2f}')