import numpy as np
import gymnasium
from gymnasium import spaces

class EnhancedPortfolioEnv(gymnasium.Env):
    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, lookback=30, 
                 max_drawdown_limit=0.25, min_diversification=0.2):
        super(EnhancedPortfolioEnv, self).__init__()
        
        self.df = df
        self.n_stocks = len(df.columns) // 5
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback = lookback
        self.max_drawdown_limit = max_drawdown_limit
        self.min_diversification = min_diversification
        
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32
        )
        
        obs_shape = (self.n_stocks * 5 + self.n_stocks + 2,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        
        self.portfolio_history = []
        self.returns_history = []
        self.trades_history = []
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_stocks)
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        
        self.portfolio_history = [self.initial_balance]
        self.returns_history = []
        self.trades_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
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
    
    def _calculate_diversification_penalty(self, weights):
        concentration = np.sum(weights ** 2)
        
        if concentration > (1 - self.min_diversification):
            penalty = (concentration - (1 - self.min_diversification)) * 50
            return penalty
        return 0
    
    def _calculate_drawdown_penalty(self):
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        if drawdown > self.max_drawdown_limit:
            penalty = (drawdown - self.max_drawdown_limit) * 100
            return penalty
        return 0
    
    def _calculate_transaction_penalty(self, trades_made):
        return sum(trades_made) * 0.1
    
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
        
        trades_made = []
        for i in range(self.n_stocks):
            diff = target_values[i] - current_holdings_value[i]
            
            if diff > 0:
                shares_to_buy = diff / prices[i]
                cost = shares_to_buy * prices[i] * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.shares_held[i] += shares_to_buy
                    self.balance -= cost
                    trades_made.append(abs(diff))
            elif diff < 0:
                shares_to_sell = -diff / prices[i]
                shares_to_sell = min(shares_to_sell, self.shares_held[i])
                revenue = shares_to_sell * prices[i] * (1 - self.transaction_cost)
                self.shares_held[i] -= shares_to_sell
                self.balance += revenue
                trades_made.append(abs(diff))
            else:
                trades_made.append(0)
        
        self.trades_history.append(trades_made)
        
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            done = True
            truncated = False
        else:
            done = False
            truncated = False
        
        next_prices = []
        for i in range(self.n_stocks):
            price = self.df.iloc[self.current_step, i * 5]
            next_prices.append(price)
        next_prices = np.array(next_prices)
        
        self.portfolio_value = self.balance + (self.shares_held * next_prices).sum()
        self.portfolio_history.append(self.portfolio_value)
        
        reward = self._calculate_reward(target_weights, trades_made)
        
        obs = self._get_observation()
        
        return obs, reward, done, truncated, {}
    
    def _calculate_reward(self, weights, trades_made):
        returns = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        self.returns_history.append(returns)
        
        base_reward = returns * 100
        
        diversification_penalty = self._calculate_diversification_penalty(weights)
        drawdown_penalty = self._calculate_drawdown_penalty()
        transaction_penalty = self._calculate_transaction_penalty(trades_made)
        
        if len(self.returns_history) >= 30:
            recent_returns = np.array(self.returns_history[-30:])
            volatility = np.std(recent_returns)
            sharpe_bonus = (np.mean(recent_returns) / (volatility + 1e-8)) * 5
        else:
            sharpe_bonus = 0
        
        total_reward = base_reward + sharpe_bonus - diversification_penalty - drawdown_penalty - transaction_penalty
        
        return total_reward
    
    def get_metrics(self):
        if len(self.portfolio_history) < 2:
            return {}
        
        portfolio_array = np.array(self.portfolio_history)
        returns_array = np.array(self.returns_history) if self.returns_history else np.array([0])
        
        total_return = (portfolio_array[-1] - portfolio_array[0]) / portfolio_array[0]
        
        if len(returns_array) > 1:
            sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - peak) / peak
        max_drawdown = np.min(drawdown)
        
        total_trades = sum([sum(trades) for trades in self.trades_history])
        
        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown) * 100,
            'final_value': portfolio_array[-1],
            'total_trades': total_trades
        }
    
    def render(self, mode='human'):
        profit = self.portfolio_value - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: ${self.balance:.2f}')
        print(f'Portfolio Value: ${self.portfolio_value:.2f}')
        print(f'Profit: ${profit:.2f} ({profit/self.initial_balance*100:.2f}%)')