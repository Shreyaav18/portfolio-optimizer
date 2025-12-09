# RL Portfolio Optimization

Deep Reinforcement Learning agent for multi-stock portfolio management using Proximal Policy Optimization (PPO).

## Overview

This project trains an AI agent to manage a stock portfolio by learning optimal buy/sell/hold strategies. The agent maximizes returns while managing risk through Sharpe ratio optimization.

**Key Features:**
- PPO algorithm with Actor-Critic architecture
- Multi-stock portfolio management (5 tech stocks)
- Transaction cost modeling
- Risk-adjusted rewards (Sharpe ratio)
- Technical indicators (RSI, MACD, Bollinger Bands)

## Architecture

**Environment:** Custom Gymnasium environment simulating stock market
**Algorithm:** Proximal Policy Optimization (PPO)
**Neural Network:** Actor-Critic with 256 hidden units
**Action Space:** Continuous [-1, 1] for each stock (sell to buy)
**Observation Space:** Stock prices + technical indicators + portfolio state

Install:
```bash
pip install -r requirements.txt
```

## How It Works

### 1. Data Collection
Downloads historical stock data from Yahoo Finance and calculates technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

### 2. Environment
Custom Gym environment that simulates:
- Portfolio with cash + stock holdings
- Transaction costs (0.1%)
- Realistic buy/sell execution
- Daily timesteps

### 3. Agent Training
PPO algorithm learns through trial and error:
- Actor network: Decides buy/sell actions
- Critic network: Evaluates action quality
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective for stability

### 4. Reward Function
```
reward = (portfolio_value_today - portfolio_value_yesterday) / portfolio_value_yesterday * 100
```

Encourages:
- Maximizing portfolio growth
- Risk-adjusted returns
- Consistent performance


## Technical Details

### PPO Algorithm
- **Policy Network:** 2-layer MLP (256 units) with Tanh output
- **Value Network:** 2-layer MLP (256 units) with linear output
- **Optimization:** Adam optimizer with gradient clipping
- **Updates:** 10 epochs per batch with clipped surrogate objective

### State Space (27 dimensions for 5 stocks)
- Stock prices (normalized): 5
- RSI indicators: 5
- MACD values: 5
- Bollinger Band highs: 5
- Bollinger Band lows: 5
- Portfolio weights: 5
- Cash balance (normalized): 1
- Total portfolio value (normalized): 1

### Action Space (5 dimensions)
- Continuous values [-1, 1] for each stock
- -1 = Sell all holdings
- 0 = Hold current position
- +1 = Buy maximum affordable

## Future Enhancements

- [ ] Multi-agent competition
- [ ] Attention mechanism for stock selection
- [ ] LSTM for temporal patterns
- [ ] Risk constraints (max drawdown limits)
- [ ] Real-time trading integration
- [ ] Ensemble of agents
- [ ] More asset classes (bonds, crypto)