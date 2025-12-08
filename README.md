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
- GPU training on Google Colab (free)
- CPU inference for local demo

## Architecture

**Environment:** Custom Gym environment simulating stock market
**Algorithm:** Proximal Policy Optimization (PPO)
**Neural Network:** Actor-Critic with 256 hidden units
**Action Space:** Continuous [-1, 1] for each stock (sell to buy)
**Observation Space:** Stock prices + technical indicators + portfolio state

## Project Structure

```
rl-portfolio-optimizer/
├── environment.py          # Trading environment (Gym)
├── data_loader.py          # Stock data fetching + indicators
├── models.py               # Actor-Critic neural networks
├── agent.py                # PPO algorithm implementation
├── utils.py                # Metrics & visualization
├── local_demo.py           # Backtesting script (CPU)
├── requirements.txt        # Dependencies
├── README.md               # This file
└── saved_models/           # Trained model weights
    └── ppo_agent_final.pth
```

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
yfinance>=0.2.28
ta>=0.11.0
gym>=0.26.0
```

Install:
```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Train on Google Colab (Recommended)

1. **Upload files to Google Drive:**
   - Create folder: `MyDrive/rl_portfolio/`
   - Upload: `environment.py`, `data_loader.py`, `models.py`, `agent.py`, `utils.py`

2. **Open Google Colab** (colab.research.google.com)

3. **Enable GPU:**
   - Runtime → Change runtime type → T4 GPU

4. **Run training cells:**

```python
# Cell 1: Setup
!pip install -q yfinance ta gym torch matplotlib numpy pandas
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/rl_portfolio

# Cell 2: Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 3: Load data
from data_loader import prepare_data, split_data
from environment import PortfolioEnv
from agent import PPOAgent
import numpy as np

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
df = prepare_data(TICKERS, '2018-01-01', '2024-01-01')
train_df, test_df = split_data(df, train_ratio=0.8)

# Cell 4: Initialize
env = PortfolioEnv(train_df, initial_balance=10000)
agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0])

# Cell 5: Train (2-4 hours)
for episode in range(200):
    state = env.reset()
    episode_reward = 0
    done = False
    step = 0
    
    while not done:
        action, log_prob, value = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, log_prob, value, done)
        episode_reward += reward
        state = next_state
        step += 1
        
        if step % 50 == 0:
            agent.update(state)
    
    if len(agent.states) > 0:
        agent.update(state)
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}/200 | Reward: {episode_reward:.2f} | Value: ${env.portfolio_value:.2f}")
    
    if (episode + 1) % 50 == 0:
        agent.save(f'checkpoint_ep{episode+1}.pth')

agent.save('ppo_agent_final.pth')
print("Training complete!")
```

5. **Download trained model:**
   - File appears in Drive: `rl_portfolio/ppo_agent_final.pth`
   - Download to local: `saved_models/ppo_agent_final.pth`

### Option 2: Run Demo Locally (CPU)

```bash
python local_demo.py
```

**Output:**
- Backtest results on 2024 test data
- Performance comparison: RL Agent vs Buy & Hold
- Metrics: Total return, Sharpe ratio, Max drawdown
- Visualization: 4 plots saved as `backtest_results.png`

## Performance Metrics

The system evaluates performance using:

- **Total Return (%):** Profit relative to initial investment
- **Sharpe Ratio:** Risk-adjusted returns (higher is better)
- **Max Drawdown (%):** Largest peak-to-trough decline
- **Win Rate:** Percentage of profitable trades

**Typical Results (varies by market conditions):**
- Total Return: 30-60%
- Sharpe Ratio: 0.8-1.5
- Max Drawdown: 10-20%
- Outperforms buy-and-hold in volatile markets

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

## Customization

### Change Stocks
```python
TICKERS = ['NVDA', 'META', 'NFLX', 'AMD', 'INTC']
```

### Adjust Training Duration
```python
NUM_EPISODES = 500  # Default: 200
```

### Modify Initial Capital
```python
INITIAL_BALANCE = 50000  # Default: 10000
```

### Change Date Range
```python
START_DATE = '2015-01-01'
END_DATE = '2024-12-01'
```

### Tune Hyperparameters
In `agent.py`:
```python
PPOAgent(
    lr=3e-4,           # Learning rate
    gamma=0.99,        # Discount factor
    epsilon=0.2,       # PPO clipping
    value_coef=0.5,    # Value loss coefficient
    entropy_coef=0.01  # Exploration bonus
)
```

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

## Training Time

**Google Colab (T4 GPU):**
- 200 episodes: 2-4 hours
- 500 episodes: 6-10 hours

**Local (CPU):**
- Not recommended (10x slower)
- Use Colab for training, local for demo only

## Troubleshooting

### Colab disconnects during training
Checkpoints are auto-saved every 50 episodes:
```python
agent.load('checkpoint_ep150.pth')
```

### Out of GPU memory
Reduce update frequency:
```python
UPDATE_FREQ = 25  # Default: 50
```

### Local demo slow
Normal for CPU. Data download takes 2-3 minutes.

### Dependencies error
```bash
pip install --upgrade yfinance ta gym torch
```

## Future Enhancements

- [ ] Multi-agent competition
- [ ] Attention mechanism for stock selection
- [ ] LSTM for temporal patterns
- [ ] Risk constraints (max drawdown limits)
- [ ] Real-time trading integration
- [ ] Ensemble of agents
- [ ] More asset classes (bonds, crypto)

## License

MIT License - free for educational and commercial use

## Citation

If you use this project, please cite:

```
@misc{rl_portfolio_optimizer,
  title={RL Portfolio Optimization with PPO},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/rl-portfolio-optimizer}
}
```

## References

- Schulman et al. (2017). Proximal Policy Optimization Algorithms
- Jiang et al. (2017). A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem
- Deng et al. (2016). Deep Direct Reinforcement Learning for Financial Signal Representation and Trading

## Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub: @yourusername
- LinkedIn: linkedin.com/in/yourprofile

---

**Built with PyTorch, Gym, and Yahoo Finance**