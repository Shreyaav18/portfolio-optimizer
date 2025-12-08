import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import prepare_data, split_data
from environment import PortfolioEnv
from agent import PPOAgent
from utils import calculate_sharpe_ratio, calculate_max_drawdown, print_performance_metrics

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
START_DATE = '2018-01-01'
END_DATE = '2024-01-01'
INITIAL_BALANCE = 10000
MODEL_PATH = 'saved_models/ppo_agent_final.pth'

print("Loading stock data...")
df = prepare_data(TICKERS, START_DATE, END_DATE)
train_df, test_df = split_data(df, train_ratio=0.8)

test_env = PortfolioEnv(test_df, initial_balance=INITIAL_BALANCE)

obs_dim = test_env.observation_space.shape[0]
action_dim = test_env.action_space.shape[0]

agent = PPOAgent(obs_dim, action_dim)

print(f"Loading trained model from {MODEL_PATH}...")
agent.load(MODEL_PATH)

print("\nRunning backtest on test data...")
state = test_env.reset()
done = False
portfolio_values = [test_env.portfolio_value]
actions_log = []

while not done:
    action, _, _ = agent.get_action(state, deterministic=True)
    state, reward, done, _ = test_env.step(action)
    
    portfolio_values.append(test_env.portfolio_value)
    actions_log.append(action)

portfolio_values = np.array(portfolio_values)

print_performance_metrics(portfolio_values, INITIAL_BALANCE, TICKERS)

equal_weight = INITIAL_BALANCE / len(TICKERS)
buy_hold_values = []

for i in range(len(test_df)):
    prices = [test_df.iloc[i, j*5] for j in range(len(TICKERS))]
    initial_prices = [test_df.iloc[0, j*5] for j in range(len(TICKERS))]
    
    value = sum([(equal_weight / initial_prices[j]) * prices[j] for j in range(len(TICKERS))])
    buy_hold_values.append(value)

buy_hold_values = np.array(buy_hold_values)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(portfolio_values, linewidth=2, label='RL Agent', color='green')
axes[0, 0].plot(buy_hold_values, linewidth=2, label='Buy & Hold', color='blue', alpha=0.7)
axes[0, 0].axhline(y=INITIAL_BALANCE, color='red', linestyle='--', alpha=0.5, label='Initial')
axes[0, 0].set_xlabel('Days')
axes[0, 0].set_ylabel('Portfolio Value ($)')
axes[0, 0].set_title('Portfolio Performance Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

agent_return = (portfolio_values[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100
buyhold_return = (buy_hold_values[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100

strategies = ['RL Agent', 'Buy & Hold']
returns = [agent_return, buyhold_return]
colors = ['green', 'blue']

bars = axes[0, 1].bar(strategies, returns, color=colors, alpha=0.7)
axes[0, 1].set_ylabel('Total Return (%)')
axes[0, 1].set_title('Strategy Returns')
axes[0, 1].grid(True, alpha=0.3, axis='y')

for i, v in enumerate(returns):
    axes[0, 1].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')

daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
axes[1, 0].hist(daily_returns * 100, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Daily Return (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Daily Returns')
axes[1, 0].grid(True, alpha=0.3, axis='y')

cumulative_max = np.maximum.accumulate(portfolio_values)
drawdown = (portfolio_values - cumulative_max) / cumulative_max * 100

axes[1, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
axes[1, 1].plot(drawdown, linewidth=2, color='red')
axes[1, 1].set_xlabel('Days')
axes[1, 1].set_ylabel('Drawdown (%)')
axes[1, 1].set_title('Portfolio Drawdown')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nGenerating trading decisions log...")
print("\nSample Trading Actions (First 10 days):")
print("-" * 80)
print(f"{'Day':<6} {'Action Vector':<50} {'Portfolio Value':<20}")
print("-" * 80)

for i in range(min(10, len(actions_log))):
    action_str = np.array2string(actions_log[i], precision=2, suppress_small=True)
    print(f"{i+1:<6} {action_str:<50} ${portfolio_values[i+1]:,.2f}")

print("\nBacktest complete! Results saved to 'backtest_results.png'")