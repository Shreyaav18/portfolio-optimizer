import numpy as np
import matplotlib.pyplot as plt

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_returns = returns - risk_free_rate / 252
    if np.std(excess_returns) == 0:
        return 0
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe

def calculate_max_drawdown(portfolio_values):
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    return max_drawdown

def plot_training_progress(episode_rewards, portfolio_values, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    axes[0].plot(np.convolve(episode_rewards, np.ones(10)/10, mode='valid'), 
                 linewidth=2, label='Moving Avg (10 eps)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(portfolio_values, linewidth=2)
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Portfolio Value ($)')
    axes[1].set_title('Portfolio Value Over Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def plot_backtest_results(agent_values, benchmark_values, tickers, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(agent_values, linewidth=2, label='RL Agent', color='green')
    axes[0].plot(benchmark_values, linewidth=2, label='Buy & Hold', color='blue', alpha=0.7)
    axes[0].set_xlabel('Days')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].set_title('RL Agent vs Buy & Hold Strategy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    agent_returns = (agent_values[-1] - agent_values[0]) / agent_values[0] * 100
    benchmark_returns = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0] * 100
    
    strategies = ['RL Agent', 'Buy & Hold']
    returns = [agent_returns, benchmark_returns]
    colors = ['green', 'blue']
    
    axes[1].bar(strategies, returns, color=colors, alpha=0.7)
    axes[1].set_ylabel('Total Return (%)')
    axes[1].set_title('Strategy Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(returns):
        axes[1].text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def print_performance_metrics(portfolio_values, initial_value, tickers):
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe = calculate_sharpe_ratio(daily_returns)
    max_dd = calculate_max_drawdown(portfolio_values)
    
    print("=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print("=" * 50)