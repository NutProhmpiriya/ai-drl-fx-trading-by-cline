from stable_baselines3 import PPO
from data_handler import ForexDataHandler
from forex_env import ForexTradingEnv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class ForexBacktester:
    def __init__(
        self,
        symbol: str = "EURUSD",
        model_path: str = "models",
        initial_balance: float = 10000.0
    ):
        self.symbol = symbol
        self.model_path = os.path.join(model_path, f"{symbol}_model")
        self.initial_balance = initial_balance
        
    def run_backtest(self) -> pd.DataFrame:
        """
        Run backtest on test data and return results
        """
        # Load data
        data_handler = ForexDataHandler(self.symbol)
        _, test_df = data_handler.prepare_data()
        
        if test_df is None:
            print("Failed to get test data")
            return None
            
        # Create environment
        env = ForexTradingEnv(test_df, initial_balance=self.initial_balance)
        
        # Load trained model
        model = PPO.load(self.model_path)
        
        # Initialize tracking variables
        trades = []
        daily_pnl = {}
        last_trade_balance = self.initial_balance
        
        # Run simulation
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            
            # Track trade
            if info['position'] != 0:
                trade_pnl = info['balance'] - last_trade_balance
                trades.append({
                    'timestamp': test_df.index[env.current_step],
                    'action': 'buy' if info['position'] > 0 else 'sell',
                    'position_size': abs(info['position']),
                    'balance': info['balance'],
                    'daily_pnl': info['daily_pnl'],
                    'trade_pnl': trade_pnl,
                    'is_win': trade_pnl > 0
                })
                last_trade_balance = info['balance']
                
            # Track daily PnL
            current_date = test_df.index[env.current_step].date()
            daily_pnl[current_date] = info['daily_pnl']
        
        # Cleanup
        data_handler.cleanup()
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Generate performance metrics
        metrics = self.calculate_performance_metrics(trades_df, daily_pnl)
        
        # Plot results
        self.plot_results(trades_df, daily_pnl)
        
        return trades_df, metrics
        
    def calculate_performance_metrics(
        self,
        trades_df: pd.DataFrame,
        daily_pnl: dict
    ) -> dict:
        """
        Calculate trading performance metrics
        """
        if trades_df.empty:
            return None
            
        # Calculate buy/sell win/loss metrics
        buy_trades = trades_df[trades_df['action'] == 'buy']
        sell_trades = trades_df[trades_df['action'] == 'sell']
        
        buy_wins = len(buy_trades[buy_trades['is_win'] == True])
        buy_losses = len(buy_trades[buy_trades['is_win'] == False])
        sell_wins = len(sell_trades[sell_trades['is_win'] == True])
        sell_losses = len(sell_trades[sell_trades['is_win'] == False])
        
        total_buy_trades = buy_wins + buy_losses
        total_sell_trades = sell_wins + sell_losses
        
        buy_win_rate = (buy_wins / total_buy_trades * 100) if total_buy_trades > 0 else 0
        sell_win_rate = (sell_wins / total_sell_trades * 100) if total_sell_trades > 0 else 0
        
        # Calculate other metrics
        total_trades = len(trades_df)
        final_balance = trades_df['balance'].iloc[-1] if not trades_df.empty else self.initial_balance
        total_return = ((final_balance - self.initial_balance) / self.initial_balance) * 100
        
        daily_returns = pd.Series(daily_pnl.values())
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 0 else 0
        
        max_drawdown = 0
        peak_balance = self.initial_balance
        
        for balance in trades_df['balance']:
            if balance > peak_balance:
                peak_balance = balance
            drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            'Total Trades': total_trades,
            'Buy Trades - Total': total_buy_trades,
            'Buy Trades - Wins': buy_wins,
            'Buy Trades - Losses': buy_losses,
            'Buy Win Rate (%)': buy_win_rate,
            'Sell Trades - Total': total_sell_trades,
            'Sell Trades - Wins': sell_wins,
            'Sell Trades - Losses': sell_losses,
            'Sell Win Rate (%)': sell_win_rate,
            'Final Balance': final_balance,
            'Total Return (%)': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100
        }
        
    def plot_results(
        self,
        trades_df: pd.DataFrame,
        daily_pnl: dict
    ):
        """
        Plot backtest results
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Account Balance
        plt.subplot(2, 1, 1)
        plt.plot(trades_df['timestamp'], trades_df['balance'])
        plt.title('Account Balance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.grid(True)
        
        # Plot 2: Daily PnL
        plt.subplot(2, 1, 2)
        dates = list(daily_pnl.keys())
        values = list(daily_pnl.values())
        plt.bar(dates, values)
        plt.title('Daily PnL')
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png')
        plt.close()

if __name__ == "__main__":
    # Run backtest
    backtester = ForexBacktester()
    trades_df, metrics = backtester.run_backtest()
    
    if metrics:
        print("\nBacktest Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
