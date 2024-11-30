import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class ForexDataHandler:
    def __init__(self, symbol: str = "EURUSD"):
        self.symbol = symbol
        self.server_name = None
        self.initialize_mt5()
        
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print("MT5 initialization failed")
            return False
        
        account_info = mt5.account_info()
        # Get server name and remove spaces, special characters
        self.server_name = account_info.server.split()[0]  # Get first part of server name
        
        print(f"MT5 Version:", mt5.__version__)
        print(f"Terminal Info:", mt5.terminal_info())
        print(f"Account Info:", account_info)
        return True
        
    def get_historical_data(self, 
                          year: int,
                          timeframe: int = mt5.TIMEFRAME_M5,
                          save_csv: bool = True) -> pd.DataFrame:
        """
        Get historical data for specified year
        timeframe: Default is 5 minutes (mt5.TIMEFRAME_M5)
        save_csv: If True, save data to CSV in data/raw folder
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year + 1, 1, 1)
        
        print(f"\nDownloading {self.symbol} data:")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        
        rates = mt5.copy_rates_range(
            self.symbol,
            timeframe,
            start_date,
            end_date
        )
        
        if rates is None:
            print(f"Failed to get data for {self.symbol}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        print(f"Downloaded {len(df)} records")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        
        if save_csv:
            self.save_to_csv(df, year, timeframe)
            
        df.set_index('time', inplace=True)
        return df
        
    def save_to_csv(self, df: pd.DataFrame, year: int, timeframe: int):
        """Save data to CSV file in data/raw folder"""
        # Create data/raw directory if not exists
        os.makedirs("data/raw", exist_ok=True)
        
        # Get timeframe string
        tf_dict = {
            mt5.TIMEFRAME_M5: "5min",
            mt5.TIMEFRAME_H1: "1hour",
            mt5.TIMEFRAME_D1: "daily"
        }
        tf_str = tf_dict.get(timeframe, "unknown")
        
        # Create filename with server name, symbol, timeframe and year
        filename = f"{self.server_name}_{self.symbol}_{tf_str}_{year}.csv"
        filepath = os.path.join("data", "raw", filename)
        
        # Save to CSV
        df.to_csv(filepath, index=True)
        print(f"Data saved to {filepath}")
        print(f"File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        
    def calculate_ema(self, 
                     df: pd.DataFrame, 
                     fast_period: int = 12, 
                     slow_period: int = 26) -> pd.DataFrame:
        """Calculate EMA indicators"""
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        return df
        
    def prepare_data(self, 
                    train_year: int = 2023, 
                    test_year: int = 2024,
                    fast_period: int = 12,
                    slow_period: int = 26) -> tuple:
        """
        Prepare training and testing datasets
        Returns: (train_df, test_df)
        """
        # Get training data
        train_df = self.get_historical_data(train_year)
        if train_df is not None:
            train_df = self.calculate_ema(train_df, fast_period, slow_period)
            
        # Get testing data
        test_df = self.get_historical_data(test_year)
        if test_df is not None:
            test_df = self.calculate_ema(test_df, fast_period, slow_period)
            
        return train_df, test_df
        
    def cleanup(self):
        """Cleanup MT5 connection"""
        mt5.shutdown()

def download_forex_data(symbols: list = ["EURUSD", "USDJPY"],
                       year: int = 2023,
                       timeframes: list = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_D1]):
    """
    Download forex data for multiple symbols and timeframes
    """
    for symbol in symbols:
        handler = ForexDataHandler(symbol=symbol)
        for timeframe in timeframes:
            handler.get_historical_data(year=year, timeframe=timeframe, save_csv=True)
        handler.cleanup()

if __name__ == "__main__":
    # Example: Download data for EURUSD and USDJPY for year 2023
    download_forex_data()
