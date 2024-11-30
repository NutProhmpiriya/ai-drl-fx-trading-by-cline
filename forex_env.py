import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Tuple, Dict, Any

class ForexTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        super(ForexTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.daily_pnl = 0
        self.last_trade_day = None
        
        # Define action space (0: hold, 1: buy, 2: sell)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # [balance, position, ema_fast, ema_slow, daily_pnl]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(5,), 
            dtype=np.float32
        )

    def calculate_position_size(self) -> float:
        """Calculate position size based on risk management rules"""
        risk_per_trade = 0.01  # 1% risk per trade
        available_balance = self.balance * (1 - abs(self.daily_pnl/self.initial_balance))
        position_size = (available_balance * risk_per_trade)
        return position_size

    def calculate_ema_signals(self) -> Tuple[float, float]:
        """Calculate EMA signals"""
        current_data = self.df.iloc[self.current_step]
        return current_data['ema_fast'], current_data['ema_slow']

    def get_observation(self) -> np.ndarray:
        """Get current state observation"""
        ema_fast, ema_slow = self.calculate_ema_signals()
        
        obs = np.array([
            self.balance,
            self.position,
            ema_fast,
            ema_slow,
            self.daily_pnl
        ], dtype=np.float32)
        
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        current_price = self.df.iloc[self.current_step]['close']
        current_date = self.df.iloc[self.current_step].name.date()
        
        # Reset daily PnL on new day
        if self.last_trade_day is None:
            self.last_trade_day = current_date
        elif current_date != self.last_trade_day:
            self.daily_pnl = 0
            self.last_trade_day = current_date
            
        reward = 0
        position_size = self.calculate_position_size()
        
        # Execute trading action
        if action == 1:  # Buy
            if self.position <= 0:
                if self.position < 0:
                    reward += self.position * (self.previous_price - current_price)
                self.position = position_size
                self.previous_price = current_price
                
        elif action == 2:  # Sell
            if self.position >= 0:
                if self.position > 0:
                    reward += self.position * (current_price - self.previous_price)
                self.position = -position_size
                self.previous_price = current_price
                
        # Calculate PnL
        if self.position != 0:
            trade_pnl = self.position * (current_price - self.previous_price)
            reward += trade_pnl
            self.daily_pnl += trade_pnl
            self.balance += trade_pnl
            
        # Check daily loss limit
        if self.daily_pnl <= -0.01 * self.initial_balance:
            done = True
            reward -= 100  # Penalty for exceeding daily loss limit
            
        # Check bankruptcy
        if self.balance <= 0:
            done = True
            reward -= 200  # Penalty for bankruptcy
            
        info = {
            'balance': self.balance,
            'daily_pnl': self.daily_pnl,
            'position': self.position
        }
        
        return self.get_observation(), reward, done, False, info

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.daily_pnl = 0
        self.last_trade_day = None
        
        return self.get_observation(), {}
