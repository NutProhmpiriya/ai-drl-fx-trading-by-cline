from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from data_handler import ForexDataHandler
from forex_env import ForexTradingEnv
import os
from datetime import datetime

def train_model(
    symbol: str = "USDJPY",
    save_path: str = "models"
):
    """
    Train the DRL model using PPO algorithm
    """
    # Create models directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialize data handler and get training data
    data_handler = ForexDataHandler(symbol)
    train_df, _ = data_handler.prepare_data()
    
    if train_df is None:
        print("Failed to get training data")
        return None
    
    # Calculate total timesteps based on number of candles
    # For 5-minute candles in a year:
    # 24 hours * 12 (5-min intervals) * ~252 trading days
    num_candles = len(train_df)
    print(f"Number of candles in training data: {num_candles}")
    
    # Set total timesteps to cover the dataset multiple times for better learning
    total_timesteps = num_candles * 3  # Train on each candle ~3 times
    print(f"Setting total timesteps to: {total_timesteps}")
        
    # Create and wrap the trading environment
    env = ForexTradingEnv(train_df)
    env = DummyVecEnv([lambda: env])
    
    # Initialize the PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model with datetime
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_path, f"{symbol}_model_{current_time}")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Cleanup
    data_handler.cleanup()
    
    return model

if __name__ == "__main__":
    # Train the model
    model = train_model()
