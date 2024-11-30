# AI DRL Forex Trading System

A Deep Reinforcement Learning system for automated Forex trading using MT5, Python, Gymnasium, and Stable-Baselines3.

## Features

- Uses MT5 for real-time and historical Forex data
- Implements DRL using Gymnasium v1.0.0 and Stable-Baselines3 2.4.0
- 5-minute timeframe candlestick data
- Position sizing with risk management
- Maximum daily loss limit of 1%
- EMA-based trading signals
- Training on 2023 data and backtesting on 2024 data
- Advanced data visualization with candlestick charts and technical indicators

## Project Structure

- `requirements.txt` - Project dependencies
- `data_handler.py` - MT5 data handling and preprocessing
- `forex_env.py` - Custom Gymnasium environment for Forex trading
- `train.py` - DRL model training script
- `backtest.py` - Backtesting and performance analysis
- `visualize_data.py` - Price data visualization with technical indicators

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure MetaTrader 5 is installed and configured on your system

## Usage

1. Visualize price data:

```bash
python visualize_data.py
```

This will generate:

- Candlestick charts with EMA indicators for both 2023 and 2024
- Multi-timeframe analysis (Daily, Hourly, 5-minute) charts
- Volume analysis

2. Training the model:

```bash
python train.py
```

3. Running backtest:

```bash
python backtest.py
```

## Data Visualization

The visualization system (`visualize_data.py`) provides:

- Candlestick charts with EMA overlays
- Multiple timeframe analysis:
  - Daily charts for overall trend
  - Hourly charts for medium-term movements
  - 5-minute charts for detailed price action
- Volume analysis
- Automatic chart saving to PNG files

## Environment Details

The trading environment (`forex_env.py`) implements:

- State space: [balance, position, ema_fast, ema_slow, daily_pnl]
- Action space: [0: hold, 1: buy, 2: sell]
- Reward function based on trade PnL
- Daily loss limit of 1%
- Position sizing based on account balance

## Data Processing

The data handler (`data_handler.py`):

- Connects to MT5
- Downloads 5-minute timeframe data
- Calculates EMA indicators (12 and 26 periods)
- Prepares separate datasets for training (2023) and testing (2024)

## Backtesting

The backtesting system (`backtest.py`) provides:

- Detailed trade analysis
- Performance metrics:
  - Total return
  - Sharpe ratio
  - Maximum drawdown
  - Number of trades
- Visual results:
  - Account balance chart
  - Daily PnL chart

## Risk Management

The system implements several risk management features:

- Position sizing based on account balance
- Maximum daily loss limit of 1%
- Stop trading when daily loss limit is reached
- Account balance monitoring

## Notes

- Ensure your MT5 platform is running and logged in before using the system
- The system is configured for EURUSD by default but can be modified for other currency pairs
- Model training parameters can be adjusted in train.py
- Backtest parameters can be modified in backtest.py
- Visualization settings can be customized in visualize_data.py
