import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from data_handler import ForexDataHandler
import os

class ForexDataVisualizer:
    def __init__(self, symbol: str = "USDJPY"):
        self.symbol = symbol
        self.data_handler = ForexDataHandler(symbol)

    def plot_data(self, year: int, save_path: str = None):
        """
        Plot candlestick chart with EMA indicators using plotly
        """
        # Get data for specified year
        df, _ = self.data_handler.prepare_data(train_year=year, test_year=year)
        
        if df is None:
            print(f"Failed to get data for {year}")
            return

        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f'{self.symbol} Price Chart - {year}', 'Volume'))

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add EMA traces
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema_fast'],
                name='EMA Fast',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema_slow'],
                name='EMA Slow',
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )

        # Add volume bar chart
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['tick_volume'],
                name='Volume'
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title_text=f"{self.symbol} Analysis - {year}",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive chart saved to {save_path}")

    def plot_multiple_timeframes(self, year: int):
        """
        Plot data for different timeframes to show market structure
        """
        # Get original 5-minute data
        df_5m, _ = self.data_handler.prepare_data(train_year=year, test_year=year)
        
        if df_5m is None:
            print(f"Failed to get data for {year}")
            return
            
        # Create daily and hourly resampled data
        df_1h = df_5m.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'tick_volume': 'sum'
        }).dropna()
        
        df_1d = df_5m.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'tick_volume': 'sum'
        }).dropna()
        
        # Calculate EMAs for resampled data
        df_1h = self.data_handler.calculate_ema(df_1h)
        df_1d = self.data_handler.calculate_ema(df_1d)
        
        # Plot each timeframe
        timeframes = [
            ('Daily', df_1d, f'{self.symbol}_daily_{year}.html'),
            ('Hourly', df_1h, f'{self.symbol}_hourly_{year}.html'),
            ('5-Minute', df_5m, f'{self.symbol}_5min_{year}.html')
        ]
        
        for title, df, filename in timeframes:
            if not df.empty:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  vertical_spacing=0.03,
                                  row_heights=[0.7, 0.3],
                                  subplot_titles=(f'{self.symbol} {title} Chart - {year}', 'Volume'))

                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price'
                    ),
                    row=1, col=1
                )

                # Add EMA traces
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['ema_fast'],
                        name='EMA Fast',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['ema_slow'],
                        name='EMA Slow',
                        line=dict(color='red', width=1)
                    ),
                    row=1, col=1
                )

                # Add volume bar chart
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['tick_volume'],
                        name='Volume'
                    ),
                    row=2, col=1
                )

                # Update layout
                fig.update_layout(
                    title_text=f"{self.symbol} {title} Analysis - {year}",
                    xaxis_rangeslider_visible=False,
                    height=800,
                    showlegend=True,
                    template='plotly_dark'
                )

                fig.write_html(filename)
                print(f"Saved {title} interactive chart to {filename}")

        # Cleanup
        self.data_handler.cleanup()

if __name__ == "__main__":
    # Create visualizer
    visualizer = ForexDataVisualizer()
    
    print("Generating interactive charts for 2023...")
    # Plot training data (2023)
    visualizer.plot_data(2023, "training_data_2023.html")
    visualizer.plot_multiple_timeframes(2023)
