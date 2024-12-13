# modules/chart_plotter.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
from modules.logger import BotLogger
import traceback

class ChartPlotter:
    def __init__(self, config):
        self.config = config
        self.chart_dir = 'trading_charts'
        os.makedirs(self.chart_dir, exist_ok=True)
        self.logger = BotLogger()
        self.last_live_candle_time = None


    def plot_historical_chart(self, five_min_data, positions, trade_history):
        """Plot historical data chart"""
        try:
            if len(five_min_data) < 2:
                self.logger.print_info("[yellow]Not enough historical data for charting[/yellow]")
                return

            self.logger.print_info(f"[cyan]Plotting historical chart with {len(five_min_data)} candles[/cyan]")
            
            # Save the last candle time
            self.last_live_candle_time = five_min_data.index[-1]
            
            # Create timestamp for historical chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'historical_chart_{timestamp}.png'
            
            # Create the chart
            self._create_chart(
                five_min_data, 
                positions, 
                trade_history, 
                filename,
                title=f'WIF/SOL Historical Data ({len(five_min_data)} candles)'
            )
            
            # Also save as latest historical
            latest_historical = os.path.join(self.chart_dir, 'latest_historical.png')
            if os.path.exists(latest_historical):
                os.remove(latest_historical)
            os.symlink(os.path.join(self.chart_dir, filename), latest_historical)
            
        except Exception as e:
            self.logger.print_error(f"Error plotting historical chart: {str(e)}")
            traceback.print_exc()

    def _create_chart(self, data, positions, trade_history, filename, title=None):
        """Create and save chart"""
        try:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot OHLC candlesticks
            dates = data.index
            ohlc_data = data[['open', 'high', 'low', 'close']].values
            
            # Calculate colors for candlesticks
            colors = ['g' if close >= open_ else 'r' 
                    for open_, close in zip(data['open'], data['close'])]
            
            # Plot candlesticks with label
            candlestick_plots = []
            for i, (date, (open_, high, low, close)) in enumerate(zip(dates, ohlc_data)):
                color = colors[i]
                body_height = close - open_
                candlestick = ax.bar(i, body_height, bottom=min(open_, close), 
                                    color=color, width=0.8, alpha=0.7)
                wick = ax.plot([i, i], [low, high], color=color, linewidth=1)
                if i == 0:  # Only add label for the first candlestick
                    candlestick_plots.append((candlestick, "Candlesticks"))
            
            # Plot Fibonacci levels and threshold
            if 'wma_fib_0' in data.columns:
                fib_line = data['wma_fib_0']
                threshold_line = fib_line * (1 + self.config['trading']['fib_entry_threshold'])
                
                # Plot Fib 0 level
                ax.plot(range(len(data)), fib_line, 
                    '--', color='yellow', label='Fib 0 Level', alpha=0.7)
                
                # Plot threshold
                ax.plot(range(len(data)), threshold_line, 
                    '--', color='red', 
                    label=f'Entry Threshold ({self.config["trading"]["fib_entry_threshold"]:.1%})', 
                    alpha=0.7)
                
                # Shade the entry zone
                ax.fill_between(range(len(data)), 
                            fib_line, threshold_line, 
                            color='purple', alpha=0.1,
                            label='Entry Zone')

            if 'wma_fib_50' in data.columns:
                ax.plot(range(len(data)), data['wma_fib_50'], 
                    '--', color='white', label='Fib 50 Level', alpha=0.7)
            
            # Plot trades
            if trade_history:
                for trade in trade_history:
                    if trade['entry_time'] in dates and trade['exit_time'] in dates:
                        entry_idx = dates.get_loc(trade['entry_time'])
                        exit_idx = dates.get_loc(trade['exit_time'])
                        
                        # Plot entry point
                        ax.scatter(entry_idx, trade['entry_price'], 
                                marker='^', color='lime', s=100,
                                label='Trade Entry' if entry_idx == 0 else "")
                        
                        # Plot exit point
                        ax.scatter(exit_idx, trade['exit_price'], 
                                marker='v', color='red', s=100,
                                label='Trade Exit' if exit_idx == 0 else "")
                        
                        # Connect entry and exit
                        ax.plot([entry_idx, exit_idx], 
                            [trade['entry_price'], trade['exit_price']], 
                            'g--', alpha=0.3)
            
            # Plot open positions
            for i, position in enumerate(positions):
                if position['entry_time'] in dates:
                    entry_idx = dates.get_loc(position['entry_time'])
                    ax.scatter(entry_idx, position['entry_price'], 
                            marker='^', color='yellow', s=100,
                            label='Open Position' if i == 0 else "")
            
            # Add trade statistics if available
            if trade_history:
                stats_text = self._get_trade_stats(trade_history)
                plt.figtext(0.02, 0.02, stats_text, fontsize=8, color='white')
            
            # Customize chart
            ax.set_xticks(range(0, len(dates), max(1, len(dates)//10)))
            ax.set_xticklabels([d.strftime('%Y-%m-%d %H:%M') for d in dates[::max(1, len(dates)//10)]], 
                            rotation=45)
            
            if title:
                plt.title(title)
            else:
                plt.title(f'WIF/SOL 5-Minute Chart ({len(data)} candles)')
            
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.2)
            
            # Add legend with only unique labels
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), 
                    loc='upper left', bbox_to_anchor=(1.05, 1))
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save chart
            filepath = os.path.join(self.chart_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.print_info(f"[green]Chart saved: {filepath}[/green]")
            
        except Exception as e:
            self.logger.print_error(f"Error in _create_chart: {str(e)}")
            traceback.print_exc()

    def _get_trade_stats(self, trade_history):
        """Calculate and format trade statistics"""
        if not trade_history:
            return "No trades yet"
        
        profits = [trade['profit'] for trade in trade_history]
        win_rate = (np.array(profits) > 0).mean() * 100
        avg_profit = np.mean(profits) * 100
        total_trades = len(trade_history)
        
        return (
            f"Trade Statistics:\n"
            f"Total Trades: {total_trades}\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Avg Profit: {avg_profit:.2f}%"
        )


    def plot_live_chart(self, five_min_data, positions, trade_history):
        """Plot live data chart"""
        try:
            if len(five_min_data) < 2:
                return
            
            # Only plot new data
            if self.last_live_candle_time:
                new_data = five_min_data[five_min_data.index > self.last_live_candle_time]
                if len(new_data) == 0:
                    return
                
                self.last_live_candle_time = five_min_data.index[-1]
            
            # Only show last 200 candles for live chart
            recent_data = five_min_data.tail(200)
            self._create_chart(recent_data, positions, trade_history, 'live_chart.png')
            
        except Exception as e:
            self.logger.print_error(f"Error plotting live chart: {str(e)}")


    def _plot_trades(self, ax, dates, trade_history, positions):
        """Plot trades on the chart"""
        # Plot closed trades
        for trade in trade_history:
            if trade['entry_time'] in dates and trade['exit_time'] in dates:
                entry_idx = dates.get_loc(trade['entry_time'])
                exit_idx = dates.get_loc(trade['exit_time'])
                
                ax.scatter(entry_idx, trade['entry_price'], 
                          marker='^', color='lime', s=100)
                ax.scatter(exit_idx, trade['exit_price'], 
                          marker='v', color='red', s=100)
                ax.plot([entry_idx, exit_idx], 
                       [trade['entry_price'], trade['exit_price']], 
                       'g--', alpha=0.3)
        
        # Plot open positions
        for position in positions:
            if position['entry_time'] in dates:
                entry_idx = dates.get_loc(position['entry_time'])
                ax.scatter(entry_idx, position['entry_price'], 
                          marker='^', color='yellow', s=100, label='Open Position')