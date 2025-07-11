import asyncio
import time
import numpy as np
from datetime import datetime
import pandas as pd
import typer
from pathlib import Path
import traceback
import os
import threading # For running LiveChart in a separate thread
import pytz # For timezone awareness, consistent with DataHandler and LiveChart

from modules.backtest_manager import BacktestManager # Not directly used in live mode, but kept
from modules.data_handler import DataHandler
from modules.trade_manager import TradeManager
from modules.position_manager import PositionManager
from modules.indicators import FibonacciCalculator
from utils.helpers import load_config, format_timestamp # Assuming format_timestamp is for logging
from modules.logger import BotLogger
from modules.live_chart import LiveChart

import signal

class TradingBot:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the trading bot with all its components"""
        self.config = load_config(config_path)
        self.logger = BotLogger() # Initialize BotLogger instance here
        self.shutdown_event = asyncio.Event()
        
        # Initialize components
        self.data_handler = DataHandler()
        # Initialize DataHandler from CSV. This will populate initial 5-min data and set last_read_file_size/timestamp.
        # This initial load is crucial for the bot to have enough historical data for indicators.
        self.data_handler.initialize_from_csv(self.config['paths']['live_data'])
        
        self.fib_calculator = FibonacciCalculator(self.config)
        self.position_manager = PositionManager(self.config)
        self.running = False

        self.tasks = []
        
        # Initialize LiveChart instance
        self.live_chart = LiveChart(config=self.config, port=8051) # Ensure this matches your LiveChart's port

        # Trade manager needs position manager and fib calculator
        self.trade_manager = TradeManager(
            self.config,
            self.position_manager,
            self.fib_calculator
        )
        
        # Initialize event loop for async operations
        try:
            self.loop = asyncio.get_event_loop()
            if self.loop.is_closed():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
        except RuntimeError: # Handle case where no loop is set
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        # Status tracking
        self.last_status_time = time.time()
        self.status_interval = 300  # 5 minutes, but console status updates faster
        self.chart_update_counter = 0 # Not directly used in current flow but kept

    async def start(self):
        """Start the trading bot"""
        try:
            self.running = True
            # CRITICAL FIX: Call start_ui *before* any other logging attempts
            self.logger.start_ui() 
            self.logger.print_info("[bold green]Initializing Trading Bot...[/bold green]")

            # CRITICAL FIX: Start LiveChart in a separate thread
            self.chart_thread = threading.Thread(
                target=self.live_chart.start
            )
            self.chart_thread.daemon = True # Allows main program to exit even if thread is running
            self.chart_thread.start()
            self.logger.print_info(f"[bold green]Live Chart server started on http://0.0.0.0:{self.live_chart.port}[/bold green]")

            # Start live market monitoring task
            # The check_interval here defines how often the *file size* is checked.
            # Actual data processing happens as new lines are detected.
            market_monitor_task = self.loop.create_task(self.monitor_market(self.config['paths']['live_data'], check_interval=5)) # Check file every 5 seconds
            self.tasks.append(market_monitor_task)
            
            # Initial status print to console
            self.logger.print_status()
            self.position_manager.print_portfolio_status(self.data_handler.get_current_price())
            
            # Wait until shutdown_event is set (bot runs until this event is set, e.g., by CTRL+C)
            await self.shutdown_event.wait()

        except KeyboardInterrupt:
            self.logger.print_info("[bold red]Shutting down gracefully...[/bold red]")
            await self.cleanup()
        except Exception as e:
            self.logger.print_error(f"Error in main loop: {str(e)}")
            traceback.print_exc() # Print full traceback for unexpected errors
            await self.cleanup()

    async def handle_shutdown(self):
        """Handle shutdown signal (e.g., SIGINT/SIGTERM)"""
        self.logger.print_info("[bold red]Shutdown signal received. Cleaning up...[/bold red]")
        self.running = False
        self.shutdown_event.set() # Signal main loop to exit
        await self.cleanup() # Perform cleanup

    # run_backtest is commented out as per your setup, keeping it for context but not active
    # async def run_backtest(self, historical_data):
    #     """Run backtest on historical data before live trading (currently commented out in start)"""
    #     # ... (existing backtest logic) ...

    def print_backtest_results(self):
        """Print detailed backtest results to console (mainly for backtest mode)"""
        # ... (existing backtest print logic) ...
        pass # Placeholder if backtest is fully disabled

    async def monitor_market(self, csv_path, check_interval=5): # Reduced check_interval to 5s for faster response
        """Monitor market data from CSV file, checking for new data periodically."""
        try:
            self.logger.print_info(f"[bold blue]Starting market monitoring... (watching {csv_path})[/bold blue]")

            if not os.path.exists(csv_path):
                self.logger.print_error(f"CSV file not found: {csv_path}")
                return

            # Initial data load and processing of any existing data in the CSV
            try:
                self.logger.print_info("[green]Performing initial data load from CSV...[/green]")
                new_data_processed_initial = await self.process_new_data(csv_path) 
                if new_data_processed_initial:
                     self.logger.print_info("[green]Initial data load complete[/green]")
                else:
                    self.logger.print_info("[yellow]No new data found during initial load (CSV might be empty or already processed).[/yellow]")
            except Exception as e:
                self.logger.print_error(f"Error during initial data load: {str(e)}")
                traceback.print_exc()
                return

            # Continue monitoring for new data in a loop
            while self.running:
                try:
                    current_time = time.time()

                    # Update console status display more frequently (e.g., every 5 seconds)
                    if current_time - self.last_status_time >= 5: 
                        try:
                            current_price = self.data_handler.get_current_price()
                            positions = self.position_manager.positions
                            balance = self.position_manager.balance_sol

                            self.logger.update_status(
                                current_price=current_price,
                                positions=positions,
                                balance=balance
                            )
                            self.last_status_time = current_time
                        except Exception as e:
                            self.logger.print_error(f"Status update error: {str(e)}\nTraceback: {traceback.format_exc()}")

                    # Check for new data in CSV file (by file size)
                    try:
                        if os.path.exists(csv_path):
                            current_file_size = os.path.getsize(csv_path)
                            
                            # Only process if the file size has increased (new data has been written)
                            if current_file_size > self.data_handler.last_read_file_size:
                                self.logger.print_info(f"Monitor: Detected new data (size increased from {self.data_handler.last_read_file_size} to {current_file_size}). Processing...")
                                new_data_processed = await self.process_new_data(csv_path)
                                if new_data_processed:
                                    self.data_handler.last_read_file_size = current_file_size 
                                    self.logger.print_info("[cyan]Monitor: Processed new market data from CSV.[/cyan]")
                                else:
                                    self.logger.print_warning("[yellow]Monitor: process_new_data returned False, no new data processed (maybe duplicate timestamps).[/yellow]")
                            # Removed the 'else' block here as it caused redundant processing if file size didn't change
                        else:
                            self.logger.print_error(f"CSV file not found during monitoring: {csv_path}. Please check path.")

                    except Exception as e:
                        self.logger.print_error(f"Monitor: Error checking/processing new data from CSV: {str(e)}")
                        traceback.print_exc()

                    await asyncio.sleep(check_interval) # Wait for the next file size check

                except asyncio.CancelledError:
                    raise # Propagate cancellation
                except Exception as e:
                    self.logger.print_error(f"Unexpected error in market monitoring loop: {str(e)}")
                    traceback.print_exc()
                    await asyncio.sleep(check_interval) # Wait before retrying after error

        except asyncio.CancelledError:
            self.logger.print_info("[yellow]Market monitoring stopped.[/yellow]")
            raise
        except Exception as e:
            self.logger.print_error(f"Fatal error in market monitoring: {str(e)}")
            traceback.print_exc()
        finally:
            self.logger.print_info("[yellow]Market monitoring finished.[/yellow]")

    async def process_new_data(self, csv_path):
        """
        Reads new data from the CSV file based on the last processed timestamp,
        processes it through the DataHandler, calculates indicators, and updates components.
        """
        try:
            if not os.path.exists(csv_path):
                self.logger.print_error(f"CSV file not found: {csv_path}")
                return False

            full_df = pd.read_csv(
                csv_path,
                header=0,
                parse_dates=['timestamp'],
                dtype={'price': float}
            )
            # Ensure 'timestamp' and 'price' columns exist
            if 'timestamp' not in full_df.columns or 'price' not in full_df.columns:
                self.logger.print_error("Process: CSV file must contain 'timestamp' and 'price' columns.")
                return False

            # Ensure all timestamps are timezone-aware (UTC) for consistent comparison
            if full_df['timestamp'].dt.tz is None:
                full_df['timestamp'] = full_df['timestamp'].dt.tz_localize(pytz.utc)
            else:
                full_df['timestamp'] = full_df['timestamp'].dt.tz_convert(pytz.utc)

            new_data_df = pd.DataFrame()
            if self.data_handler.last_processed_timestamp:
                # Filter for data strictly newer than the last processed timestamp
                last_ts_dt = self.data_handler.last_processed_timestamp.tz_convert(pytz.utc)
                new_data_df = full_df[full_df['timestamp'] > last_ts_dt].copy()
                self.logger.print_info(f"Process: Filtering for data > {last_ts_dt}. Found {len(new_data_df)} new rows.")
            else:
                # If no last_processed_timestamp (first run), process all data
                new_data_df = full_df.copy()
                self.logger.print_info(f"Process: No last_processed_timestamp, processing all {len(new_data_df)} rows.")

            if new_data_df.empty:
                self.logger.print_info("[yellow]Process: No truly new data to process after timestamp filtering.[/yellow]")
                return False

            new_data_df = new_data_df.sort_values(by='timestamp').reset_index(drop=True)

            self.logger.print_info(f"[cyan]Processing {len(new_data_df)} new price updates...[/cyan]")
            processed_count = 0

            # Process each new 1-minute price point
            for idx, row in new_data_df.iterrows():
                try:
                    timestamp = row['timestamp']
                    price = row['price']

                    # Update LiveChart with current price for every 1-min data point
                    # fib_data_for_chart is not available here yet, so pass None for now.
                    # It will be updated later when a new 5-min candle is formed.
                    self.live_chart.update_data({'timestamp': timestamp, 'price': price}, fib_data=None) 
                    self.logger.print_info(f"Processing 1-min: Time={timestamp}, Price={price:.8f}")

                    # Process the price update through DataHandler to aggregate into 5-min candles
                    new_candle, bucket = self.data_handler.process_new_price(timestamp, price)
                    processed_count += 1

                    if new_candle is not None:
                        self.logger.print_info(f"Process Loop: New 5-min candle formed at {new_candle.index[0]}. Close: {new_candle['close'].iloc[-1]:.8f}")
                        
                        # Update five minute data in DataHandler and get the updated DataFrame.
                        # This also ensures the data_handler.five_min_data is up-to-date with the new candle.
                        self.data_handler.five_min_data = self.data_handler.update_data(new_candle) 
                        
                        # Calculate fib levels based on the updated five_min_data
                        # fib_levels_calculated now holds the DataFrame with OHLC + original fibs + WMA fibs
                        fib_levels_calculated = self.fib_calculator.calculate_fib_levels(self.data_handler.five_min_data)

                        # CRITICAL FIX: Update DataHandler's internal DataFrame with the one containing Fib levels
                        # This ensures trade_manager receives data with fibs.
                        self.data_handler.five_min_data = fib_levels_calculated 

                        # Prepare fib_data for LiveChart (assuming it expects individual values for latest levels)
                        fib_data_for_chart = {
                            'Fib 0': fib_levels_calculated['wma_fib_0'].iloc[-1] if 'wma_fib_0' in fib_levels_calculated.columns and not pd.isna(fib_levels_calculated['wma_fib_0'].iloc[-1]) else None,
                            'Fib 50': fib_levels_calculated['wma_fib_50'].iloc[-1] if 'wma_fib_50' in fib_levels_calculated.columns and not pd.isna(fib_levels_calculated['wma_fib_50'].iloc[-1]) else None,
                            'Entry Threshold': (fib_levels_calculated['wma_fib_0'].iloc[-1] * (1 + self.config['trading']['fib_entry_threshold'])) if 'wma_fib_0' in fib_levels_calculated.columns and not pd.isna(fib_levels_calculated['wma_fib_0'].iloc[-1]) else None
                        }
                        
                        # Update LiveChart with the completed candle and its fib levels.
                        self.live_chart.update_data(new_candle.iloc[0], fib_data_for_chart)

                        # --- Update last_candle_below_fib for TradeManager ---
                        # This should reflect the state of the *previous* candle.
                        # Ensure enough data exists for the *previous* candle's WMA Fib 0 to be valid.
                        if len(self.data_handler.five_min_data) >= 2 and \
                           'wma_fib_0' in self.data_handler.five_min_data.columns and \
                           not pd.isna(self.data_handler.five_min_data['wma_fib_0'].iloc[-2]): # Check for NaN in prev WMA Fib 0
                            
                            prev_candle_close = self.data_handler.five_min_data['close'].iloc[-2]
                            prev_wma_fib_0 = self.data_handler.five_min_data['wma_fib_0'].iloc[-2]
                            self.trade_manager.last_candle_below_fib = (prev_candle_close <= prev_wma_fib_0)
                            self.logger.print_info(f"DEBUG(Main): TradeManager.last_candle_below_fib set to: {self.trade_manager.last_candle_below_fib} (Prev Close: {prev_candle_close:.4f}, Prev WMA Fib 0: {prev_wma_fib_0:.4f})")
                        else:
                            self.trade_manager.last_candle_below_fib = False # Default if not enough valid history
                            self.logger.print_info("DEBUG(Main): Not enough valid data for previous candle's WMA Fib 0 or it's NaN. Setting last_candle_below_fib to False.")

                        # Check for trades (now self.data_handler.five_min_data should contain the WMA Fibs)
                        self.trade_manager.check_for_trades(
                            new_candle.iloc[0], # Pass the specific candle row
                            self.data_handler.five_min_data # Pass the full 5-min data for context
                        )
                        
                        # Update live chart with latest trades and positions separately
                        self.live_chart.update_trades(
                            trades=self.position_manager.trade_history
                        )
                        self.live_chart.update_positions( 
                            positions=self.position_manager.positions
                        )
                        
                        # Update trade statistics
                        stats = self.position_manager.get_trade_statistics()
                        self.live_chart.update_trade_statistics(stats)
                        
                        # Update armed status
                        is_armed = self.trade_manager.entry_switch_armed
                        self.live_chart.update_is_armed_status(is_armed)
                        
                        # Update console status display (logger)
                        self.logger.update_status(
                            current_price=price, # Use the last processed 1-min price
                            positions=self.position_manager.positions,
                            balance=self.position_manager.balance_sol
                        )
                        # await asyncio.sleep(0.3) #uncomment for step-through
                    else:
                        self.logger.print_info(f"Process Loop: No new 5-min candle formed yet for price at {timestamp}.")

                except ValueError as e:
                    self.logger.print_error(f"Invalid data at row {idx}: {str(e)}. Skipping row.")
                    continue
                except Exception as e:
                    self.logger.print_error(f"Error processing row {idx} (timestamp: {row['timestamp'] if 'timestamp' in row else 'N/A'}, price: {row['price'] if 'price' in row else 'N/A'}): {str(e)}")
                    traceback.print_exc()
                    continue

            if processed_count > 0:
                # Update the DataHandler's last_processed_timestamp to prevent reprocessing
                # This should be the timestamp of the very last processed row in new_data_df
                self.data_handler.last_processed_timestamp = new_data_df['timestamp'].iloc[-1]
                self.logger.print_info(f"Processed {processed_count} new rows. Last processed timestamp: {self.data_handler.last_processed_timestamp}")
                return True
            else:
                return False

        except Exception as e:
            self.logger.print_error(f"Error processing new data: {str(e)}")
            traceback.print_exc()
            return False

    async def update_status(self):
        """Update bot status display (to console logger)"""
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_price = self.data_handler.get_current_price() # Get latest price from DataHandler

        status_info = f"""
        [bold white]=== Bot Status Update ===[/bold white]
        Time: {current_time_str}
        Current Price: {current_price:.8f}
        Open Positions: {len(self.position_manager.positions)}
        Balance: {self.position_manager.balance_sol:.3f} SOL
        Total Trades: {len(self.position_manager.trade_history)}
        """
        self.logger.print_info(status_info)

    async def cleanup(self):
        """Cleanup resources before shutdown"""
        if not self.running:
            return

        self.running = False
        self.logger.print_info("[bold yellow]Performing cleanup...[/bold yellow]")

        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.print_error(f"Error cancelling task: {e}")

        # Save trade history
        self.position_manager.save_trade_history()

        # Final status update to console
        await self.update_status()

        self.logger.print_info("[bold green]Cleanup completed[/bold green]")
        self.logger.stop() # Stop the console logger UI

def main():
    """Main entry point with Typer CLI"""
    app = typer.Typer(help="Trading Bot CLI")

    @app.callback()
    def callback():
        """
        Trading Bot - A cryptocurrency trading bot CLI
        """
        pass

    @app.command(help="Start the trading bot")
    def start(
        config_path: str = typer.Option(
            "config/config.yaml",
            "--config",
            "-c",
            help="Path to configuration file"
        ),
        interval: int = typer.Option( # This interval is for monitor_market, not general processing loop
            60,
            "--interval",
            "-i",
            help="Check interval in seconds for new CSV data"
        ),
    ):
        """Start the trading bot with specified configuration"""
        bot = TradingBot(config_path)

        async def signal_handler(sig):
            await bot.handle_shutdown()

        try:
            # Set up signal handlers for graceful shutdown (Ctrl+C, etc.)
            for sig in (signal.SIGINT, signal.SIGTERM):
                bot.loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(signal_handler(s)))

            bot.loop.run_until_complete(bot.start())

        except KeyboardInterrupt:
            bot.logger.print_info("[bold red]Bot shutdown initiated by user[/bold red]")
            bot.loop.run_until_complete(bot.cleanup()) # Ensure cleanup on direct KeyboardInterrupt
        finally:
            if not bot.loop.is_closed():
                bot.loop.close()

    app() # Run the Typer CLI application

if __name__ == "__main__":
    main()