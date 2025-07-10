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

from modules.backtest_manager import BacktestManager
from modules.data_handler import DataHandler
from modules.trade_manager import TradeManager
from modules.position_manager import PositionManager
from modules.indicators import FibonacciCalculator
from utils.helpers import load_config, format_timestamp
from modules.logger import BotLogger
from modules.live_chart import LiveChart

import signal

class TradingBot:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the trading bot with all its components"""
        self.config = load_config(config_path)
        self.logger = BotLogger()
        self.shutdown_event = asyncio.Event()
        # self.last_processed_timestamp is now managed by DataHandler
        
        # Initialize components
        self.data_handler = DataHandler()
        # Initialize DataHandler from CSV. This will populate initial 5-min data and set last_read_file_size/timestamp.
        self.data_handler.initialize_from_csv(self.config['paths']['live_data'])
        
        self.fib_calculator = FibonacciCalculator(self.config)
        self.position_manager = PositionManager(self.config)
        self.running = False

        self.tasks = []
        
        # Initialize LiveChart instance
        self.live_chart = LiveChart(port=8051) # Ensure this matches your LiveChart's port

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
        self.status_interval = 300  # 5 minutes
        self.chart_update_counter = 0 # Not directly used in current flow but kept

    async def start(self):
        """Start the trading bot"""
        try:
            self.running = True
            self.logger.start_ui()
            self.logger.print_info("[bold green]Initializing Trading Bot...[/bold green]")

            # CRITICAL FIX: Start LiveChart in a separate thread
            # The Dash app's run() method (called by self.live_chart.start()) is blocking.
            # Run it in a new daemon thread so the main asyncio loop can continue.
            self.chart_thread = threading.Thread(
                target=self.live_chart.start # Call the start method of LiveChart instance
            )
            self.chart_thread.daemon = True # Allows main program to exit even if thread is running
            self.chart_thread.start()
            self.logger.print_info(f"[bold green]Live Chart server started on http://0.0.0.0:{self.live_chart.port}[/bold green]")


            """ # Load historical data (commented out as per your original code)
                historical_data = pd.read_csv(
                    self.config['paths']['historical_data'],
                    parse_dates=['timestamp'],
                    index_col='timestamp'
                )
                await self.run_backtest(historical_data)
            """
            # Start websocket server for live chart (commented out as per your original code)
            # """  websocket_task = self.loop.create_task(self.websocket_handler.start_server())
            #      self.tasks.append(websocket_task) """

            # Start live market monitoring task
            market_monitor_task = self.loop.create_task(self.monitor_market(self.config['paths']['live_data']))
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
            await self.cleanup()

    async def handle_shutdown(self):
        """Handle shutdown signal (e.g., SIGINT/SIGTERM)"""
        self.logger.print_info("[bold red]Shutdown signal received. Cleaning up...[/bold red]")
        self.running = False
        self.shutdown_event.set() # Signal main loop to exit
        await self.cleanup() # Perform cleanup

    async def run_backtest(self, historical_data):
        """Run backtest on historical data before live trading (currently commented out in start)"""
        self.logger.print_info("[bold blue]Starting Backtest Phase...[/bold blue]")
        try:
            # Reset any existing state
            self.position_manager.reset()
            self.data_handler.reset()

            # Process historical data
            for idx, row in historical_data.iterrows():
                try:
                    candle = row.copy()
                    candle.name = idx  # Set the timestamp as name

                    # Update five minute data in DataHandler and calculate fibonacci levels
                    updated_data = self.fib_calculator.calculate_fib_levels(
                        self.data_handler.update_data(pd.DataFrame([candle])) # Pass candle as DataFrame
                    )
                    self.data_handler.five_min_data = updated_data # Update data_handler's internal state

                    # Check for trades
                    self.trade_manager.check_for_trades(candle, self.data_handler.five_min_data)
                    
                    # Update live chart if running
                    if hasattr(self, 'live_chart'):
                        # Prepare fib_data for LiveChart (assuming it expects individual values for latest levels)
                        fib_data = {
                            'Fib 0': updated_data['wma_fib_0'].iloc[-1] if 'wma_fib_0' in updated_data.columns else None,
                            'Fib 50': updated_data['wma_fib_50'].iloc[-1] if 'wma_fib_50' in updated_data.columns else None,
                            'Entry Threshold': (updated_data['wma_fib_0'].iloc[-1] * (1 + self.config['trading']['fib_entry_threshold'])) if 'wma_fib_0' in updated_data.columns else None
                        }
                        # Pass the completed candle (Series) and fib_data to LiveChart.update_data
                        self.live_chart.update_data(candle, fib_data) 
                        
                        # Update live chart with latest trades and positions
                        self.live_chart.update_trades(trades=self.position_manager.trade_history)
                        self.live_chart.update_positions(positions=self.position_manager.positions) # Separate call
                        
                        # Update trade statistics and armed status
                        stats = self.position_manager.get_trade_statistics()
                        self.live_chart.update_trade_statistics(stats)
                        is_armed = self.trade_manager.entry_switch_armed
                        self.live_chart.update_is_armed_status(is_armed)

                    self.logger.print_info(f"[bold green]Processed candle at {idx}[/bold green]")
                except Exception as candle_e:
                    self.logger.print_error(f"Error processing candle at {idx}: {str(candle_e)}")
                    traceback.print_exc()
                    continue
            # Print final backtest results
            self.print_backtest_results()

            self.logger.print_info("[bold green]Backtest Complete - Transitioning to Live Trading[/bold green]")

        except Exception as e:
            self.logger.print_error(f"Error in backtest: {str(e)}")
            traceback.print_exc()

    def print_backtest_results(self):
        """Print detailed backtest results to console"""
        trades = self.position_manager.trade_history
        if not trades:
            self.logger.print_info("No trades during backtest period")
            return

        # Calculate statistics
        profits = [trade['profit'] for trade in trades]
        sol_profits = [trade['sol_profit'] for trade in trades]
        win_rate = (np.array(profits) > 0).mean() * 100

        stats = f"""
        [bold cyan]═══ Backtest Results ═══[/bold cyan]
        Total Trades: {len(trades)}
        Win Rate: {win_rate:.2f}%
        Average Profit: {np.mean(profits):.2%}
        Total SOL Profit: {sum(sol_profits):.3f}

        Best Trade: {max(profits):.2%}
        Worst Trade: {min(profits):.2%}

        Exit Types:
        Target 1: {sum(1 for t in trades if t['exit_type'] == 'target_1')}
        Target 2: {sum(1 for t in trades if t['exit_type'] == 'target_2')}
        Cumulative: {sum(1 for t in trades if t['exit_type'] == 'cumulative_target')}
        Average Hold Time: {np.mean([t['candles_held'] for t in trades if 'candles_held' in t]):.1f} candles
        """

        self.logger.print_info(stats)

        # Save detailed trade log to CSV
        df = pd.DataFrame(trades)
        df.to_csv('backtest_trades.csv', index=False)
        self.logger.print_info("Detailed trade log saved to 'backtest_trades.csv'")


    async def monitor_market(self, csv_path, check_interval=60):
        """Monitor market data from CSV file, checking for new data periodically."""
        try:
            self.logger.print_info(f"[bold blue]Starting market monitoring... ({csv_path})[/bold blue]")

            if not os.path.exists(csv_path):
                self.logger.print_error(f"CSV file not found: {csv_path}")
                return

            # Initial data load (processes existing data and prepares DataHandler for live updates)
            try:
                await self.process_new_data(csv_path) 
                self.logger.print_info("[green]Initial data load complete[/green]")
            except Exception as e:
                self.logger.print_error(f"Error during initial data load: {str(e)}")
                traceback.print_exc()
                return

            # Continue monitoring for new data in a loop
            while self.running:
                try:
                    current_time = time.time()

                    # Update console status display every 5 seconds
                    if current_time - self.last_status_time >= 5: 
                        try:
                            current_price = self.data_handler.get_current_price() # Get latest price from DataHandler

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

                    # Check for new data in CSV file
                    try:
                        if os.path.exists(csv_path):
                            current_file_size = os.path.getsize(csv_path)
                            
                            # Use last_read_file_size from DataHandler
                            # Only process if the file size has increased (new data has been written)
                            if current_file_size > self.data_handler.last_read_file_size:
                                self.logger.print_info(f"Monitor: Detected new data (size increased from {self.data_handler.last_read_file_size} to {current_file_size}). Processing...")
                                new_data_processed = await self.process_new_data(csv_path)
                                if new_data_processed:
                                    # Update DataHandler's last_read_file_size *only if* new data was successfully processed
                                    self.data_handler.last_read_file_size = current_file_size 
                                    self.logger.print_info("[cyan]Monitor: Processed new market data[/cyan]")
                                else:
                                    self.logger.print_warning("[yellow]Monitor: process_new_data returned False, no new data processed.[/yellow]")
                            # --- CRITICAL FIX: Removed the 'else' here ---
                            # If file size hasn't changed, we don't re-process.
                            # The previous 'else' block here was causing issues if file size didn't change
                            # but data was still being processed (e.g., due to timestamp filtering).
                        else:
                            self.logger.print_error(f"CSV file not found during monitoring: {csv_path}")

                    except Exception as e:
                        self.logger.print_error(f"Monitor: Error checking/processing new data: {str(e)}")
                        traceback.print_exc()

                    await asyncio.sleep(check_interval) # Wait for the next check

                except asyncio.CancelledError:
                    raise # Propagate cancellation
                except Exception as e:
                    self.logger.print_error(f"Unexpected error in market monitoring: {str(e)}")
                    traceback.print_exc()
                    await asyncio.sleep(check_interval) # Wait before retrying after error

        except asyncio.CancelledError:
            self.logger.print_info("[yellow]Market monitoring stopped[/yellow]")
            raise
        except Exception as e:
            self.logger.print_error(f"Fatal error in market monitoring: {str(e)}")
            traceback.print_exc()
        finally:
            self.logger.print_info("[yellow]Market monitoring finished[/yellow]")

    async def process_new_data(self, csv_path):
        """
        Reads new data from the CSV file based on the last processed timestamp,
        processes it through the DataHandler, and updates the LiveChart.
        """
        try:
            if not os.path.exists(csv_path):
                self.logger.print_error(f"CSV file not found: {csv_path}")
                return False

            # Read entire CSV and filter by timestamp. This is robust.
            full_df = pd.read_csv(
                csv_path,
                header=0,
                parse_dates=['timestamp'],
                dtype={'price': float}
            )
            self.logger.print_info(f"Process: Read {len(full_df)} total rows from CSV.")

            # Ensure 'timestamp' and 'price' columns exist
            if 'timestamp' not in full_df.columns or 'price' not in full_df.columns:
                self.logger.print_error("Process: CSV file must contain 'timestamp' and 'price' columns.")
                return False

            # Filter for new data based on the last processed timestamp from DataHandler
            new_data_df = pd.DataFrame()
            if self.data_handler.last_processed_timestamp: # Use DataHandler's timestamp to find truly new data
                # Ensure full_df timestamps are timezone-aware for correct comparison
                if full_df['timestamp'].dt.tz is None:
                    full_df['timestamp'] = full_df['timestamp'].dt.tz_localize(pytz.utc)
                else:
                    full_df['timestamp'] = full_df['timestamp'].dt.tz_convert(pytz.utc)

                if self.data_handler.last_processed_timestamp.tzinfo is None:
                     last_ts_dt = self.data_handler.last_processed_timestamp.tz_localize(pytz.utc)
                else:
                    last_ts_dt = self.data_handler.last_processed_timestamp.tz_convert(pytz.utc)

                new_data_df = full_df[full_df['timestamp'] > last_ts_dt].copy()
                self.logger.print_info(f"Process: Filtering for data > {last_ts_dt}. Found {len(new_data_df)} new rows.")
            else:
                # If no last_processed_timestamp (first run or DataHandler reset), process all data
                new_data_df = full_df.copy()
                # Ensure timestamps are localized to UTC for consistency
                if new_data_df['timestamp'].dt.tz is None:
                    new_data_df['timestamp'] = new_data_df['timestamp'].dt.tz_localize(pytz.utc)
                else:
                    new_data_df['timestamp'] = new_data_df['timestamp'].dt.tz_convert(pytz.utc)
                
                self.logger.print_info(f"Process: No last_processed_timestamp, processing all {len(new_data_df)} rows.")

            if new_data_df.empty:
                self.logger.print_info("[yellow]Process: No new data to process after timestamp filtering.[/yellow]")
                return False

            # Sort by timestamp to ensure processing order
            new_data_df = new_data_df.sort_values(by='timestamp').reset_index(drop=True)

            self.logger.print_info(f"[cyan]Processing {len(new_data_df)} new price updates...[/cyan]")
            processed_count = 0

            # Process each new 1-minute price point
            for idx, row in new_data_df.iterrows():
                try:
                    timestamp = row['timestamp']
                    price = row['price']

                    # Update LiveChart with current price for every 1-min data point
                    # This ensures the "Current Price Line" on the chart updates frequently.
                    # LiveChart.update_data now intelligently handles raw price dicts.
                    self.live_chart.update_data({'timestamp': timestamp, 'price': price}) 
                    self.logger.print_info(f"Processing: Time={timestamp}, Price={price:.8f}")

                    # Process the price update through DataHandler to aggregate into 5-min candles
                    new_candle, bucket = self.data_handler.process_new_price(timestamp, price)
                    processed_count += 1

                    if new_candle is not None:
                        self.logger.print_info(f"Process Loop: New 5-min candle formed at {new_candle.index[0]} Close: {new_candle['close'].iloc[-1]}")
                        
                        # Update five minute data in DataHandler and get the updated DataFrame.
                        # This also ensures the data_handler.five_min_data is up-to-date for fib_calculator.
                        updated_five_min_data = self.data_handler.update_data(new_candle) 
                        
                        # Calculate fib levels based on the updated five_min_data
                        fib_levels_calculated = self.fib_calculator.calculate_fib_levels(updated_five_min_data)
                        
                        # Prepare fib_data for LiveChart (assuming it expects individual values for latest levels)
                        fib_data_for_chart = {
                            'Fib 0': fib_levels_calculated['wma_fib_0'].iloc[-1] if 'wma_fib_0' in fib_levels_calculated.columns else None,
                            'Fib 50': fib_levels_calculated['wma_fib_50'].iloc[-1] if 'wma_fib_50' in fib_levels_calculated.columns else None,
                            'Entry Threshold': (fib_levels_calculated['wma_fib_0'].iloc[-1] * (1 + self.config['trading']['fib_entry_threshold'])) if 'wma_fib_0' in fib_levels_calculated.columns else None
                        }
                        
                        # Update LiveChart with the completed candle and its fib levels.
                        # LiveChart.update_data now intelligently handles completed candle Series.
                        self.live_chart.update_data(new_candle.iloc[0], fib_data_for_chart)

                        # Check for trades (this will update position_manager.trade_history and .positions)
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
                        
                        # Update status display (console logger)
                        self.logger.update_status(
                            current_price=price,
                            positions=self.position_manager.positions,
                            balance=self.position_manager.balance_sol
                        )
                        #await asyncio.sleep(0.3) #uncomment for step-through
                    else:
                        self.logger.print_info(f"Process Loop: No new 5-min candle formed yet for price at {timestamp}.")

                except ValueError as e:
                    self.logger.print_error(f"Invalid data at row {idx}: {str(e)}")
                    continue
                except Exception as e:
                    self.logger.print_error(f"Error processing row {idx} (timestamp: {row['timestamp'] if 'timestamp' in row else 'N/A'}, price: {row['price'] if 'price' in row else 'N/A'}): {str(e)}")
                    traceback.print_exc()
                    continue

            # Update the last processed line count (for console logger's internal tracking, if used)
            if processed_count > 0:
                self.data_handler.last_processed_line += processed_count 
                self.logger.print_info(f"Processed {processed_count} rows")

            return True

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

        # Cancel all active asyncio tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.print_error(f"Error cancelling task: {e}")

        # Remove uninitialized websocket_handler call
        # If you later implement a WebSocketHandler, uncomment and properly initialize it.
        # await self.websocket_handler.stop_server() 

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
        interval: int = typer.Option(
            60,
            "--interval",
            "-i",
            help="Check interval in seconds"
        ),
    ):
        """Start the trading bot with specified configuration"""
        bot = TradingBot(config_path)

        async def signal_handler(sig):
            await bot.handle_shutdown()

        try:
            # Set up signal handlers for graceful shutdown (Ctrl+C, etc.)
            for sig in (signal.SIGINT, signal.SIGTERM):
                # asyncio.create_task is used to run the async handler
                bot.loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(signal_handler(s)))

            # Run the main bot coroutine until it completes or is interrupted
            bot.loop.run_until_complete(bot.start())

        except KeyboardInterrupt:
            bot.logger.print_info("[bold red]Bot shutdown initiated by user[/bold red]")
            # Cleanup is handled by the signal_handler or by the bot.start() exception
            # but ensure final cleanup is run if KeyboardInterrupt is caught here directly.
            bot.loop.run_until_complete(bot.cleanup())

        finally:
            # Ensure the event loop is closed properly
            if not bot.loop.is_closed():
                bot.loop.close()

    app() # Run the Typer CLI application

if __name__ == "__main__":
    main()