import asyncio
import time
import numpy as np
from datetime import datetime
import pandas as pd
import typer
from pathlib import Path
import traceback
import os
import threading # <--- ADD THIS IMPORT
import signal

# Assuming these modules exist and are correctly implemented
from modules.backtest_manager import BacktestManager
from modules.data_handler import DataHandler
from modules.trade_manager import TradeManager
from modules.position_manager import PositionManager
from modules.indicators import FibonacciCalculator
from utils.helpers import load_config, format_timestamp
from modules.logger import BotLogger
from modules.live_chart import LiveChart
# from modules.websocket_handler import WebSocketHandler # Uncomment if you have this module

class TradingBot:
    def __init__(self, config_path='config/config.yaml'):
        """Initialize the trading bot with all its components"""
        # Load configuration
        self.config = load_config(config_path)
        self.logger = BotLogger()
        self.shutdown_event = asyncio.Event()
        self.last_processed_timestamp = None
        # Initialize components
        self.data_handler = DataHandler()
        self.data_handler.initialize_from_csv(self.config['paths']['live_data'])
        self.fib_calculator = FibonacciCalculator(self.config)
        self.position_manager = PositionManager(self.config)
        self.running = False

        self.tasks = []
        
        # Initialize LiveChart but DON'T start it here
        self.live_chart = LiveChart(port=8051)
        
        # Initialize WebSocketHandler if you intend to use it
        # self.websocket_handler = WebSocketHandler() # Uncomment if you have this module

        # Trade manager needs position manager and fib calculator
        self.trade_manager = TradeManager(
            self.config,
            self.position_manager,
            self.fib_calculator
        )
        # Initialize event loop for async operations
        # It's generally better to get the loop once and reuse it, or let run_until_complete create it.
        # This part might be better handled in main() or by letting asyncio.run() handle it.
        # For now, keeping your existing loop setup.
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
        self.chart_update_counter = 0

    async def start(self):
        """Start the trading bot"""
        try:
            self.running = True
            self.logger.start_ui()
            self.logger.print_info("[bold green]Initializing Trading Bot...[/bold green]")

            # Start LiveChart in a separate thread
            # The Dash app's run() method is blocking, so it needs its own thread.
            self.chart_thread = threading.Thread(
                target=self.live_chart.app.run, # Directly call the Dash app's run method
                kwargs={'debug': False, 'port': self.live_chart.port, 'host': '0.0.0.0'}
            )
            self.chart_thread.daemon = True # Allow main program to exit even if thread is running
            self.chart_thread.start()
            self.logger.print_info(f"[bold green]Live Chart started on http://0.0.0.0:{self.live_chart.port}[/bold green]")


            """ # Load historical data
                historical_data = pd.read_csv(
                    self.config['paths']['historical_data'],
                    parse_dates=['timestamp'],
                    index_col='timestamp'
                )

                # Run backtest first
                await self.run_backtest(historical_data)
            """
            # Start websocket server for live chart (if you uncommented WebSocketHandler)
            # if hasattr(self, 'websocket_handler'):
            #     websocket_task = self.loop.create_task(self.websocket_handler.start_server())
            #     self.tasks.append(websocket_task)

            # Start live market monitoring
            market_monitor_task = self.loop.create_task(self.monitor_market(self.config['paths']['live_data']))
            self.tasks.append(market_monitor_task)
            self.logger.print_status()
            self.position_manager.print_portfolio_status(self.data_handler.get_current_price())
            # Wait until shutdown_event is set
            await self.shutdown_event.wait()

        except KeyboardInterrupt:
            self.logger.print_info("[bold red]Shutting down gracefully...[/bold red]")
            await self.cleanup()
        except Exception as e:
            self.logger.print_error(f"Error in main loop: {str(e)}")
            await self.cleanup()

    async def handle_shutdown(self):
        """Handle shutdown signal"""
        self.logger.print_info("[bold red]Shutdown signal received. Cleaning up...[/bold red]")
        self.running = False
        self.shutdown_event.set()
        await self.cleanup()

    async def run_backtest(self, historical_data):
        """Run backtest on historical data before live trading"""
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

                    # Update five minute data
                    updated_data = self.fib_calculator.calculate_fib_levels(
                        self.data_handler.update_data(pd.DataFrame([candle]))
                    )
                    self.data_handler.five_min_data = updated_data

                    # Check for trades
                    self.trade_manager.check_for_trades(candle, self.data_handler.five_min_data)
                    # Update live chart
                    if hasattr(self, 'live_chart'):
                        fib_data = {
                            'Fib 0': updated_data['wma_fib_0'].values
                               if 'wma_fib_0' in updated_data.columns else None,
                            'Fib 50': updated_data['wma_fib_50'].values
                            if 'wma_fib_50' in updated_data.columns else None,
                            'Entry Threshold': (updated_data['wma_fib_0'] * (1 + self.config['trading']['fib_entry_threshold'])).values
                            if 'wma_fib_0' in updated_data.columns else None
                        }
                        # Pass a Series or dict for single candle update, not a DataFrame
                        self.live_chart.update_data(candle, fib_data) 
                    self.logger.print_info(f"[bold green]Processed candle at {idx}[/bold green]")
                except Exception as candle_e:
                    self.logger.print_error(f"Error processing candle at {idx}: {str(candle_e)}")
                    traceback.print_exc()
                    continue
            # Print backtest results
            self.print_backtest_results()

            self.logger.print_info("[bold green]Backtest Complete - Transitioning to Live Trading[/bold green]")

        except Exception as e:
            self.logger.print_error(f"Error in backtest: {str(e)}")
            traceback.print_exc()

    def print_backtest_results(self):
        """Print detailed backtest results"""
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

        # Save detailed trade log
        df = pd.DataFrame(trades)
        df.to_csv('backtest_trades.csv', index=False)
        self.logger.print_info("Detailed trade log saved to 'backtest_trades.csv'")


    async def monitor_market(self, csv_path, check_interval=60):
        """Monitor market data from CSV file"""
        try:
            self.logger.print_info(f"[bold blue]Starting market monitoring... ({csv_path})[/bold blue]")

            # Verify file exists
            if not os.path.exists(csv_path):
                self.logger.print_error(f"CSV file not found: {csv_path}")
                return

            # Initial data load
            try:
                await self.process_new_data(csv_path)
                self.logger.print_info("[green]Initial data load complete[/green]")
            except Exception as e:
                self.logger.print_error(f"Error during initial data load: {str(e)}")
                traceback.print_exc()
                return

            # Continue monitoring for new data
            while self.running:
                try:
                    current_time = time.time()

                    # Update status display with safe data access
                    if current_time - self.last_status_time >= 5:
                        try:
                            current_price = None
                            if (hasattr(self.data_handler, 'five_min_data') and
                                not self.data_handler.five_min_data.empty and
                                'close' in self.data_handler.five_min_data.columns):
                                current_price = self.data_handler.five_min_data['close'].iloc[-1]

                            positions = getattr(self.position_manager, 'positions', [])
                            balance = getattr(self.position_manager, 'balance_sol', 0.0)

                            self.logger.update_status(
                                current_price=current_price,
                                positions=positions,
                                balance=balance
                            )
                            self.last_status_time = current_time
                        except Exception as e:
                            self.logger.print_error(f"Status update error: {str(e)}\nTraceback: {traceback.format_exc()}")

                    # Check for new data
                    try:
                        if os.path.exists(csv_path):
                            # Get the current size of the file
                            current_file_size = os.path.getsize(csv_path)
                            
                            # Only process if file size has increased (new data added)
                            if current_file_size > self.data_handler.last_read_file_size:
                                new_data_processed = await self.process_new_data(csv_path)
                                if new_data_processed:
                                    self.logger.print_info("[cyan]Processed new market data[/cyan]")
                                    self.data_handler.last_read_file_size = current_file_size # Update last read size
                            else:
                                self.logger.print_info("[yellow]No new data in CSV file (size unchanged)[/yellow]")
                        else:
                            self.logger.print_error(f"CSV file not found during monitoring: {csv_path}")

                    except Exception as e:
                        self.logger.print_error(f"Error checking/processing new data: {str(e)}")

                    await asyncio.sleep(check_interval)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.print_error(f"Unexpected error in market monitoring: {str(e)}")
                    traceback.print_exc()
                    await asyncio.sleep(check_interval)

        except asyncio.CancelledError:
            self.logger.print_info("[yellow]Market monitoring stopped[/yellow]")
            raise
        except Exception as e:
            self.logger.print_error(f"Fatal error in market monitoring: {str(e)}")
            traceback.print_exc()
        finally:
            self.logger.print_info("[yellow]Market monitoring finished[/yellow]")

    async def process_new_data(self, csv_path):
        """Process new data from CSV file"""
        try:
            # Verify file exists
            if not os.path.exists(csv_path):
                self.logger.print_error(f"CSV file not found: {csv_path}")
                return False

            # Read CSV. Read the entire file and then filter new data.
            # This is more robust than skiprows if the file is being actively written to.
            full_df = pd.read_csv(
                csv_path,
                header=0,  # Assume the first row contains headers
                parse_dates=['timestamp'], # Parse timestamp column directly
                dtype={'price': float} # Ensure price is float
            )

            # Ensure 'timestamp' and 'price' columns exist
            if 'timestamp' not in full_df.columns or 'price' not in full_df.columns:
                self.logger.print_error("CSV file must contain 'timestamp' and 'price' columns.")
                return False

            # Filter for new data based on the last processed timestamp
            new_data_df = pd.DataFrame()
            if self.last_processed_timestamp:
                new_data_df = full_df[full_df['timestamp'] > self.last_processed_timestamp].copy()
            else:
                # If no last_processed_timestamp, process all data (initial load)
                new_data_df = full_df.copy()

            if new_data_df.empty:
                self.logger.print_info("[yellow]No new data to process[/yellow]")
                return False

            # Sort by timestamp to ensure processing order
            new_data_df = new_data_df.sort_values(by='timestamp').reset_index(drop=True)

            self.logger.print_info(f"[cyan]Processing {len(new_data_df)} new price updates...[/cyan]")
            processed_count = 0

            for idx, row in new_data_df.iterrows():
                try:
                    timestamp = row['timestamp']
                    price = row['price']

                    self.logger.print_info(f"Processing: Time={timestamp}, Price={price:.8f}")

                    # Process the price update
                    new_candle, bucket = self.data_handler.process_new_price(timestamp, price)
                    processed_count += 1

                    if new_candle is not None:
                        # Update five minute data with fibonacci levels
                        # Ensure update_data returns a DataFrame for calculate_fib_levels
                        self.data_handler.update_data(new_candle) # This updates internal data_handler state
                        updated_data = self.fib_calculator.calculate_fib_levels(self.data_handler.five_min_data)
                        self.data_handler.five_min_data = updated_data # Update the data_handler's five_min_data

                        # Update live chart with completed candle and fib levels
                        fib_data = {
                            'Fib 0': updated_data['wma_fib_0'].iloc[-1] if 'wma_fib_0' in updated_data.columns else None,
                            'Fib 50': updated_data['wma_fib_50'].iloc[-1] if 'wma_fib_50' in updated_data.columns else None,
                            'Entry Threshold': (updated_data['wma_fib_0'].iloc[-1] * (1 + self.config['trading']['fib_entry_threshold'])) if 'wma_fib_0' in updated_data.columns else None
                        }
                        # Pass the last candle data as a Series or dict
                        self.live_chart.update_data(new_candle.iloc[0], fib_data) 
                        
                        # Check for trades
                        self.trade_manager.check_for_trades(
                            new_candle.iloc[0], # Pass the specific candle row
                            self.data_handler.five_min_data
                        )
                        # Update live chart with latest trades and positions
                        self.live_chart.update_trades(
                            trades=self.position_manager.trade_history,
                            positions=self.position_manager.positions
                        )
                        # Update trade statistics
                        stats = self.position_manager.get_trade_statistics()
                        self.live_chart.update_trade_statistics(stats)
                        is_armed = self.trade_manager.entry_switch_armed
                        self.live_chart.update_is_armed_status(is_armed)
                        # Update status display
                        self.logger.update_status(
                            current_price=price,
                            positions=self.position_manager.positions,
                            balance=self.position_manager.balance_sol
                        )
                        #await asyncio.sleep(0.3) #uncomment for step-through
                except ValueError as e:
                    self.logger.print_error(f"Invalid data at row {idx}: {str(e)}")
                    continue
                except Exception as e:
                    self.logger.print_error(f"Error processing row {idx} (timestamp: {row['timestamp'] if 'timestamp' in row else 'N/A'}, price: {row['price'] if 'price' in row else 'N/A'}): {str(e)}")
                    traceback.print_exc()
                    continue

            # Update the last processed timestamp
            if not new_data_df.empty:
                self.last_processed_timestamp = new_data_df['timestamp'].iloc[-1]
                self.logger.print_info(f"Successfully processed {processed_count} rows up to {self.last_processed_timestamp}")

            return True

        except Exception as e:
            self.logger.print_error(f"Error processing new data: {str(e)}")
            traceback.print_exc()
            return False

    async def update_status(self):
        """Update bot status display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_price = (
            self.data_handler.five_min_data['close'].iloc[-1]
            if not self.data_handler.five_min_data.empty
            else 0
        )

        status_info = f"""
        [bold white]=== Bot Status Update ===[/bold white]
        Time: {current_time}
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

        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop LiveChart thread (if it was started)
        if hasattr(self, 'chart_thread') and self.chart_thread.is_alive():
            self.logger.print_info("[bold yellow]Stopping LiveChart thread...[/bold yellow]")
            # Dash's development server doesn't have a direct stop() method.
            # The daemon=True flag allows the main program to exit, killing the thread.
            # For a cleaner shutdown, you might need to send a signal to the Dash app's process
            # or implement a shutdown endpoint in LiveChart if it's a production server.
            # For development, setting daemon=True is often sufficient.
            # If you need a more explicit shutdown, LiveChart would need a method to stop its server.
            # For now, we'll rely on daemon=True.

        # Stop websocket server (if you uncommented WebSocketHandler and it has a stop_server method)
        # if hasattr(self, 'websocket_handler'):
        #     await self.websocket_handler.stop_server()

        # Save trade history
        self.position_manager.save_trade_history()

        # Final status update
        await self.update_status()

        self.logger.print_info("[bold green]Cleanup completed[/bold green]")
        self.logger.stop()

def main():
    """Main entry point with Typer CLI"""
    app = typer.Typer(help="Trading Bot CLI")

    @app.callback()
    def callback():
        """
        Trading Bot - A cryptocurrency trading bot
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
            # Set up signal handlers
            for sig in (signal.SIGINT, signal.SIGTERM):
                bot.loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(signal_handler(s)))

            # Run the bot
            bot.loop.run_until_complete(bot.start())

        except KeyboardInterrupt:
            bot.logger.print_info("[bold red]Bot shutdown initiated by user[/bold red]")
            # Cleanup is handled by signal_handler or the main loop's exception handler
            # if the KeyboardInterrupt is caught directly by run_until_complete.
            # It's good to ensure cleanup is called.
            bot.loop.run_until_complete(bot.cleanup())

        finally:
            # Ensure the loop is closed only after all tasks and cleanup are done
            if not bot.loop.is_closed():
                bot.loop.close()

    app()

if __name__ == "__main__":
    main()
