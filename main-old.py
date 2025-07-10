import asyncio
import time
import numpy as np
from datetime import datetime
import pandas as pd
import typer
from pathlib import Path
import traceback
import os
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
        self.live_chart = LiveChart(port=8051)
        self.live_chart.start()
        # Trade manager needs position manager and fib calculator
        self.trade_manager = TradeManager(
            self.config,
            self.position_manager,
            self.fib_calculator
        )
        # Initialize event loop for async operations
        if asyncio.get_event_loop().is_closed():
            asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = asyncio.get_event_loop()
        
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
    
            """             # Load historical data 
                        historical_data = pd.read_csv( 
                            self.config['paths']['historical_data'], 
                            parse_dates=['timestamp'], 
                            index_col='timestamp' 
                        ) 
                        
                        # Run backtest first 
                        await self.run_backtest(historical_data) 
                        """
            # Start websocket server for live chart 
            """  websocket_task = self.loop.create_task(self.websocket_handler.start_server()) 
                self.tasks.append(websocket_task) """ 
            
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
                        self.live_chart.update_data(candle, fib_data) 
                    self.logger.print_info(f"[bold green]Processed candle at {idx}[/bold green]") 
                except Exception as candle_e: 
                    self.logger.print_error(f"Error processincandleat {idx}: {str(candle_e)}") 
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
        self.logger.print_info("Detailed trade log saved tO 'backtest_trades.csv'")


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
                            file_size = os.path.getsize(csv_path)
                            if file_size > self.data_handler.last_processed_line:
                                new_data = await self.process_new_data(csv_path)
                                if new_data:
                                    self.logger.print_info("[cyan]Processed new market data[/cyan]")
                    except Exception as e:
                        self.logger.print_error(f"Error processing new data: {str(e)}")

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

            # Read CSV with explicit column names
            df = pd.read_csv(
            csv_path,
            skiprows=range(1, self.data_handler.last_processed_line + 1) if self.data_handler.last_processed_line > 0 else None,
            header=0  # Assume the first row contains headers
            )

            # Ensure 'timestamp' and 'price' columns exist
            if 'timestamp' not in df.columns or 'price' not in df.columns:
                self.logger.print_error("CSV file must contain 'timestamp' and 'price' columns.")
                return False

            # Parse the 'timestamp' column
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            # Ensure 'price' is a float
            df['price'] = df['price'].astype(float)

                
            if df.empty:
                self.logger.print_info("[yellow]No new data to process[/yellow]")
                return False

            self.logger.print_info(f"[cyan]Processing {len(df)} new price updates...[/cyan]")
            processed_count = 0
            
            # Process each new line
            for idx, row in df.iterrows():
                try:
                    # Convert timestamp with explicit format
                    timestamp = pd.to_datetime(row.iloc[0], format='mixed')
                    price = float(row.iloc[1])  # Get price from second column
                    
                    # Update live chart with current price
                    self.live_chart.update_data({
                        'timestamp': timestamp,
                        'price': price
                    })
                    
                    self.logger.print_info(f"Processing: Time={timestamp}, Price={price:.8f}")
                    
                    # Process the price update
                    new_candle, bucket = self.data_handler.process_new_price(timestamp, price)
                    processed_count += 1
                    
                    if new_candle is not None:
                        # Update five minute data with fibonacci levels
                        updated_data = self.fib_calculator.calculate_fib_levels(
                            self.data_handler.update_data(new_candle)
                        )
                        self.data_handler.five_min_data = updated_data
                        
                        # Update live chart with completed candle and fib levels
                        fib_data = {
                            'Fib 0': updated_data['wma_fib_0'] if 'wma_fib_0' in updated_data.columns else None,
                            'Fib 50': updated_data['wma_fib_50'] if 'wma_fib_50' in updated_data.columns else None,
                            'Entry Threshold': (updated_data['wma_fib_0'] * (1 + self.config['trading']['fib_entry_threshold'])) if 'wma_fib_0' in updated_data.columns else None
                        }
                        self.live_chart.update_data(new_candle.iloc[0], fib_data)
                        
                        # Check for trades
                        self.trade_manager.check_for_trades(
                            new_candle.iloc[0],
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
                    self.logger.print_error(f"Error processing row {idx}: {str(e)}")
                    traceback.print_exc()
                    continue

            # Update the last processed line count
            if processed_count > 0:
                self.data_handler.last_processed_line += processed_count
                self.logger.print_info(f"Processed {processed_count} rows")
            
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
          
          # Stop websocket server
          await self.websocket_handler.stop_server()
          
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
            bot.loop.run_until_complete(bot.cleanup()) 
     
        finally:
            bot.loop.close()

    app()

if __name__ == "__main__":
    main()

