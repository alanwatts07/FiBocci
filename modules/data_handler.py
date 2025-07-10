# modules/data_handler.py

import pandas as pd
import numpy as np
from datetime import datetime
import os
import pytz # <-- ADDED: For robust timezone handling
from modules.logger import BotLogger
import traceback

class DataHandler:
    def __init__(self):
        self.current_bucket = None # This will hold the start of the current 5-min candle being built
        self.last_processed_line = 0 # Retaining this, though last_read_file_size/timestamp are preferred
        self.last_processed_timestamp = None # Last 1-min timestamp seen by process_new_price
        
        # This will hold the aggregated 5-minute OHLCV data
        self.five_min_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self.five_min_data.index.name = 'timestamp' # Ensure index has a name for consistency

        # Prices for the current (incomplete) 5-minute bucket
        self.current_prices_in_bucket = [] 
        self.current_timestamps_in_bucket = [] 
        
        # Buffer for incoming 1-minute price points before they are aggregated
        # (This buffer is implicitly used by current_prices_in_bucket now for the current bucket,
        # but kept as a DataFrame if you plan to use it for other buffering purposes)
        self.one_min_buffer = pd.DataFrame(columns=['timestamp', 'price'])
        self.one_min_buffer.set_index('timestamp', inplace=True) 
        
        self.logger = BotLogger()
        
        # --- CRITICAL FIX 1: Initialize last_read_file_size ---
        self.last_read_file_size = 0 

    def initialize_from_csv(self, csv_path):
        """
        Initialize the data handler by loading and processing existing 1-minute
        price data from the CSV into 5-minute OHLC candles. This method also
        aligns the internal clock (current_bucket) with the real-world time
        after processing historical data.
        """
        try:
            self.logger.print_info(f"[bold blue]DataHandler Init: Starting initialization from CSV: {csv_path}[/bold blue]")

            if not os.path.exists(csv_path):
                self.logger.print_error(f"DataHandler Init: CSV file not found: {csv_path}. Starting with empty data.")
                # Ensure state is clean if file doesn't exist
                self.last_read_file_size = 0 
                self.last_processed_timestamp = None
                self.five_min_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                self.five_min_data.index.name = 'timestamp'
                self.current_bucket = None 
                self.current_prices_in_bucket = [] 
                self.current_timestamps_in_bucket = [] 
                self.one_min_buffer = pd.DataFrame(columns=['timestamp', 'price'])
                self.one_min_buffer.set_index('timestamp', inplace=True)
                return

            df = pd.read_csv(
                csv_path,
                header=0,
                parse_dates=['timestamp'],
                dtype={'price': float}
            )
            
            if df.empty:
                self.logger.print_info(f"DataHandler Init: CSV file '{csv_path}' is empty. Initializing with empty data.")
                self.last_read_file_size = os.path.getsize(csv_path) 
                self.last_processed_timestamp = None
                self.five_min_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                self.five_min_data.index.name = 'timestamp'
                self.current_bucket = None 
                self.current_prices_in_bucket = [] 
                self.current_timestamps_in_bucket = [] 
                self.one_min_buffer = pd.DataFrame(columns=['timestamp', 'price'])
                self.one_min_buffer.set_index('timestamp', inplace=True)
                return

            # --- CRITICAL FIX 2: Set last_read_file_size to current size of file ---
            self.last_read_file_size = os.path.getsize(csv_path)
            
            self.logger.print_info(f"DataHandler Init: Found {len(df)} initial 1-min price entries. Processing into 5-min candles...")

            # Reset internal state for processing existing data (important before looping through df)
            self.current_bucket = None # Temporarily reset to allow initial bucket to be set by first data point
            self.current_prices_in_bucket = []
            self.current_timestamps_in_bucket = []
            self.one_min_buffer = pd.DataFrame(columns=['timestamp', 'price'])
            self.one_min_buffer.set_index('timestamp', inplace=True)
            self.five_min_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']) # Clear existing candles
            self.five_min_data.index.name = 'timestamp'
            
            initial_five_min_candles = []
            
            # --- CRITICAL FIX 3: Re-set current_bucket for initial historical processing if needed ---
            # This ensures process_new_price starts aggregating correctly from the start of the historical data.
            if not df.empty:
                first_timestamp = df['timestamp'].iloc[0]
                # If current_bucket is None, initialize it with the first 5-min bucket from the historical data
            self.current_bucket = first_timestamp.floor('5min').tz_localize('UTC') # <--- ADD .tz_localize('UTC')                     self.logger.print_info(f"DataHandler Init: Initial current_bucket set to {self.current_bucket} based on first historical data point.")
            
            # Process all historical data point by point to build 5-min candles
            for idx, row in df.iterrows():
                timestamp = row['timestamp']
                price = row['price']
                
                # Use process_new_price to build candles from historical data
                new_candle_ohlc, _ = self.process_new_price(timestamp, price)
                if new_candle_ohlc is not None:
                    initial_five_min_candles.append(new_candle_ohlc)
                
                # Always update last_processed_timestamp with the latest 1-min price timestamp from CSV
                self.last_processed_timestamp = timestamp 

            if initial_five_min_candles:
                self.five_min_data = pd.concat(initial_five_min_candles)
                self.five_min_data.index = pd.to_datetime(self.five_min_data.index, format='mixed')
                self.five_min_data = self.five_min_data.sort_index()

                # Keep a limited number of candles for display/indicator lookback (e.g., last 1000 candles)
                max_rows_to_keep = 1000 
                if len(self.five_min_data) > max_rows_to_keep:
                    self.five_min_data = self.five_min_data.iloc[-max_rows_to_keep:]
                    
                self.logger.print_info(f"[green]DataHandler Init: Successfully populated {len(self.five_min_data)} 5-min candles from initial load.[/green]")
            else:
                self.logger.print_warning(f"[yellow]DataHandler Init: No 5-min candles were formed from initial CSV data.[/yellow]")
                self.five_min_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
                self.five_min_data.index.name = 'timestamp'
                
            # --- CRITICAL FIX 4: Align current_bucket to the actual current real-world 5-min bucket ---
            # This ensures that when live data starts coming in, the DataHandler is ready
            # to process it for the current, actual 5-minute period.
            self.current_bucket = datetime.now(tz=pytz.utc).floor('5min') 
            self.current_prices_in_bucket = [] # Clear buffer as this is for a new, live bucket
            self.current_timestamps_in_bucket = [] # Clear buffer

            self.logger.print_info(f"DataHandler Init: Final last read file size: {self.last_read_file_size} bytes.")
            self.logger.print_info(f"DataHandler Init: Final last processed 1-min timestamp: {self.last_processed_timestamp}")
            self.logger.print_info(f"DataHandler Init: Prepared for live data, current_bucket set to: {self.current_bucket}")


        except Exception as e:
            self.logger.print_error(f"DataHandler Init: Error during initialization: {str(e)}\nTraceback: {traceback.format_exc()}")
            # Ensure all state is cleanly reset on any error during initialization
            self.last_processed_line = 0
            self.last_read_file_size = 0
            self.last_processed_timestamp = None
            self.five_min_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            self.five_min_data.index.name = 'timestamp'
            self.current_bucket = None
            self.current_prices_in_bucket = []
            self.current_timestamps_in_bucket = []
            self.one_min_buffer = pd.DataFrame(columns=['timestamp', 'price'])
            self.one_min_buffer.set_index('timestamp', inplace=True)


    def process_new_price(self, timestamp, price):
        """
        Process a new 1-minute price data point.
        Aggregates 1-minute prices into 5-minute OHLCV candles.
        Returns the completed 5-minute candle and its bucket, or None if no candle is complete.
        """
        try:
            # Ensure timestamp is datetime object and timezone-aware (UTC)
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp, format='mixed')
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize('UTC') # Assume UTC if no timezone
            else:
                timestamp = timestamp.tz_convert('UTC') # Convert to UTC if it has a timezone

            price = float(price)

            # Determine the 5-minute bucket for the current incoming timestamp
            current_five_min_bucket = timestamp.floor('5min')

            new_candle = None
            
            # --- Aggregation Logic ---
            # If this is the very first price processed, or if we've moved to a new bucket
            if self.current_bucket is None:
                # Initialize current_bucket with the bucket of the first price
                self.current_bucket = current_five_min_bucket
                self.current_prices_in_bucket.append(price)
                self.current_timestamps_in_bucket.append(timestamp)
                self.logger.print_info(f"DataHandler: First price at {timestamp}. Starting new bucket {self.current_bucket}.")
            elif current_five_min_bucket > self.current_bucket:
                # A 5-minute candle has completed for the PREVIOUS self.current_bucket
                if len(self.current_prices_in_bucket) > 0:
                    open_price = self.current_prices_in_bucket[0]
                    high_price = max(self.current_prices_in_bucket)
                    low_price = min(self.current_prices_in_bucket)
                    close_price = self.current_prices_in_bucket[-1]
                    volume = len(self.current_prices_in_bucket) # Simple volume: count of 1-min prices

                    new_candle = pd.DataFrame([{
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume 
                    }], index=[self.current_bucket])
                    new_candle.index.name = 'timestamp'
                    self.logger.print_info(f"DataHandler: Formed new 5-min candle for {self.current_bucket} (Close: {close_price:.4f}).")
                else:
                    self.logger.print_warning(f"DataHandler: No prices collected for completed bucket {self.current_bucket}. Skipping candle formation.")

                # Now, start the new bucket for the current incoming timestamp
                self.current_bucket = current_five_min_bucket
                self.current_prices_in_bucket = [price] # Reset with the new price
                self.current_timestamps_in_bucket = [timestamp]
            else:
                # Still within the same 5-minute bucket, just add price to buffer
                self.current_prices_in_bucket.append(price)
                self.current_timestamps_in_bucket.append(timestamp)
                self.logger.print_info(f"DataHandler: Buffering price {price:.4f} for bucket {self.current_bucket}.")
            
            # Always update last_processed_timestamp with the latest 1-min timestamp seen
            self.last_processed_timestamp = timestamp 
            
            return new_candle, current_five_min_bucket

        except Exception as e:
            self.logger.print_error(f"Error processing new price {timestamp}, {price}: {str(e)}\nTraceback: {traceback.format_exc()}")
            return None, None

    def update_data(self, new_candle):
        """Update the main dataset with a new candle"""
        try:
            # Ensure the index of the new candle is set correctly as datetime
            if isinstance(new_candle, pd.Series):
                new_candle_df = new_candle.to_frame().T # Convert Series to DataFrame for concat
            else:
                new_candle_df = new_candle # Assume it's already a DataFrame

            if not isinstance(new_candle_df.index, pd.DatetimeIndex):
                new_candle_df.index = pd.to_datetime(new_candle_df.index, format='mixed')
            
            new_candle_df.index.name = 'timestamp' # Ensure consistent index name

            if self.five_min_data.empty:
                self.five_min_data = new_candle_df
            else:
                # Concatenate and handle potential duplicate index if a candle is re-added
                # For simplicity and robustness with live data, we concat and then sort/drop_duplicates.
                # This ensures we don't have multiple entries for the same 5-min candle.
                self.five_min_data = pd.concat([self.five_min_data, new_candle_df])
                self.five_min_data = self.five_min_data[~self.five_min_data.index.duplicated(keep='last')]


            # Ensure chronological order
            self.five_min_data = self.five_min_data.sort_index()

            # Keep only the most recent data (e.g., last 1000 candles for chart)
            max_rows_to_keep = 1000 
            if len(self.five_min_data) > max_rows_to_keep:
                self.five_min_data = self.five_min_data.iloc[-max_rows_to_keep:]
            
            self.logger.print_info(f"DataHandler: Updated 5-min data. Total candles: {len(self.five_min_data)}. Latest close: {self.five_min_data['close'].iloc[-1]:.8f}")

            return self.five_min_data

        except Exception as e:
            self.logger.print_error(f"Error updating five_min_data: {str(e)}\nTraceback: {traceback.format_exc()}")
            return self.five_min_data

    def reset(self):
        """Reset the data handler to its initial state."""
        self.current_bucket = None
        self.last_processed_line = 0
        self.last_processed_timestamp = None
        self.five_min_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self.five_min_data.index.name = 'timestamp'
        self.one_min_buffer = pd.DataFrame(columns=['timestamp', 'price'])
        self.one_min_buffer.set_index('timestamp', inplace=True)
        self.current_prices_in_bucket = []
        self.current_timestamps_in_bucket = []
        self.last_read_file_size = 0 # --- MINIMAL CHANGE 4: Reset this too ---

    def convert_buffer_to_ohlc(self):
        """This method's logic has been integrated into process_new_price for continuous aggregation.
           It is retained here for minimal changes but is likely no longer directly called in the main flow.
           If you choose to use it, ensure it's called with a full 5-min period's data.
        """
        try:
            if len(self.one_min_buffer) == 0:
                return None

            buffer = self.one_min_buffer.copy()
            buffer['timestamp'] = pd.to_datetime(buffer['timestamp'], format='mixed')
            buffer = buffer.sort_values('timestamp')

            latest_bucket = buffer['timestamp'].max().floor('5min')
            period_data = buffer[
                (buffer['timestamp'] >= latest_bucket) & 
                (buffer['timestamp'] < latest_bucket + pd.Timedelta(minutes=5))
            ]

            if len(period_data) > 0:
                ohlc = pd.DataFrame(index=[latest_bucket])
                ohlc.index = pd.to_datetime(ohlc.index, format='mixed')  # Ensure datetime index
                ohlc['open'] = period_data.iloc[0]['price']
                ohlc['high'] = period_data['price'].max()
                ohlc['low'] = period_data['price'].min()
                ohlc['close'] = period_data.iloc[-1]['price']
                ohlc['volume'] = len(period_data) # Add volume here too

                return ohlc

            return None

        except Exception as e:
            self.logger.print_error(f"Error converting buffer to OHLC: {str(e)}\nTraceback: {traceback.format_exc()}")
            return None


    def get_current_price(self):
        """Safely get the current price. Prioritize current in-bucket price if available, then last closed candle."""
        try:
            # If there are prices in the current (incomplete) bucket, return the very last one
            if self.current_prices_in_bucket:
                return self.current_prices_in_bucket[-1]
            
            # Otherwise, return the close of the last completed 5-min candle
            if not self.five_min_data.empty and 'close' in self.five_min_data.columns:
                return self.five_min_data['close'].iloc[-1]
            
            return None # No price available
        except Exception as e:
            self.logger.print_error(f"Error getting current price: {str(e)}\nTraceback: {traceback.format_exc()}")
            return None

    @staticmethod
    def calculate_initial_fib_level(historical_csv_path, lookback_periods=42):
        """Calculate initial fib level from historical data.
        Assumes the historical_csv_path contains OHLC data, or at least 'price' which can be used for high/low.
        """
        try:
            if not os.path.exists(historical_csv_path):
                print(f"Warning: Historical CSV not found for initial fib level calculation: {historical_csv_path}")
                return 0.0

            df = pd.read_csv(historical_csv_path, parse_dates=['timestamp'])
            
            if df.empty:
                print("Warning: Historical CSV is empty for fib level calculation.")
                return 0.0

            df = df.set_index('timestamp').sort_index()

            # Ensure 'high' and 'low' exist, or derive from 'price'
            if 'high' not in df.columns or 'low' not in df.columns:
                if 'price' in df.columns:
                    df['high'] = df['price']
                    df['low'] = df['price']
                else:
                    raise ValueError("Historical CSV for fib calculation must contain 'high' and 'low' columns or a 'price' column.")

            recent_data = df.tail(lookback_periods)
            if recent_data.empty:
                print(f"Warning: Not enough data ({len(df)} rows) for {lookback_periods} lookback periods for fib calculation.")
                # Return the min low from all available data if recent_data is too small
                return df['low'].min() if not df.empty else 0.0

            highest_high = recent_data['high'].max()
            lowest_low = recent_data['low'].min()
            
            # Ensure lowest_low is not None/NaN before returning
            if pd.isna(lowest_low):
                print("Warning: Lowest low is NaN, likely no valid price data in recent_data.")
                return 0.0

            fib_0 = lowest_low 

            return fib_0
        except Exception as e:
            print(f"Error calculating initial fib level: {str(e)}\nTraceback: {traceback.format_exc()}")
            return 0.0
