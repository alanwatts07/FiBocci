# modules/data_handler.py

import pandas as pd
import numpy as np
from datetime import datetime
import os
from modules.logger import BotLogger
import traceback

class DataHandler:
    def __init__(self):
        self.current_bucket = None
        self.last_processed_line = 0
        self.last_processed_timestamp = None
        self.five_min_data = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        self.current_prices = []
        self.one_min_buffer = pd.DataFrame(columns=['timestamp', 'price'])
        self.current_timestamps = []
        self.logger = BotLogger()

   
    def process_new_price(self, timestamp, price):
        """Process new 1-minute price data"""
        try:
            # Ensure timestamp is timezone-aware
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp, format='mixed')
            elif timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize('UTC')
            
            price = float(price)
            
            # Add new price to buffer
            new_data = pd.DataFrame({
                'timestamp': [timestamp],
                'price': [price]
            })
            
            if self.one_min_buffer.empty:
                self.one_min_buffer = new_data
            else:
                self.one_min_buffer = pd.concat([self.one_min_buffer, new_data])
            
            # Convert timestamp to 5-minute bucket
            five_min_bucket = timestamp.floor('5min')
            
            # Check if we have moved to a new 5-minute period
            if self.last_processed_timestamp is None or five_min_bucket > self.last_processed_timestamp:
                # Convert buffered 1-minute data to 5-minute OHLC
                if len(self.one_min_buffer) >= 1:
                    new_candle = self.convert_buffer_to_ohlc()
                    if new_candle is not None:
                        self.last_processed_timestamp = five_min_bucket
                        
                        # Clean up buffer
                        self.one_min_buffer = self.one_min_buffer[
                            self.one_min_buffer['timestamp'] > five_min_bucket
                        ].copy()
                        
                        return new_candle, five_min_bucket
            
            return None, None
            
        except Exception as e:
            print(f"Error processing price: {str(e)}")
            traceback.print_exc()
            return None, None

    def update_data(self, new_candle):
        """Update the main dataset with a new candle"""
        try:
            # Ensure the index is datetime
            if self.five_min_data.empty:
                self.five_min_data = new_candle
            else:
                self.five_min_data = pd.concat([self.five_min_data, new_candle])
                
            # Convert index to datetime if it's not already
            if not isinstance(self.five_min_data.index, pd.DatetimeIndex):
                self.five_min_data.index = pd.to_datetime(format='mixed', arg=self.five_min_data.index)
           
            
            return self.five_min_data
            
        except Exception as e:
            print(f"Error updating data: {str(e)}")
            traceback.print_exc()
            return self.five_min_data

    def reset(self):
        """Reset the data handler"""
        self.current_bucket = None
        self.last_processed_line = 0
        self.five_min_data = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        self.temp_df = pd.DataFrame(columns=['timestamp', 'price'])

    def convert_buffer_to_ohlc(self):
        """Convert buffered 1-minute data to 5-minute OHLC"""
        try:
            if len(self.one_min_buffer) == 0:
                return None
                
            buffer = self.one_min_buffer.copy()
            buffer['timestamp'] = pd.to_datetime(buffer['timestamp'], format='mixed')
            buffer = buffer.sort_values('timestamp')
            
            latest_bucket = buffer['timestamp'].max().floor('5min')
            period_data = buffer[buffer['timestamp'] <= latest_bucket + pd.Timedelta(minutes=5)]
            
            if len(period_data) > 0:
                ohlc = pd.DataFrame(index=[latest_bucket])
                ohlc.index = pd.to_datetime(arg=ohlc.index, format='mixed')  # Ensure datetime index
                ohlc['open'] = period_data.iloc[0]['price']
                ohlc['high'] = period_data['price'].max()
                ohlc['low'] = period_data['price'].min()
                ohlc['close'] = period_data.iloc[-1]['price']
                
                return ohlc
                
            return None
            
        except Exception as e:
            print(f"Error converting buffer")

    def initialize_from_csv(self, csv_path):
        """Initialize the last processed line count"""
        try:
            # Check if file exists
            if not os.path.exists(csv_path):
                self.logger.print_error(f"CSV file not found: {csv_path}")
                return
            
            # Start from beginning
            self.last_processed_line = 0
            
            self.logger.print_info(f"Initialized data handler with CSV: {csv_path}")
            self.logger.print_info(f"Starting from beginning of file")
            
        except Exception as e:
            self.logger.print_error(f"Error initializing from CSV: {str(e)}")
            self.last_processed_line = 0

    def get_current_price(self):
        """Safely get the current price"""
        try:
            if not self.five_min_data.empty and 'close' in self.five_min_data.columns:
                return self.five_min_data['close'].iloc[-1]
            return None
        except Exception as e:
            self.logger.print_error(f"Error getting current price: {str(e)}")
            return None
        

   
    @staticmethod
    def calculate_initial_fib_level(historical_csv_path, lookback_periods=42):
        """Calculate initial fib level from historical data"""
        df = pd.read_csv(historical_csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df = df.set_index('timestamp').sort_index()
        
        recent_data = df.tail(lookback_periods)
        highest_high = recent_data['high'].max()
        lowest_low = recent_data['low'].min()
        fib_0 = lowest_low
        
        return fib_0