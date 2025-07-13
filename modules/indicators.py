# modules/indicators.py
import pandas as pd
import numpy as np # Ensure numpy is imported for isnull check

class FibonacciCalculator:
    def __init__(self, config):
        self.config = config
        # You might want a logger here too, but for quick debug, print is fine.
        # from modules.logger import BotLogger
        # self.logger = BotLogger()

    def calculate_fib_levels(self, df):
        if len(df) < 66: # Increased to ensure enough data for rolling WMA
            print(f"DEBUG(FibCalc): Not enough data for WMA fibs ({len(df)} < 66).")
            return df
        
        df_copy = df.copy()
        
        df_copy['highest_high'] = df_copy['high'].rolling(window=42).max()
        df_copy['lowest_low'] = df_copy['low'].rolling(window=42).min()
        
        diff = df_copy['highest_high'] - df_copy['lowest_low']
        df_copy['fib_0'] = df_copy['lowest_low']
        df_copy['fib_50'] = df_copy['highest_high'] - (diff * 0.5)
        
        # Calculate the original WMA levels
        df_copy['wma_fib_0'] = df_copy['fib_0'].rolling(window=24).mean()
        df_copy['wma_fib_50'] = df_copy['fib_50'].rolling(window=24).mean()

        # --- NEW: APPLY THE CONFIGURABLE OFFSET ---
        # Get offset from config, default to 0.0 if not present
        offset_pct = self.config['trading'].get('wma_fib_0_offset_pct', 0.0)

        # If an offset is defined in the config, apply it
        if offset_pct > 0.0:
            print(f"DEBUG(FibCalc): Applying {offset_pct:.2%} downward offset to WMA Fib 0.")
            df_copy['wma_fib_0'] = df_copy['wma_fib_0'] * (1 - offset_pct)
        # --- END OF NEW CODE ---

        print(f"DEBUG(FibCalc): After WMA fibs. Latest wma_fib_0: {df_copy['wma_fib_0'].iloc[-1]}, wma_fib_50: {df_copy['wma_fib_50'].iloc[-1]}")
        
        return df_copy

    # ... (check_fib_conditions methods) ...
    def check_fib_conditions(self, price, fib_level, threshold_level):
        """Check if price meets Fibonacci entry conditions"""
        price_to_fib_percent = (price - fib_level) / fib_level
        
        return {
            'price_to_fib_percent': price_to_fib_percent,
            'below_threshold': price <= threshold_level,
            'above_fib': price > fib_level
        }