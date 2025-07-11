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
        if len(df) < 42:
            print(f"DEBUG(FibCalc): Not enough data for initial fibs ({len(df)} < 42). Returning original DF.")
            return df
        
        df_copy = df.copy() # Renamed to df_copy to avoid confusion with the input df
        
        df_copy['highest_high'] = df_copy['high'].rolling(window=42).max()
        df_copy['lowest_low'] = df_copy['low'].rolling(window=42).min()
        
        # Debugging intermediate steps
        print(f"DEBUG(FibCalc): After highest/lowest. Latest highest_high: {df_copy['highest_high'].iloc[-1]}, lowest_low: {df_copy['lowest_low'].iloc[-1]}")
        print(f"DEBUG(FibCalc): NaN count in highest_high: {df_copy['highest_high'].isnull().sum()}")

        diff = df_copy['highest_high'] - df_copy['lowest_low']
        df_copy['fib_0'] = df_copy['lowest_low']
        df_copy['fib_50'] = df_copy['highest_high'] - (diff*0.5)

        print(f"DEBUG(FibCalc): After fib_0/fib_50. Latest fib_0: {df_copy['fib_0'].iloc[-1]}, fib_50: {df_copy['fib_50'].iloc[-1]}")
        print(f"DEBUG(FibCalc): NaN count in fib_0: {df_copy['fib_0'].isnull().sum()}")
        
        # These rolling averages need 24 non-NaN values *after* fib_0/fib_50 are calculated
        df_copy['wma_fib_0'] = df_copy['fib_0'].rolling(window=24).mean()
        df_copy['wma_fib_50'] = df_copy['fib_50'].rolling(window=24).mean()
        
        print(f"DEBUG(FibCalc): After WMA fibs. Latest wma_fib_0: {df_copy['wma_fib_0'].iloc[-1]}, wma_fib_50: {df_copy['wma_fib_50'].iloc[-1]}")
        print(f"DEBUG(FibCalc): NaN count in wma_fib_0: {df_copy['wma_fib_0'].isnull().sum()}")
        print(f"DEBUG(FibCalc): Length of DF passed to fib_calculator: {len(df)}")
        
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