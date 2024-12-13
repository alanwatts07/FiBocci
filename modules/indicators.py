import pandas as pd
import numpy as np

class FibonacciCalculator:
    def __init__(self, config):
        self.config = config

    def calculate_fib_levels(self, df):
        """Calculate Fibonacci levels for the latest data"""
        if len(df) < 42:
            return df
        
        df = df.copy()
        df['highest_high'] = df['high'].rolling(window=42).max()
        df['lowest_low'] = df['low'].rolling(window=42).min()
        
        diff = df['highest_high'] - df['lowest_low']
        df['fib_0'] = df['lowest_low']
        df['fib_50'] = df['highest_high'] - (diff*0.5)
        df['wma_fib_0'] = df['fib_0'].rolling(window=24).mean()
        df['wma_fib_50'] = df['fib_50'].rolling(window=24).mean()
        
        return df

    def check_fib_conditions(self, price, fib_level, threshold_level):
        """Check if price meets Fibonacci entry conditions"""
        price_to_fib_percent = (price - fib_level) / fib_level
        
        return {
            'price_to_fib_percent': price_to_fib_percent,
            'below_threshold': price <= threshold_level,
            'above_fib': price > fib_level
        }