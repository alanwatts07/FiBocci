import pandas as pd
from datetime import datetime
import numpy as np 

class PositionManager:
    def __init__(self, config):
        self.positions = []
        self.trade_history = []
        self.config = config
        self.balance_sol = config['trading']['initial_balance']
        self.holdings_wif = 0
        self.trade_size_sol = config['trading']['trade_size']

    def reset(self):
        """Reset position manager state"""
        self.positions = []
        self.trade_history = []
        self.balance_sol = self.config['trading']['initial_balance']
        self.holdings_wif = 0

    def add_position(self, price, timestamp):
        """Add a new position"""
        if len(self.positions) >= self.config['trading']['max_positions']:
            return False, "Maximum positions reached"
            
        if self.balance_sol < self.trade_size_sol:
            return False, "Insufficient balance"
            
        wif_amount = self.trade_size_sol / price
        
        position = {
            'id': len(self.trade_history) + len(self.positions) + 1, # Simple unique ID
            'entry_time': timestamp,
            'entry_price': price,
            'wif_amount': wif_amount,
            'sol_spent': self.trade_size_sol,
            'trigger_low': price # Keep this if used for stop-loss or trailing logic
        }
        
        self.positions.append(position)
        self.balance_sol -= self.trade_size_sol
        self.holdings_wif += wif_amount
        
        return True, position

    def close_position(self, position, current_price, current_time, exit_type):
        """Close a specific position"""
        if position not in self.positions:
            return False, "Position not found"
            
        sol_received = position['wif_amount'] * current_price
        # Ensure 'profit' is calculated as a float, not rounded prematurely
        profit_percentage = (current_price - position['entry_price']) / position['entry_price']
        
        self.balance_sol += sol_received
        self.holdings_wif -= position['wif_amount']
        
        trade_record = {
            'entry_time': position['entry_time'],
            'exit_time': current_time,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'profit': float(profit_percentage), # Ensure it's explicitly a float
            'sol_profit': sol_received - position['sol_spent'],
            'exit_type': exit_type,
            'candles_held': None # This will be calculated by trade manager
        }
        
        self.trade_history.append(trade_record)
        self.save_trade_history() # Save trade log after closing a position
        self.positions.remove(position)
        return True, trade_record

    def close_all_positions(self, current_price, current_time, exit_type):
        """Close all open positions"""
        results = []
        for position in self.positions[:]: 
            success, result = self.close_position(position, current_price, current_time, exit_type)
            if success:
                results.append(result)
        return results

    def get_position_metrics(self, current_price):
        """
        Calculates and returns various metrics for open positions and overall trading.
        Includes cumulative profit percentage (total capital change as percentage of initial capital).
        """
        total_unrealized_profit_sol = 0
        total_unrealized_profit_pct_open_positions = 0 

        if self.positions:
            total_open_positions_initial_sol_value = 0 
            for pos in self.positions:
                unrealized_profit_sol = (current_price - pos['entry_price']) * pos['wif_amount']
                pos['unrealized_profit_sol'] = unrealized_profit_sol
                pos['unrealized_profit_pct'] = (unrealized_profit_sol / pos['sol_spent']) * 100 if pos['sol_spent'] != 0 else 0
                
                total_unrealized_profit_sol += unrealized_profit_sol
                total_open_positions_initial_sol_value += pos['sol_spent']

            if total_open_positions_initial_sol_value != 0:
                total_unrealized_profit_pct_open_positions = (total_unrealized_profit_sol / total_open_positions_initial_sol_value) * 100
        
        cumulative_realized_sol_profit = sum(trade['sol_profit'] for trade in self.trade_history)
        
        total_capital_change_sol = cumulative_realized_sol_profit + total_unrealized_profit_sol

        initial_trading_capital = self.config['trading']['initial_balance']
        
        cumulative_profit = (total_capital_change_sol / initial_trading_capital) if initial_trading_capital != 0 else 0

        return {
            'total_unrealized_profit_sol': total_unrealized_profit_sol,
            'total_unrealized_profit_pct_open_positions': total_unrealized_profit_pct_open_positions,
            'cumulative_profit': cumulative_profit, 
            'num_open_positions': len(self.positions),
            'current_balance_sol': self.balance_sol 
        }

    def get_trade_statistics(self):
        """Calculate and return general trade statistics from trade history."""
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit_pct_sum_individual': 0.0, 
                'average_profit_pct': 0.0,
                'best_trade_pct': 0.0,
                'worst_trade_pct': 0.0,
                'total_sol_profit': 0.0,
                'overall_capital_gain_pct': 0.0 
            }
            
        # Ensure these are properly cast to float if there's any chance they became something else
        profits_pct = [float(trade['profit']) for trade in self.trade_history] 
        sol_profits = [float(trade['sol_profit']) for trade in self.trade_history]
        
        win_rate = (np.array(profits_pct) > 0).mean() * 100 # This correctly uses percentages
        
        # total_profit_pct_sum_individual = sum(profits_pct) # Can be confusing, consider if truly needed.
                                                         # The `overall_capital_gain_pct` is usually what's desired.
        
        average_profit_pct = np.mean(profits_pct) * 100 # Multiply by 100 for percentage display
        best_trade_pct = max(profits_pct) * 100 # Multiply by 100 for percentage display
        worst_trade_pct = min(profits_pct) * 100 # Multiply by 100 for percentage display
        
        total_sol_profit = sum(sol_profits)

        initial_balance = self.config['trading']['initial_balance']
        overall_capital_gain_pct = (total_sol_profit / initial_balance) * 100 if initial_balance != 0 else 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': overall_capital_gain_pct, # Now total_profit reflects overall gain
            'average_profit': average_profit_pct, 
            'best_trade': best_trade_pct, 
            'worst_trade': worst_trade_pct, 
            'total_sol_profit': total_sol_profit,
            'overall_capital_gain_pct': overall_capital_gain_pct 
        }

    def save_trade_history(self):
        """Save trade history to CSV"""
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            # Ensure 'profit' column is formatted as percentage if not already
            if 'profit' in df.columns:
                # Ensure the 'profit' column itself is numeric before formatting
                df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0)
                df['profit_pct_display'] = df['profit'].apply(lambda x: f"{x:.2%}")
            df.to_csv(self.config['paths']['trade_history'], index=False)
            print("\nTrade history saved to trade_history.csv")
        else:
            print("\nNo trade history to save.")

    def print_portfolio_status(self, current_price):
        """Print current portfolio status to console (for initial log)"""
        print("\n=== Portfolio Status ===")
        print(f"SOL Balance: {self.balance_sol:.3f}")
        print(f"WIF Holdings: {self.holdings_wif:.3f}")
        print(f"Open Positions: {len(self.positions)}")
        print(f"Total Trades: {len(self.trade_history)}")
        
        if self.positions:
            metrics = self.get_position_metrics(current_price)
            if metrics:
                print(f"\nTotal Unrealized Profit (SOL): {metrics['total_unrealized_profit_sol']:.3f}")
                print(f"Total Unrealized Profit (%): {metrics['total_unrealized_profit_pct_open_positions']:.2f}%")
            else:
                print("\nNo detailed metrics for open positions (no open positions).")
        
        metrics_overall = self.get_position_metrics(current_price) 
        if metrics_overall and 'cumulative_profit' in metrics_overall:
             print(f"Overall Cumulative Profit: {metrics_overall['cumulative_profit']:.2f}%")
        else:
            print("Overall Cumulative Profit: 0.00%")