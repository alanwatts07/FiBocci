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
            'entry_time': timestamp,
            'entry_price': price,
            'wif_amount': wif_amount,
            'sol_spent': self.trade_size_sol,
            'trigger_low': price
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
        profit = (current_price - position['entry_price']) / position['entry_price']
        
        self.balance_sol += sol_received
        self.holdings_wif -= position['wif_amount']
        
        trade_record = {
            'entry_time': position['entry_time'],
            'exit_time': current_time,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'profit': profit,
            'sol_profit': sol_received - position['sol_spent'],
            'exit_type': exit_type,
            'candles_held': None  # This will be calculated by trade manager
        }
        
        self.trade_history.append(trade_record)
        self.save_trade_history()  # Save trade log after closing a position
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
        """Calculate metrics for all open positions"""
        if not self.positions:
            return None
            
        total_sol_invested = sum(pos['sol_spent'] for pos in self.positions)
        total_wif_held = sum(pos['wif_amount'] for pos in self.positions)
        avg_entry_price = total_sol_invested / total_wif_held if total_wif_held > 0 else 0
        cumulative_profit = (current_price - avg_entry_price) / avg_entry_price if avg_entry_price > 0 else 0
        
        return {
            'total_sol_invested': total_sol_invested,
            'total_wif_held': total_wif_held,
            'avg_entry_price': avg_entry_price,
            'cumulative_profit': cumulative_profit,
            'position_count': len(self.positions),
        }
    def get_trade_statistics(self):
            """Calculate and return trade statistics"""
            total_trades = len(self.trade_history)
            if total_trades == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0,
                    'average_profit': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0,
                    'total_sol_profit': 0.0,
                }
            
            profits = [trade['profit'] for trade in self.trade_history]
            sol_profits = [trade['sol_profit'] for trade in self.trade_history]
            
            win_rate = (np.array(profits) > 0).mean() * 100
            total_profit = sum(profits)
            average_profit = np.mean(profits)
            best_trade = max(profits)
            worst_trade = min(profits)
            total_sol_profit = sum(sol_profits)
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'average_profit': average_profit,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'total_sol_profit': total_sol_profit,
            }
    def save_trade_history(self):
        """Save trade history to CSV"""
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            df.to_csv(self.config['paths']['trade_history'], index=False)
            print("\nTrade history saved to trade_history.csv")

    def print_portfolio_status(self, current_price):
        """Print current portfolio status"""
        print("\n=== Portfolio Status ===")
        print(f"SOL Balance: {self.balance_sol:.3f}")
        print(f"WIF Holdings: {self.holdings_wif:.3f}")
        print(f"Open Positions: {len(self.positions)}")
        print(f"Total Trades: {len(self.trade_history)}")
        
        if self.positions:
            metrics = self.get_position_metrics(current_price)
            print(f"\nAverage Entry: {metrics['avg_entry_price']:.8f}")
            print(f"Cumulative Profit: {metrics['cumulative_profit']:.2%}")