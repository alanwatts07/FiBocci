import pandas as pd
from datetime import datetime
import numpy as np 

class PositionManager:
    def __init__(self, config):
        self.positions = []
        self.trade_history = []
        self.config = config
        
        # We'll assume 'initial_balance' and 'trade_size' in config.yaml
        # are now implicitly in USDC, even if named 'sol'.
        self.balance_sol = config['trading']['initial_balance'] # This will now represent USDC balance
        self.holdings_wif = 0 # This represents actual SOL tokens held
        self.trade_size_sol = config['trading']['trade_size'] # This represents USDC amount per trade

        # Store initial balance for overall profit calculation
        self.initial_capital_usdc = config['trading']['initial_balance']

    def reset(self):
        """Reset position manager state"""
        self.positions = []
        self.trade_history = []
        self.balance_sol = self.config['trading']['initial_balance'] # Reset to initial USDC balance
        self.holdings_wif = 0

    def add_position(self, price, timestamp):
        """Add a new position (spending USDC, acquiring SOL)"""
        if len(self.positions) >= self.config['trading']['max_positions']:
            return False, "Maximum positions reached"
            
        # Check against current USDC balance (named 'balance_sol')
        if self.balance_sol < self.trade_size_sol:
            return False, "Insufficient USDC balance"
            
        # Calculate SOL amount to acquire
        sol_amount_acquired = self.trade_size_sol / price 
        
        position = {
            'id': len(self.trade_history) + len(self.positions) + 1,
            'entry_time': timestamp,
            'entry_price': price,           # Price in USDC/SOL
            'sol_amount': sol_amount_acquired, # Amount of SOL acquired
            'usdc_spent_for_trade': self.trade_size_sol, # USDC spent for this trade
            'trigger_low': price 
        }
        
        self.positions.append(position)
        self.balance_sol -= self.trade_size_sol # Deduct USDC from balance
        self.holdings_wif += sol_amount_acquired # Add acquired SOL to holdings (still 'holdings_wif')
        
        return True, position

    def close_position(self, position, current_price, current_time, exit_type):
        """Close a specific position (selling SOL, receiving USDC)"""
        if position not in self.positions:
            return False, "Position not found"

        usdc_received_from_sale = position['sol_amount'] * current_price

        # --- ADD DEBUG PRINTS HERE ---
        print(f"DEBUG_CLOSE: Position ID: {position.get('id', 'N/A')}")
        print(f"DEBUG_CLOSE: Entry Price: {position['entry_price']:.8f}")
        print(f"DEBUG_CLOSE: Exit Price (current_price): {current_price:.8f}")
        print(f"DEBUG_CLOSE: SOL Amount: {position['sol_amount']:.8f}")
        print(f"DEBUG_CLOSE: USDC Spent for Trade: {position['usdc_spent_for_trade']:.8f}")
        print(f"DEBUG_CLOSE: USDC Received from Sale: {usdc_received_from_sale:.8f}")
        # --- END DEBUG PRINTS ---

        # Calculate profit in USDC
        usdc_profit_from_trade = usdc_received_from_sale - position['usdc_spent_for_trade']

        # Profit percentage (based on USDC spent for this trade)
        profit_percentage = usdc_profit_from_trade / position['usdc_spent_for_trade'] if position['usdc_spent_for_trade'] != 0 else 0

        # --- ADD DEBUG PRINTS FOR CALCULATED PROFITS ---
        print(f"DEBUG_CLOSE: Calculated USDC Profit: {usdc_profit_from_trade:.8f}")
        print(f"DEBUG_CLOSE: Calculated Profit Percentage: {profit_percentage:.8f}")
        # --- END DEBUG PRINTS ---

        self.balance_sol += usdc_received_from_sale 
        self.holdings_wif -= position['sol_amount'] 

        trade_record = {
            'entry_time': position['entry_time'],
            'exit_time': current_time,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'profit': float(profit_percentage), # Individual trade profit percentage
            'sol_profit': usdc_profit_from_trade, # This is profit in USDC
            'usdc_spent': position['usdc_spent_for_trade'], # Track original USDC spent
            'exit_type': exit_type,
            'candles_held': None 
        }

        self.trade_history.append(trade_record)
        self.save_trade_history() 
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
        All profit/loss values are now in USDC.
        """
        total_unrealized_profit_usdc = 0
        total_unrealized_profit_pct_open_positions = 0

        if self.positions:
            total_open_positions_usdc_spent = 0
            for pos in self.positions:
                # Ensure 'sol_amount' and 'usdc_spent_for_trade' are present in position
                # This ensures old trade data doesn't cause errors if new keys are missing.
                sol_amount = pos.get('sol_amount', 0)
                usdc_spent_for_trade = pos.get('usdc_spent_for_trade', 0)

                unrealized_profit_usdc = (current_price - pos['entry_price']) * sol_amount
                pos['unrealized_profit_usdc'] = unrealized_profit_usdc
                pos['unrealized_profit_pct'] = (unrealized_profit_usdc / usdc_spent_for_trade) * 100 if usdc_spent_for_trade != 0 else 0

                total_unrealized_profit_usdc += unrealized_profit_usdc
                total_open_positions_usdc_spent += usdc_spent_for_trade

            if total_open_positions_usdc_spent != 0:
                total_unrealized_profit_pct_open_positions = (total_unrealized_profit_usdc / total_open_positions_usdc_spent) * 100

        # Calculate cumulative profit for ALL historical trades (realized) in USDC
        # Ensure 'sol_profit' is used, as it now represents profit in USDC from closed trades.
        # Use .get() with a default of 0 to safely sum from trade_history.
        cumulative_realized_usdc_profit = sum(trade.get('sol_profit', 0) for trade in self.trade_history)

        # Total capital change in USDC: Realized profits + current unrealized profits
        total_capital_change_usdc = cumulative_realized_usdc_profit + total_unrealized_profit_usdc

        # This 'cumulative_profit' will represent the total percentage gain/loss of capital.
        # It's calculated relative to the initial USDC balance.
        # Assuming self.initial_capital_usdc is correctly set in __init__
        cumulative_profit = (total_capital_change_usdc / self.initial_capital_usdc) if self.initial_capital_usdc != 0 else 0

        return {
            'total_unrealized_profit_usdc': total_unrealized_profit_usdc,
            'total_unrealized_profit_pct_open_positions': total_unrealized_profit_pct_open_positions,
            'cumulative_profit': cumulative_profit,  # This key must match what TradeManager expects
            'num_open_positions': len(self.positions),
            'current_balance_usdc': self.balance_sol  # Still called balance_sol, but represents USDC
        }

    def get_trade_statistics(self):
        """Calculate and return general trade statistics from trade history. All profits in USDC."""
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit_pct': 0.0, # Now explicitly a percentage in decimal form
                'average_profit_pct': 0.0, 
                'best_trade_pct': 0.0, 
                'worst_trade_pct': 0.0, 
                'total_usdc_profit': 0.0, 
                'overall_capital_gain_pct': 0.0 
            }
            
        # These should already be decimals (e.g., 0.0246) from close_position
        profits_pct_decimal = [float(trade['profit']) for trade in self.trade_history] 
        usdc_profits_from_trades = [float(trade['sol_profit']) for trade in self.trade_history] 
        
        win_rate = (np.array(profits_pct_decimal) > 0).mean() * 100 
        
        average_profit_pct = np.mean(profits_pct_decimal)
        best_trade_pct = max(profits_pct_decimal)
        worst_trade_pct = min(profits_pct_decimal)
        
        total_usdc_profit = sum(usdc_profits_from_trades) 

        overall_capital_gain_pct_decimal = (total_usdc_profit / self.initial_capital_usdc) if self.initial_capital_usdc != 0 else 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit_pct': overall_capital_gain_pct_decimal, # Total profit as a decimal percentage
            'average_profit_pct': average_profit_pct, 
            'best_trade_pct': best_trade_pct, 
            'worst_trade_pct': worst_trade_pct, 
            'total_usdc_profit': total_usdc_profit, 
            'overall_capital_gain_pct': overall_capital_gain_pct_decimal # Overall capital gain as a percentage decimal
        }

    def save_trade_history(self):
        """Save trade history to CSV. Ensure profit_pct and usdc_profit columns are correctly stored."""
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            
            # Convert 'profit' to numeric and then to display format
            if 'profit' in df.columns:
                df['profit'] = pd.to_numeric(df['profit'], errors='coerce').fillna(0)
                df['profit_pct_display'] = df['profit'].apply(lambda x: f"{x:.2%}")
            
            # Ensure 'sol_profit' (now USDC profit) is numeric
            if 'sol_profit' in df.columns:
                df['sol_profit'] = pd.to_numeric(df['sol_profit'], errors='coerce').fillna(0)

            df.to_csv(self.config['paths']['trade_history'], index=False)
            print("\nTrade history saved to trade_history.csv")
        else:
            print("\nNo trade history to save.")

    def print_portfolio_status(self, current_price):
        """Print current portfolio status to console. Balances in USDC, holdings in SOL."""
        print("\n=== Portfolio Status ===")
        print(f"USDC Balance: {self.balance_sol:.3f}") # Displaying as USDC, though variable is 'balance_sol'
        print(f"SOL Holdings: {self.holdings_wif:.3f}")  # Displaying as SOL, though variable is 'holdings_wif'
        print(f"Open Positions: {len(self.positions)}")
        print(f"Total Trades: {len(self.trade_history)}")
        
        if self.positions:
            metrics = self.get_position_metrics(current_price)
            if metrics:
                print(f"\nTotal Unrealized Profit (USDC): {metrics['total_unrealized_profit_usdc']:.3f}")
                print(f"Total Unrealized Profit (%): {metrics['total_unrealized_profit_pct_open_positions']:.2f}%")
            else:
                print("\nNo detailed metrics for open positions (no open positions).")
        
        metrics_overall = self.get_position_metrics(current_price) 
        if metrics_overall and 'cumulative_profit' in metrics_overall:
             print(f"Overall Cumulative Profit: {metrics_overall['cumulative_profit']:.2f}%")
        else:
            print("Overall Cumulative Profit: 0.00%")