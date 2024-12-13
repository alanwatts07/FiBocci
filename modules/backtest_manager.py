# modules/backtest_manager.py

import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class BacktestManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.trades = []
        self.equity_curve = []
        self.initial_balance = config['backtesting']['initial_balance']
        self.current_balance = self.initial_balance

    def prepare_data(self, csv_path):
        """Prepare historical data for backtesting"""
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # If the data is already in OHLC format
        if all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.set_index('timestamp')
        
        # If the data is in raw price format
        elif 'price' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Create 5-minute OHLC candles
            ohlc = df.resample('5min').agg({
                'price': [
                    ('open', 'first'),
                    ('high', 'max'),
                    ('low', 'min'),
                    ('close', 'last')
                ]
            })
            
            # Flatten column names
            ohlc.columns = ohlc.columns.get_level_values(1)
            
            return ohlc
        
        else:
            raise ValueError("CSV file must contain either OHLC columns or a 'price' column")

    def run_backtest(self, historical_data, trade_manager):
        """Run backtest on historical data"""
        for timestamp, row in historical_data.iterrows():
            # Process each candle
            trade_manager.check_for_trades(row, historical_data)

    def calculate_equity(self, balance, positions, current_price):
        """Calculate total equity including open positions"""
        position_value = sum(pos['size'] * current_price for pos in positions)
        return balance + position_value

    def execute_buy(self, signal, timestamp, candle):
        """Execute a buy trade in backtest"""
        price = candle['close']
        size = self.config['backtesting']['trade_size'] / price
        cost = size * price
        
        self.current_balance -= cost
        
        position = {
            'entry_time': timestamp,
            'entry_price': price,
            'size': size,
            'cost': cost
        }
        
        self.trades.append({
            'type': 'buy',
            'time': timestamp,
            'price': price,
            'size': size,
            'balance': self.current_balance
        })
        
        return position

    def execute_sell(self, signal, timestamp, candle, positions):
        """Execute a sell trade in backtest"""
        price = candle['close']
        for position in positions:
            profit = (price - position['entry_price']) / position['entry_price']
            revenue = position['size'] * price
            self.current_balance += revenue
            
            self.trades.append({
                'type': 'sell',
                'time': timestamp,
                'price': price,
                'size': position['size'],
                'profit': profit,
                'balance': self.current_balance
            })
        
        positions.clear()

    def generate_backtest_report(self):
        """Generate detailed backtest report"""
        if not self.trades:
            self.logger.print_info("No trades executed during backtest")
            return
        
        # Calculate statistics
        trades_df = pd.DataFrame(self.trades)
        profits = trades_df[trades_df['type'] == 'sell']['profit']
        
        stats = {
            'total_trades': len(profits),
            'win_rate': (profits > 0).mean() * 100,
            'avg_profit': profits.mean() * 100,
            'max_profit': profits.max() * 100,
            'max_loss': profits.min() * 100,
            'final_balance': self.current_balance,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance * 100
        }
        
        # Generate report
        report = f"""
        [bold cyan]═══ Backtest Results ═══[/bold cyan]
        Period: {self.trades[0]['time']} to {self.trades[-1]['time']}
        Initial Balance: {self.initial_balance:.3f} SOL
        Final Balance: {stats['final_balance']:.3f} SOL
        Total Return: {stats['total_return']:.2f}%
        
        Total Trades: {stats['total_trades']}
        Win Rate: {stats['win_rate']:.2f}%
        Average Profit: {stats['avg_profit']:.2f}%
        Best Trade: {stats['max_profit']:.2f}%
        Worst Trade: {stats['max_loss']:.2f}%
        """
        
        self.logger.print_info(report)
        
        # Save detailed results
        trades_df.to_csv('backtest_trades.csv', index=False)
        self.equity_curve.to_csv('equity_curve.csv', index=False)
        
        # Plot equity curve
        self.plot_equity_curve()

    def plot_equity_curve(self):
        """Plot equity curve and drawdown"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve['timestamp'],
                y=self.equity_curve['equity'],
                name='Portfolio Value'
            ),
            row=1, col=1
        )
        
        # Calculate and plot drawdown
        rolling_max = self.equity_curve['equity'].cummax()
        drawdown = (self.equity_curve['equity'] - rolling_max) / rolling_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve['timestamp'],
                y=drawdown,
                name='Drawdown %',
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Backtest Results',
            yaxis_title='Portfolio Value (SOL)',
            yaxis2_title='Drawdown %'
        )
        
        fig.write_html('backtest_results.html')