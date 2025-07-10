# modules/live_chart.py
from dash import dcc, html  # Ensure you have imported html
from dash.dependencies import Input, Output, State
import threading
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import threading
import queue
import traceback

class LiveChart:
    def __init__(self, port=8051):
        self.app = dash.Dash(__name__)
        self.port = port
        self.positions = []  # Initialize open positions list
        self.trades = []  
        self.data_queue = queue.Queue()
        self.df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
        self.fib_levels = {}
        self.current_candle = {
            'timestamp': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'prices': []
        }
        self.trade_statistics = {}
        self.stats_lock = threading.Lock()         # Add view state storage
        self.view_state = {
            'xrange': None,
            'yrange': None
        }
        self.is_armed = False
        # Setup the layout
        self.app.layout = html.Div([
            html.H1('Live Trading Chart', style={'color': 'white'}),
            html.Div(id='trade-stats', style={'color': 'white', 'marginBottom': 20}), 
            html.Div(id='armed-status', style={'color': 'white', 'marginBottom': 20}),  # New Div for armed status 
            dcc.Graph(id='live-chart'),
            dcc.Interval(
                id='interval-component',
                interval=3000,  # 1 second updates
                n_intervals=0
            ),
            # Add store component for view state
            dcc.Store(id='view-state')
        ], style={'backgroundColor': '#1a1a1a'})

        # Setup callbacks
        @self.app.callback(
            Output('live-chart', 'figure'),
            [Input('interval-component', 'n_intervals'),
            Input('view-state', 'data')]
        )
        def update_chart(n, view_state):
            # If needed, handle view state restoration
            
            fig = self.create_figure()  # This will include the current candle and all previous ones
            return fig
        
        @self.app.callback(
            Output('view-state', 'data'),
            [Input('live-chart', 'relayoutData')],
            prevent_initial_call=True
        )
        def save_view_state(relayout_data):
            if relayout_data is None:
                return dash.no_update
            
            # Save zoom level and pan position
            view_state = {}
            if 'xaxis.range[0]' in relayout_data:
                view_state['xrange'] = [
                    relayout_data['xaxis.range[0]'],
                    relayout_data['xaxis.range[1]']
                ]
            if 'yaxis.range[0]' in relayout_data:
                view_state['yrange'] = [
                    relayout_data['yaxis.range[0]'],
                    relayout_data['yaxis.range[1]']
                ]
            return view_state
            # Add this callback to update trade statistics
        @self.app.callback(
            Output('trade-stats', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_trade_stats(n):
            return self.render_trade_statistics()
        
    def update_is_armed_status(self, is_armed):
        """Update the armed status"""
        with self.stats_lock:
            self.is_armed = is_armed
    def create_figure(self): 
        """Create the plotly figure""" 
        fig = make_subplots(rows=2, cols=1,  
                        shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        subplot_titles=('Price', 'Time'), 
                        row_heights=[0.8, 0.2]) 

        # Add completed candles 
        if not self.df.empty: 
            fig.add_trace( 
                go.Candlestick( 
                    x=self.df['timestamp'], 
                    open=self.df['open'], 
                    high=self.df['high'], 
                    low=self.df['low'], 
                    close=self.df['close'], 
                    name='Completed Candles' 
                ), 
                row=1, col=1 
            ) 

        # Add current forming candle if exists 
        if self.current_candle['timestamp'] is not None:
            offset_time = timedelta(seconds=300)  # Adjust as needed
            print("Adding Current Candle:", self.current_candle)
            fig.add_trace(
                go.Candlestick(
                    x=[self.current_candle['timestamp'] + offset_time],
                    open=[self.current_candle['open']],
                    high=[self.current_candle['high']],
                    low=[self.current_candle['low']],
                    close=[self.current_candle['close']],
                    name='Current Candle',
                    increasing_line_color='rgba(0, 255, 0, 0.5)',
                    decreasing_line_color='rgba(255, 0, 0, 0.5)'
                ),
                row=1, col=1
            )

            # Add price line for current candle 
            if self.current_candle['prices']: 
                times = [self.current_candle['timestamp'] + timedelta(seconds=300)  
                        for i in range(len(self.current_candle['prices']))] 
                fig.add_trace( 
                    go.Scatter( 
                        x=times, 
                        y=self.current_candle['prices'], 
                        mode='lines', 
                        line=dict(color='white', width=1), 
                        name='Current Price' 
                    ), row=1, col=1
                ) 

        # Add Fibonacci levels 
        for name, series in self.fib_levels.items():
            if series is not None:
                fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode='lines',
                        name=name,
                        line=dict(dash='dash')
                    ), row=1, col=1
                )

        # Add Trade Markers for Closed Trades
        for trade in self.trades:
            fig.add_trace(
                go.Scatter(
                    x=[trade['entry_time'], trade['exit_time']],
                    y=[trade['entry_price'], trade['exit_price']],
                    mode='markers+lines',
                    name=f"Trade ({trade['profit']:.2%})",
                    line=dict(color='green' if trade['profit'] > 0 else 'red'),
                    marker=dict(
                        symbol=['triangle-up', 'triangle-down'],
                        size=10,
                        color=['lime', 'red']
                    )
                ), row=1, col=1
            )

        # **Add Trade Markers for Open Positions**
        current_time = datetime.now()
        for position in self.positions:
            entry_time = position['entry_time']
            entry_price = position['entry_price']
            # Use the latest close price as the current price
            current_price = self.current_candle['close'] if self.current_candle['close'] is not None else entry_price
            
            fig.add_trace(
                go.Scatter(
                    x=[entry_time, current_time],
                    y=[entry_price, current_price],
                    mode='lines+markers',
                    name='Open Position',
                    line=dict(color='blue', dash='dot'),
                    marker=dict(
                        symbol=['triangle-up', 'circle'],
                        size=10,
                        color=['cyan', 'cyan']
                    ),
                    showlegend=True
                ), row=1, col=1
            )

        # Update layout
        fig.update_layout(
            title_text="WIF/SOL Live Trading Chart",
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_dark',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            uirevision='constant',
            updatemenus=[{
                'buttons': [
                    {'label': '1h', 'method': 'relayout','args': [{'xaxis.range': [datetime.now() - timedelta(hours=1), datetime.now()]}]},
                    {'label': '4h','method': 'relayout','args': [{'xaxis.range': [datetime.now() - timedelta(hours=4), datetime.now()]}]},
                    {'label': '1d','method': 'relayout','args': [{'xaxis.range': [datetime.now() - timedelta(days=1),datetime.now()]}]},
                    {'label': 'All','method': 'relayout','args': [{'xaxis.autorange': True}]},
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 10},
                'showactive': True,
                'type': 'buttons',
                'x': 0.1,
                'y': 1.1,
                'xanchor': 'right',
                'yanchor': 'top'
            }]
        )

        # Update axes
    
        fig.update_xaxes(gridcolor='#333333')
        fig.update_yaxes(gridcolor='#333333')

        return fig
    def update_trade_statistics(self, stats):
        """Update the trade statistics"""
        with self.stats_lock:
            self.trade_statistics = stats.copy()
             # Retrieve the is_armed status  # Use copy to avoid shared references


    def update_trades(self, trades, positions): 
        """Store trade markers to be plotted""" 
        self.trades = trades  # Store trades data to be used in 
        self.positions = positions  # Store open positions

    def update_price(self, timestamp, price):
        current_time = pd.to_datetime(arg=timestamp, format='mixed')
        candle_start = current_time.floor('5min')  # Assume 5-minute intervals

        if (self.current_candle['timestamp'] is None or 
            candle_start != self.current_candle['timestamp']):
            
            if (self.current_candle['timestamp'] is not None and 
                self.current_candle['open'] is not None and 
                self.current_candle['close'] is not None):
                self.save_completed_candle()

            self.current_candle = {
                'timestamp': candle_start,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'prices': [price]
            }
        else:
            # Update current candle values
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price  # Always update close
            self.current_candle['prices'].append(price)

    def save_completed_candle(self):
        """Save the current candle to completed candles"""
        print('saving completed cadle')
        new_row = pd.DataFrame({
            'timestamp': [self.current_candle['timestamp']],
            'open': [self.current_candle['open']],
            'high': [self.current_candle['high']],
            'low': [self.current_candle['low']],
            'close': [self.current_candle['close']]
        })
        

    def update_data(self, new_candle, fib_data=None):
        """Update the chart data"""
        try:
            timestamp = new_candle.name if hasattr(new_candle, 'name') else new_candle['timestamp']
            
            # If it's a completed candle
            if isinstance(new_candle, pd.Series):
                new_row = pd.DataFrame({
                    'timestamp': [timestamp],
                    'open': [new_candle['open']],
                    'high': [new_candle['high']],
                    'low': [new_candle['low']],
                    'close': [new_candle['close']]
                })
                
                self.df = pd.concat([self.df, new_row])
                #self.df = self.df.tail(200)  # Keep only last 200 candles
                
                # Update current candle with latest price
                self.update_price(timestamp, new_candle['close'])
                
            # If it's a price update
            else:
                self.update_price(timestamp, new_candle['price'])
            
            # Update Fibonacci levels
            if fib_data is not None:
                self.fib_levels = fib_data
                
        except Exception as e:
            print(f"Error updating live chart data: {e}")
            traceback.print_exc()

    
    def update_fib_levels(self, fib_data):
        """Update Fibonacci levels"""
        self.fib_levels = fib_data

    def render_trade_statistics(self):
        """Generate HTML content for trade statistics"""
        with self.stats_lock:
            stats = self.trade_statistics.copy()
            is_armed = self.is_armed
        if not stats:
            return html.Div("No trade statistics available yet.")
        else:
           stats_content = html.Div([
                html.H4("Trade Statistics"),
                html.P(f"Total Trades: {stats.get('total_trades', 0)}"),
                html.P(f"Win Rate: {stats.get('win_rate', 0.0):.2f}%"),
                html.P(f"Total Profit: {stats.get('total_profit', 0.0):.2%}"),
                html.P(f"Average Profit per Trade: {stats.get('average_profit', 0.0):.2%}"),
                html.P(f"Best Trade: {stats.get('best_trade', 0.0):.2%}"),
                html.P(f"Worst Trade: {stats.get('worst_trade', 0.0):.2%}"),
                html.P(f"Total SOL Profit: {stats.get('total_sol_profit', 0.0):.3f} SOL"),
            ])

        armed_status_content = html.Div([
            html.H4("Bot Status"),
            html.P(f"Entry Switch Armed: {'Yes' if is_armed else 'No'}"),
        ])

        # Return combined content
        return html.Div([stats_content, armed_status_content])
        
    def start(self):
        """Start the chart server in a separate thread"""
        def run_server():
            self.app.run_server(debug=False, port=self.port)

        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()