# modules/live_chart.py
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html # Correct modern import for Dash 2.0+
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import threading
import queue # queue.Queue is not directly used for shared chart data in this revised version, but kept for minimal change
import traceback
import pytz # Added for robust timezone handling

class LiveChart:
    def __init__(self, port=8050):
        self.app = dash.Dash(__name__)
        self.port = port
        
        # --- ORIGINAL SHARED DATA VARIABLES ---
        # Initialize with appropriate types/structures as per your original code
        self.positions = []  # Initialize open positions list
        self.trades = [] # List of completed trades
        self.df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']) # Main DataFrame for completed candles
        # Ensure df has a datetime index for consistency with Plotly and DataHandler
        self.df.index.name = 'timestamp' # Set index name consistently
        
        self.fib_levels = {} # Dictionary of Fibonacci levels
        self.current_candle = { # Dictionary for the currently forming candle
            'timestamp': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'prices': [] # List of 1-min prices in the current bucket for the current price line
        }
        self.trade_statistics = {} # Dictionary for trade statistics
        self.is_armed = False # Boolean for armed status

        # --- CRITICAL ADDITION: Threading Lock for ALL Shared Data ---
        # This lock MUST protect all variables above that are accessed by both
        # the TradingBot's thread (updating) and Dash's callback thread (reading).
        self.data_lock = threading.Lock() 

        # queue.Queue is not used for chart data directly in this pattern, but if your
        # system uses it for other async messaging, it's fine to keep.
        self.data_queue = queue.Queue() 
        
        # Add view state storage - managed by callbacks, so not directly shared like data
        self.view_state = {
            'xrange': None,
            'yrange': None
        }

        # --- Layout Setup ---
        self.app.layout = html.Div([
            html.H1('Live Trading Chart', style={'color': 'white', 'textAlign': 'center', 'paddingTop': '10px'}),
            # Changed div IDs to match callback outputs and avoid conflicts
            html.Div(id='trade-stats-output', style={'color': 'white', 'marginBottom': 20, 'paddingLeft': '20px'}),
            html.Div(id='armed-status-output', style={'color': 'white', 'marginBottom': 20, 'paddingLeft': '20px'}), 
            dcc.Graph(id='live-chart', style={'height': '70vh'}), # Make chart responsive
            
            # --- CRITICAL: Interval component for live updates ---
            dcc.Interval(
                id='interval-component',
                interval=5000, # 5 seconds update. Adjust as needed (e.g., 3000ms for 3s)
                n_intervals=0
            ),
            dcc.Store(id='view-state-store') # Store component for view state
        ], style={'backgroundColor': '#1a1a1a', 'fontFamily': 'Arial, sans-serif'})

        # --- Register Callbacks ---
        self._register_callbacks()

    def _register_callbacks(self):
        # Main callback to update the chart and status panels
        @self.app.callback(
            [Output('live-chart', 'figure'),
             Output('trade-stats-output', 'children'), # Use the new output ID
             Output('armed-status-output', 'children'), # Use the new output ID
             Output('view-state-store', 'data')], # Also output view state to store it
            [Input('interval-component', 'n_intervals'),
             Input('view-state-store', 'data')], # Input current view state for restoration
            [State('live-chart', 'relayoutData')] # Get current zoom/pan from user interaction
        )
        def update_chart_and_status(n_intervals, stored_view_state, relayout_data):
            # Save current view state from user interaction before reading new data
            current_xrange = None
            current_yrange = None
            if relayout_data and 'xaxis.range[0]' in relayout_data:
                current_xrange = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
            if relayout_data and 'yaxis.range[0]' in relayout_data:
                current_yrange = [relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]

            # Acquire lock to read ALL shared data safely
            with self.data_lock:
                # Copy data from shared instance variables to local variables for rendering
                df_copy = self.df.copy()
                fib_levels_copy = {k: v.copy() if isinstance(v, pd.Series) else v for k,v in self.fib_levels.items()} 
                trades_copy = [t.copy() for t in self.trades] 
                positions_copy = [p.copy() for p in self.positions] 
                trade_statistics_copy = self.trade_statistics.copy()
                is_armed_copy = self.is_armed
                # current_candle is a dict, so a shallow copy is often sufficient, but deep copy if nested mutable objects
                current_candle_copy = self.current_candle.copy() 
                # Ensure prices list inside current_candle_copy is also copied
                current_candle_copy['prices'] = self.current_candle['prices'].copy()


            # Create the figure with the latest data
            # Passing copies ensures that the data doesn't change while Plotly is building the figure
            fig = self._create_figure(df_copy, fib_levels_copy, trades_copy, positions_copy, current_candle_copy)
            
            # Restore view state
            if stored_view_state:
                if 'xrange' in stored_view_state and stored_view_state['xrange'] is not None:
                    fig.update_layout(xaxis_range=stored_view_state['xrange'])
                if 'yrange' in stored_view_state and stored_view_state['yrange'] is not None:
                    fig.update_layout(yaxis_range=stored_view_state['yrange'])
            # Override with user's current interaction if available
            if current_xrange:
                fig.update_layout(xaxis_range=current_xrange)
            if current_yrange:
                fig.update_layout(yaxis_range=current_yrange)

            # Format HTML content for status panels
            trade_stats_layout = self._format_trade_statistics_html(trade_statistics_copy)
            armed_status_layout = self._format_armed_status_html(is_armed_copy)

            # Store the current view state for next update (Plotly figure's ranges)
            new_view_state_to_store = {
                'xrange': fig.layout.xaxis.range, 
                'yrange': fig.layout.yaxis.range
            }
            
            return fig, trade_stats_layout, armed_status_layout, new_view_state_to_store


    def _create_figure(self, df_data, fib_levels_data, trades_data, positions_data, current_candle_data):
        """
        Create the plotly figure using the provided data (which are copies of the shared variables).
        """
        fig = make_subplots(rows=1, cols=1) # Removed subplot_titles for simplicity as you only have one subplot here

        # Add completed candles (using df_data)
        if not df_data.empty:
            fig.add_trace(
                go.Candlestick(
                    x=df_data.index, # Assuming timestamp is now the index
                    open=df_data['open'],
                    high=df_data['high'],
                    low=df_data['low'],
                    close=df_data['close'],
                    name='Completed Candles',
                    increasing_line_color='green', 
                    decreasing_line_color='red'
                ),
                row=1, col=1
            )

        # Add current forming candle (using current_candle_data)
        # This will show the partial candle currently being built
        if current_candle_data['timestamp'] is not None and current_candle_data['open'] is not None:
            # The current candle should be at its own bucket start time
            candle_x = [current_candle_data['timestamp']]
            fig.add_trace(
                go.Candlestick(
                    x=candle_x,
                    open=[current_candle_data['open']],
                    high=[current_candle_data['high']],
                    low=[current_candle_data['low']],
                    close=[current_candle_data['close']],
                    name='Current Candle',
                    increasing_line_color='rgba(0, 255, 0, 0.5)', # Semi-transparent
                    decreasing_line_color='rgba(255, 0, 0, 0.5)' # Semi-transparent
                ),
                row=1, col=1
            )
            
            # Add price line for current candle's last recorded price (if it has prices in its bucket)
            if current_candle_data['prices']:
                # The 'prices' list contains the 1-min prices within the current candle's bucket
                latest_price_in_current_bucket = current_candle_data['prices'][-1]
                
                # To make the current price line visible, make it span a small recent range
                # or extend from the last completed candle.
                # Use current_candle_data['timestamp'] as start and a little bit into the future
                start_line_x = current_candle_data['timestamp']
                end_line_x = datetime.now(pytz.utc).replace(second=0, microsecond=0) + pd.Timedelta(minutes=1) # Extend to current minute + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=[start_line_x, end_line_x],
                        y=[latest_price_in_current_bucket, latest_price_in_current_bucket],
                        mode='lines',
                        line=dict(color='lime', width=1, dash='dot'),
                        name='Current Price Line' # Use a distinct name for clarity
                    ), row=1, col=1
                )


        # Add Fibonacci levels (using fib_levels_data)
        for name, value in fib_levels_data.items():
            if value is not None:
                # If it's a pandas Series (e.g., historical WMA Fib levels), plot it as a line
                if isinstance(value, pd.Series):
                    fig.add_trace(go.Scatter(
                        x=value.index,
                        y=value.values,
                        mode='lines',
                        name=name,
                        line=dict(dash='dash', width=1)
                    ), row=1, col=1)
                else: # Assume it's a single value (e.g., latest fib level)
                    fig.add_hline(y=value, line_dash="dot", 
                                  line_color="purple" if "0" in name else ("orange" if "50" in name else "blue"),
                                  annotation_text=name, annotation_position="top left", name=name,
                                  row=1, col=1)

        # Add Trade Markers for Closed Trades (using trades_data)
        for trade in trades_data:
            entry_time_dt = pd.to_datetime(trade['entry_time'], utc=True)
            exit_time_dt = pd.to_datetime(trade['exit_time'], utc=True)
            
            # Use separate traces for entry and exit markers for better control and legend
            fig.add_trace(
                go.Scatter(
                    x=[entry_time_dt],
                    y=[trade['entry_price']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if trade.get('type', 'LONG') == 'LONG' else 'triangle-down', # Entry symbol
                        size=10,
                        color='green' if trade.get('type', 'LONG') == 'LONG' else 'red',
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name=f"Entry ({trade.get('type')})",
                    showlegend=False, # Show in hover, not always in legend
                    hovertemplate=f"Entry: {trade.get('type')}<br>Time: {trade.get('entry_time')}<br>Price: {trade.get('entry_price'):.4f}"
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[exit_time_dt],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(
                        symbol='circle', # Exit symbol
                        size=10,
                        color='lightgrey',
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name=f"Exit ({trade.get('profit', 0):.2%})",
                    showlegend=False, # Show in hover, not always in legend
                    hovertemplate=f"Exit: {trade.get('type')}<br>Time: {trade.get('exit_time')}<br>Price: {trade.get('exit_price'):.4f}<br>Profit: {trade.get('profit', 0):.2%}"
                ), row=1, col=1
            )
            # Add a line connecting entry and exit
            fig.add_trace(
                go.Scatter(
                    x=[entry_time_dt, exit_time_dt],
                    y=[trade['entry_price'], trade['exit_price']],
                    mode='lines',
                    line=dict(color='green' if trade.get('profit', 0) > 0 else 'red', width=1, dash='dot'),
                    name=f"Trade Path ({trade.get('profit', 0):.2%})",
                    showlegend=False # Don't clutter legend with every trade path
                ), row=1, col=1
            )


        # Add Trade Markers for Open Positions (using positions_data)
        current_time_for_open_pos = datetime.now(pytz.utc) # Ensure timezone awareness
        for position in positions_data:
            if position.get('entry_time') and position.get('entry_price'):
                entry_time_dt = pd.to_datetime(position['entry_time'], utc=True)
                entry_price = position['entry_price']
                
                # Use the latest price from the current_candle for the end of the open position line
                current_price_for_open_pos_line = current_candle_data['prices'][-1] if current_candle_data['prices'] else entry_price 

                fig.add_trace(
                    go.Scatter(
                        x=[entry_time_dt, current_time_for_open_pos],
                        y=[entry_price, current_price_for_open_pos_line],
                        mode='lines+markers',
                        name='Open Position', # This will appear once in the legend
                        line=dict(color='blue', dash='dashdot', width=2),
                        marker=dict(
                            symbol='circle', # Entry marker
                            size=10,
                            color='cyan',
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        showlegend=True,
                        hovertemplate=f"Open Position<br>Entry Time: {position['entry_time']}<br>Entry Price: {position['entry_price']:.4f}<br>Current Price: {current_price_for_open_pos_line:.4f}"
                    ), row=1, col=1
                )

        # Update layout
        fig.update_layout(
            title_text="SOL/USDC Live Trading Chart", # Still hardcoded, but you can make it dynamic
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_dark',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            uirevision='constant', # This helps preserve zoom/pan when data updates
            updatemenus=[{
                'buttons': [
                    {'label': '1h', 'method': 'relayout','args': [{'xaxis.range': [datetime.now(pytz.utc) - timedelta(hours=1), datetime.now(pytz.utc)]}]},
                    {'label': '4h','method': 'relayout','args': [{'xaxis.range': [datetime.now(pytz.utc) - timedelta(hours=4), datetime.now(pytz.utc)]}]},
                    {'label': '1d','method': 'relayout','args': [{'xaxis.range': [datetime.now(pytz.utc) - timedelta(days=1), datetime.now(pytz.utc)]}]},
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

        fig.update_xaxes(gridcolor='#333333', zeroline=False)
        fig.update_yaxes(gridcolor='#333333', zeroline=False)

        return fig

    # --- Methods called by TradingBot to update chart data ---
    # Re-introducing original update_data signature, but with robust internal handling

    def update_data(self, new_data, fib_data=None):
        """
        Updates the chart data. This method intelligently handles either:
        1. A completed 5-min candle (pandas Series or DataFrame)
        2. A 1-minute raw price update (dictionary like {'timestamp': ..., 'price': ...})

        Args:
            new_data: A pandas Series (for completed candle) or a dictionary
                      (for 1-min price update, e.g., {'timestamp': ..., 'price': ...}).
            fib_data: Dictionary of Fibonacci levels (e.g., {'Fib 0': ..., 'Fib 50': ..., 'Entry Threshold': ...}).
        """
        with self.data_lock: # Protect all shared data updates
            is_completed_candle = False
            # Determine if new_data is a completed candle (has 'open', 'high', 'low', 'close')
            if isinstance(new_data, pd.Series) and all(key in new_data for key in ['open', 'high', 'low', 'close']):
                is_completed_candle = True
            # Also check if it's a dict that looks like a completed candle (less common from TradingBot)
            elif isinstance(new_data, dict) and all(key in new_data for key in ['open', 'high', 'low', 'close']):
                is_completed_candle = True

            if is_completed_candle:
                # If it's a completed candle, append it to self.df
                if isinstance(new_data, pd.Series):
                    temp_df_row = new_data.to_frame().T
                else: # Assume it's a dictionary for a completed candle
                    temp_df_row = pd.DataFrame([new_data]).set_index('timestamp') # Assuming timestamp is a key
                
                # Ensure the new candle has a DatetimeIndex
                if not isinstance(temp_df_row.index, pd.DatetimeIndex):
                    temp_df_row.index = pd.to_datetime(temp_df_row.index, format='mixed', utc=True)
                temp_df_row.index.name = 'timestamp' # Ensure consistent index name

                self.df = pd.concat([self.df, temp_df_row])
                self.df = self.df[~self.df.index.duplicated(keep='last')] # Remove duplicates if same timestamp
                self.df = self.df.sort_index() # Ensure chronological order

                # Keep only last N candles for memory management
                max_rows_to_keep = 1000
                if len(self.df) > max_rows_to_keep:
                    self.df = self.df.iloc[-max_rows_to_keep:]
                
                # print(f"LiveChart: df (ohlc_data) updated. Last candle: {self.df.index[-1] if not self.df.empty else 'N/A'}")
                
                # Also update current_candle with the close of this completed candle
                # This ensures the current price line starts correctly on the new bucket
                timestamp_dt = temp_df_row.index[0] # Get the timestamp from the DataFrame
                self._update_current_candle_from_price(timestamp_dt, temp_df_row['close'].iloc[0])


            else: # Assume new_data is a dictionary for a 1-minute price update
                if 'timestamp' in new_data and 'price' in new_data:
                    timestamp_val = new_data['timestamp']
                    price_val = new_data['price']
                    self._update_current_candle_from_price(timestamp_val, price_val) # Use the internal helper
                else:
                    print(f"Warning: update_data received unknown data format: {new_data}")

            # Update Fibonacci levels if provided
            if fib_data is not None:
                self.fib_levels = fib_data # Assuming fib_data is already a copy or safe to assign

    def _update_current_candle_from_price(self, timestamp, price):
        """
        Internal helper to update the self.current_candle based on a single price.
        This logic was extracted from the old update_price method.
        """
        # Ensure timestamp is datetime object and timezone-aware (UTC)
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp, format='mixed', utc=True)
        elif timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_convert('UTC')
        
        price = float(price)

        five_min_bucket = timestamp.floor('5min')

        if self.current_candle['timestamp'] is None or five_min_bucket > self.current_candle['timestamp'].floor('5min'):
            # Initialize a new current_candle if this is the first price or a new 5-min bucket starts
            self.current_candle = {
                'timestamp': five_min_bucket, # The start of the 5-min bucket
                'open': price,
                'high': price,
                'low': price,
                'close': price, # Close of the in-progress candle
                'prices': [price] # List of all 1-min prices in this bucket
            }
            # print(f"LiveChart: _update_current_candle_from_price: Started new current_candle for {five_min_bucket}")
        else:
            # Update existing current_candle
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price # Always the latest price
            self.current_candle['prices'].append(price)
            # print(f"LiveChart: _update_current_candle_from_price: Updated current_candle for {five_min_bucket} with price {price}")


    def update_fib_levels(self, fib_data): # Separate method kept for clarity if TradingBot wants to call it distinctly
        """Update Fibonacci levels"""
        with self.data_lock:
            self.fib_levels = fib_data # Assuming fib_data is already a copy or safe to assign

    def update_trades(self, trades):
        """Store trade markers to be plotted"""
        with self.data_lock:
            self.trades = [t.copy() for t in trades] # Deep copy to avoid shared references and mutation issues

    def update_positions(self, positions): 
        """Store open positions"""
        with self.data_lock:
            self.positions = [p.copy() for p in positions] # Deep copy

    def update_trade_statistics(self, stats):
        """Update the trade statistics"""
        with self.data_lock:
            self.trade_statistics = stats.copy() # Use copy to avoid shared references

    def update_is_armed_status(self, is_armed_status): # Renamed input arg to avoid clash with self.is_armed
        """Update the armed status"""
        with self.data_lock:
            self.is_armed = is_armed_status

    # Helper functions to render HTML for status panels
    def _format_trade_statistics_html(self, stats):
        # Use the passed 'stats' argument directly, which is already a copy
        if not stats:
            return html.Div([
                html.H4("Trade Statistics", style={'marginBottom': '5px'}),
                html.P("No trade statistics available yet.", style={'fontSize': '0.9em'})
            ], style={'border': '1px solid #444', 'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#222'})

        return html.Div([
            html.H4("Trade Statistics", style={'marginBottom': '5px'}),
            html.P(f"Total Trades: {stats.get('total_trades', 0)}", style={'fontSize': '0.9em'}),
            html.P(f"Win Rate: {stats.get('win_rate', 0.0):.2f}%", style={'fontSize': '0.9em'}),
            html.P(f"Total Profit: {stats.get('total_profit', 0.0):.2f}%", style={'fontSize': '0.9em'}),
            html.P(f"Average Profit per Trade: {stats.get('average_profit', 0.0):.2f}%", style={'fontSize': '0.9em'}),
            html.P(f"Best Trade: {stats.get('best_trade', 0.0):.2f}%", style={'fontSize': '0.9em'}),
            html.P(f"Worst Trade: {stats.get('worst_trade', 0.0):.2f}%", style={'fontSize': '0.9em'}),
            html.P(f"Total SOL Profit: {stats.get('total_sol_profit', 0.0):.3f} SOL", style={'fontSize': '0.9em'}),
        ], style={'border': '1px solid #444', 'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#222'})

    def _format_armed_status_html(self, is_armed_status): 
        status_color = 'lightgreen' if is_armed_status else 'lightcoral'
        return html.Div([
            html.H4("Bot Status", style={'marginBottom': '5px'}),
            html.P(f"Entry Switch Armed: {'Yes' if is_armed_status else 'No'}", style={'fontSize': '0.9em', 'color': status_color}),
        ], style={'border': '1px solid #444', 'padding': '10px', 'borderRadius': '5px', 'backgroundColor': '#222'})

    # The start() method is now implicitly called by the threading.Thread in TradingBot.
    # We remove the inner `run_server` function and just keep the app.run call for clarity.
    def start(self):
        """This method is called by the TradingBot to start the Dash server in a new thread."""
        self.app.run(debug=False, port=self.port, host='0.0.0.0')
