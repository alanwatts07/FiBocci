# modules/live_chart.py
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import threading
import queue
import traceback
import pytz # Added for robust timezone handling

class LiveChart:
    def __init__(self, config, port=8050):
        self.app = dash.Dash(__name__)
        self.port = port
        self.config = config
        self.positions = []  # Initialize open positions list
        self.trades = [] # List of completed trades
        self.df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']) # Main DataFrame for completed candles
        self.df.index.name = 'timestamp' # Set index name consistently
        
        # NEW: DataFrame to store historical Fibonacci levels for plotting as lines
        self.fib_df = pd.DataFrame(columns=['wma_fib_0', 'wma_fib_50', 'entry_threshold'])
        self.fib_df.index.name = 'timestamp' 
        
        self.fib_levels = {} # Dictionary to store only the *latest* Fibonacci level values (not historical series)
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

        # CRITICAL ADDITION: Threading Lock for ALL Shared Data
        self.data_lock = threading.Lock() 
        self.data_queue = queue.Queue() # Kept for potential future use or existing async messaging
        
        self.view_state = {
            'xrange': None,
            'yrange': None
        }

        # --- Layout Setup ---
        self.app.layout = html.Div([
            html.H1('SOL/USDC Live Trading Chart', style={'color': 'white', 'textAlign': 'center', 'paddingTop': '10px'}),
            html.Div(id='trade-stats-output', style={'color': 'white', 'marginBottom': 20, 'paddingLeft': '20px'}),
            html.Div(id='armed-status-output', style={'color': 'white', 'marginBottom': 20, 'paddingLeft': '20px'}), 
            dcc.Graph(id='live-chart', style={'height': '70vh'}), 
            
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
        @self.app.callback(
            [Output('live-chart', 'figure'),
             Output('trade-stats-output', 'children'),
             Output('armed-status-output', 'children'),
             Output('view-state-store', 'data')],
            [Input('interval-component', 'n_intervals'),
             Input('view-state-store', 'data')],
            [State('live-chart', 'relayoutData')]
        )
        def update_chart_and_status(n_intervals, stored_view_state, relayout_data):
            current_xrange = None
            current_yrange = None
            if relayout_data and 'xaxis.range[0]' in relayout_data:
                current_xrange = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
            if relayout_data and 'yaxis.range[0]' in relayout_data:
                current_yrange = [relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]

            with self.data_lock:
                df_copy = self.df.copy()
                fib_df_copy = self.fib_df.copy() # NEW: Copy the historical fib DataFrame
                trades_copy = [t.copy() for t in self.trades] 
                positions_copy = [p.copy() for p in self.positions] 
                trade_statistics_copy = self.trade_statistics.copy()
                is_armed_copy = self.is_armed
                current_candle_copy = self.current_candle.copy() 
                current_candle_copy['prices'] = self.current_candle['prices'].copy()


            # Pass fib_df_copy for plotting
            fig = self._create_figure(df_copy, fib_df_copy, trades_copy, positions_copy, current_candle_copy)
            
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

            trade_stats_layout = self._format_trade_statistics_html(trade_statistics_copy)
            armed_status_layout = self._format_armed_status_html(is_armed_copy)

            new_view_state_to_store = {
                'xrange': fig.layout.xaxis.range, 
                'yrange': fig.layout.yaxis.range
            }
            
            return fig, trade_stats_layout, armed_status_layout, new_view_state_to_store


    # modules/live_chart.py

# ... (rest of LiveChart class structure) ...

    def _create_figure(self, df_data, fib_df_data, trades_data, positions_data, current_candle_data):
        """
        Create the plotly figure using the provided data (which are copies of the shared variables).
        """
        fig = make_subplots(rows=1, cols=1)

        # Retrieve chart styles from config
        styles = self.config.get('chart_styles', {}) # Use .get() with default empty dict for safety

        # Styles for Fibonacci lines
        fib_0_style = styles.get('fib_0_line', {'color': 'purple', 'dash': 'dot', 'width': 1})
        fib_50_style = styles.get('fib_50_line', {'color': 'orange', 'dash': 'dot', 'width': 1})
        entry_threshold_style = styles.get('entry_threshold_line', {'color': 'blue', 'dash': 'dot', 'width': 1})
        current_price_style = styles.get('current_price_line', {'color': 'lime', 'dash': 'dot', 'width': 1})
        open_position_style = styles.get('open_position_line', {'color': 'blue', 'dash': 'dashdot', 'width': 2})
        trade_path_win_style = styles.get('trade_path_win_line', {'color': 'green', 'dash': 'dot', 'width': 1})
        trade_path_loss_style = styles.get('trade_path_loss_line', {'color': 'red', 'dash': 'dot', 'width': 1})

        # Add completed candles (using df_data)
        if not df_data.empty:
            fig.add_trace(
                go.Candlestick(
                    x=df_data.index,
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
        if current_candle_data['timestamp'] is not None and current_candle_data['open'] is not None:
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
                latest_price_in_current_bucket = current_candle_data['prices'][-1]
                
                end_line_x = datetime.now(pytz.utc).replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)
                
                fig.add_trace(
                    go.Scatter(
                        x=[current_candle_data['timestamp'], end_line_x],
                        y=[latest_price_in_current_bucket, latest_price_in_current_bucket],
                        mode='lines',
                        # --- Apply Current Price Line Style ---
                        line=dict(
                            color=current_price_style['color'], 
                            width=current_price_style['width'], 
                            dash=current_price_style['dash']
                        ),
                        name='Current Price Line'
                    ), row=1, col=1
                )

        # Add WMA Fibonacci levels as continuous lines (using fib_df_data)
        if not fib_df_data.empty:
            # Plot WMA Fib 0
            if 'wma_fib_0' in fib_df_data.columns and not fib_df_data['wma_fib_0'].isnull().all():
                fig.add_trace(go.Scatter(
                    x=fib_df_data.index,
                    y=fib_df_data['wma_fib_0'],
                    mode='lines',
                    name='WMA Fib 0',
                    # --- Apply Fib 0 Line Style ---
                    line=dict(
                        dash=fib_0_style['dash'], 
                        width=fib_0_style['width'], 
                        color=fib_0_style['color']
                    ),
                    hovertemplate='WMA Fib 0: %{y:.4f}<extra></extra>'
                ), row=1, col=1)

            # Plot WMA Fib 50
            if 'wma_fib_50' in fib_df_data.columns and not fib_df_data['wma_fib_50'].isnull().all():
                fig.add_trace(go.Scatter(
                    x=fib_df_data.index,
                    y=fib_df_data['wma_fib_50'],
                    mode='lines',
                    name='WMA Fib 50',
                    # --- Apply Fib 50 Line Style ---
                    line=dict(
                        dash=fib_50_style['dash'], 
                        width=fib_50_style['width'], 
                        color=fib_50_style['color']
                    ),
                    hovertemplate='WMA Fib 50: %{y:.4f}<extra></extra>'
                ), row=1, col=1)

            # Plot Entry Threshold
            if 'entry_threshold' in fib_df_data.columns and not fib_df_data['entry_threshold'].isnull().all():
                fig.add_trace(go.Scatter(
                    x=fib_df_data.index,
                    y=fib_df_data['entry_threshold'],
                    mode='lines',
                    name='Entry Threshold',
                    # --- Apply Entry Threshold Line Style ---
                    line=dict(
                        dash=entry_threshold_style['dash'], 
                        width=entry_threshold_style['width'], 
                        color=entry_threshold_style['color']
                    ),
                    hovertemplate='Entry Threshold: %{y:.4f}<extra></extra>'
                ), row=1, col=1)

        # Add Trade Markers for Closed Trades (using trades_data)
        for trade in trades_data:
            entry_time_dt = pd.to_datetime(trade['entry_time'], utc=True)
            exit_time_dt = pd.to_datetime(trade['exit_time'], utc=True)
            
            fig.add_trace(
                go.Scatter(
                    x=[entry_time_dt],
                    y=[trade['entry_price']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if trade.get('type', 'LONG') == 'LONG' else 'triangle-down',
                        size=10,
                        color='green' if trade.get('type', 'LONG') == 'LONG' else 'red',
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name=f"Entry ({trade.get('type')})",
                    showlegend=False,
                    hovertemplate=f"Entry: {trade.get('type')}<br>Time: {trade.get('entry_time')}<br>Price: {trade.get('entry_price'):.4f}"
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[exit_time_dt],
                    y=[trade['exit_price']],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='lightgrey',
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    name=f"Exit ({trade.get('profit', 0):.2%})",
                    showlegend=False,
                    hovertemplate=f"Exit: {trade.get('type')}<br>Time: {trade.get('exit_time')}<br>Price: {trade.get('exit_price'):.4f}<br>Profit: {trade.get('profit', 0):.2%}"
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[entry_time_dt, exit_time_dt],
                    y=[trade['entry_price'], trade['exit_price']],
                    mode='lines',
                    # --- Apply Trade Path Line Style ---
                    line=dict(
                        color=trade_path_win_style['color'] if trade.get('profit', 0) > 0 else trade_path_loss_style['color'], 
                        width=trade_path_win_style['width'], 
                        dash=trade_path_win_style['dash']
                    ),
                    name=f"Trade Path ({trade.get('profit', 0):.2%})",
                    showlegend=False
                ), row=1, col=1
            )

        # Add Trade Markers for Open Positions (using positions_data)
        current_time_for_open_pos = datetime.now(pytz.utc) 
        for position in positions_data:
            if position.get('entry_time') and position.get('entry_price'):
                entry_time_dt = pd.to_datetime(position['entry_time'], utc=True)
                entry_price = position['entry_price']
                
                current_price_for_open_pos_line = current_candle_data['prices'][-1] if current_candle_data['prices'] else entry_price 

                current_profit_loss_pct = ((current_price_for_open_pos_line - entry_price) / entry_price) * 100 if entry_price != 0 else 0

                fig.add_trace(
                    go.Scatter(
                        x=[entry_time_dt, current_time_for_open_pos],
                        y=[entry_price, current_price_for_open_pos_line],
                        mode='lines+markers',
                        name='Open Position', 
                        # --- Apply Open Position Line Style ---
                        line=dict(
                            color=open_position_style['color'], 
                            dash=open_position_style['dash'], 
                            width=open_position_style['width']
                        ),
                        marker=dict(
                            symbol='circle',
                            size=10,
                            color=open_position_style['color'], # Use line color for marker too
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        showlegend=True,
                        hovertemplate=(
                            f"Open Position<br>"
                            f"Entry Time: {position['entry_time']}<br>"
                            f"Entry Price: {entry_price:.4f}<br>"
                            f"Current Price: {current_price_for_open_pos_line:.4f}<br>"
                            f"Profit/Loss: {current_profit_loss_pct:.2f}%"
                        )
                    ), row=1, col=1
                )

        # Update layout
        fig.update_layout(
            title_text="SOL/USDC Live Trading Chart", 
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_dark',
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            uirevision='constant', 
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
        with self.data_lock:
            is_completed_candle = False
            if isinstance(new_data, pd.Series) and all(key in new_data for key in ['open', 'high', 'low', 'close']):
                is_completed_candle = True
            elif isinstance(new_data, dict) and all(key in new_data for key in ['open', 'high', 'low', 'close']):
                is_completed_candle = True

            if is_completed_candle:
                if isinstance(new_data, pd.Series):
                    temp_df_row = new_data.to_frame().T
                else: 
                    temp_df_row = pd.DataFrame([new_data]).set_index('timestamp')
                
                if not isinstance(temp_df_row.index, pd.DatetimeIndex):
                    temp_df_row.index = pd.to_datetime(temp_df_row.index, format='mixed', utc=True)
                temp_df_row.index.name = 'timestamp'

                self.df = pd.concat([self.df, temp_df_row])
                self.df = self.df[~self.df.index.duplicated(keep='last')]
                self.df = self.df.sort_index()

                max_rows_to_keep = 1000
                if len(self.df) > max_rows_to_keep:
                    self.df = self.df.iloc[-max_rows_to_keep:]
                
                timestamp_dt = temp_df_row.index[0] 
                self._update_current_candle_from_price(timestamp_dt, temp_df_row['close'].iloc[0])

                # NEW: Store historical WMA Fib levels
                if fib_data is not None:
                    if isinstance(new_data, pd.Series):
                        current_candle_timestamp = new_data.name
                    elif isinstance(new_data, dict):
                        current_candle_timestamp = new_data['timestamp']
                    else: 
                        current_candle_timestamp = datetime.now(pytz.utc).replace(second=0, microsecond=0)

                    if isinstance(current_candle_timestamp, str):
                        current_candle_timestamp = pd.to_datetime(current_candle_timestamp, format='mixed', utc=True)
                    elif current_candle_timestamp.tzinfo is None:
                        current_candle_timestamp = current_candle_timestamp.tz_localize('UTC')
                    else:
                        current_candle_timestamp = current_candle_timestamp.tz_convert('UTC')
                    
                    new_fib_row = pd.DataFrame([{
                        'wma_fib_0': fib_data.get('Fib 0'),
                        'wma_fib_50': fib_data.get('Fib 50'),
                        'entry_threshold': fib_data.get('Entry Threshold')
                    }], index=[current_candle_timestamp])
                    
                    new_fib_row.index.name = 'timestamp'
                    
                    self.fib_df = pd.concat([self.fib_df, new_fib_row])
                    self.fib_df = self.fib_df[~self.fib_df.index.duplicated(keep='last')]
                    self.fib_df = self.fib_df.sort_index()

                    if len(self.fib_df) > max_rows_to_keep:
                        self.fib_df = self.fib_df.iloc[-max_rows_to_keep:]

            else: # new_data is a dictionary for a 1-minute price update
                if 'timestamp' in new_data and 'price' in new_data:
                    timestamp_val = new_data['timestamp']
                    price_val = new_data['price']
                    self._update_current_candle_from_price(timestamp_val, price_val)
                else:
                    print(f"Warning: update_data received unknown data format: {new_data}")

            # Always update self.fib_levels with the LATEST values for potential internal use (not plotting continuous lines)
            if fib_data is not None:
                 self.fib_levels = fib_data

    def _update_current_candle_from_price(self, timestamp, price):
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp, format='mixed', utc=True)
        elif timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize('UTC')
        else:
            timestamp = timestamp.tz_convert('UTC')
        
        price = float(price)

        five_min_bucket = timestamp.floor('5min')

        if self.current_candle['timestamp'] is None or five_min_bucket > self.current_candle['timestamp'].floor('5min'):
            self.current_candle = {
                'timestamp': five_min_bucket,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'prices': [price]
            }
        else:
            self.current_candle['high'] = max(self.current_candle['high'], price)
            self.current_candle['low'] = min(self.current_candle['low'], price)
            self.current_candle['close'] = price
            self.current_candle['prices'].append(price)

    def update_fib_levels(self, fib_data): # Method kept for clarity, but update_data handles it too
        with self.data_lock:
            self.fib_levels = fib_data

    def update_trades(self, trades):
        with self.data_lock:
            self.trades = [t.copy() for t in trades]

    def update_positions(self, positions): 
        with self.data_lock:
            self.positions = [p.copy() for p in positions]

    def update_trade_statistics(self, stats):
        with self.data_lock:
            self.trade_statistics = stats.copy()

    def update_is_armed_status(self, is_armed_status):
        with self.data_lock:
            self.is_armed = is_armed_status

    def _format_trade_statistics_html(self, stats):
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

    def start(self):
        """This method is called by the TradingBot to start the Dash server in a new thread."""
        self.app.run(debug=False, port=self.port, host='0.0.0.0')