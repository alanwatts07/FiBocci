# modules/trade_manager.py
import pandas as pd
from modules.logger import BotLogger
import pytz # Import pytz for timezone awareness consistency

class TradeManager:
    def __init__(self, config, position_manager, fib_calculator):
        self.config = config
        self.position_manager = position_manager
        self.fib_calculator = fib_calculator
        self.entry_switch_armed = False
        self.last_candle_below_fib = False 
        self.logger = BotLogger()

    def check_for_trades(self, candle, five_min_data):
        if len(five_min_data) < 65:
            self.logger.print_info(f"DEBUG: Not enough 5-min data ({len(five_min_data)} candles) for full WMA Fibs. Skipping trade checks.")
            return

        current_price_close = candle['close']
        current_time = candle.name

        self._check_exit_conditions(current_price_close, current_time, five_min_data)

        if len(self.position_manager.positions) < self.config['trading']['max_positions']:
            self._check_entry_conditions(candle, five_min_data)
        else:
            self.logger.print_info(f"DEBUG: Max positions reached ({len(self.position_manager.positions)}). Skipping entry checks.")

    def _check_entry_conditions(self, candle, five_min_data):
        if 'wma_fib_0' not in five_min_data.columns or five_min_data['wma_fib_0'].isnull().iloc[-1]:
            self.logger.print_warning("🚨 WMA Fib 0 not available for current candle or is NaN. Skipping entry checks.")
            return

        current_candle_low = candle['low']
        current_candle_close = candle['close']
        
        wma_fib_0_level = five_min_data['wma_fib_0'].iloc[-1]
        threshold_level = wma_fib_0_level * (1 + self.config['trading']['fib_entry_threshold'])
        
        self.logger.print_info(f"DEBUG: TradeManager State: Armed={self.entry_switch_armed}, LastBelowFib={self.last_candle_below_fib}")
        self.logger.print_info(f"DEBUG: Current Candle (L/C): {current_candle_low:.8f}/{current_candle_close:.8f}, WMA Fib 0: {wma_fib_0_level:.8f}, Threshold: {threshold_level:.8f}")

        if not self.entry_switch_armed and current_candle_low <= threshold_level:
            self.entry_switch_armed = True
            self.logger.print_info("🔔 Entry switch ARMED - Price LOW touched threshold.")
        
        price_to_fib_percent = (current_candle_close - wma_fib_0_level) / wma_fib_0_level
        if self.entry_switch_armed and price_to_fib_percent > self.config['trading']['fib_reset_threshold']:
            self.entry_switch_armed = False
            self.logger.print_info(f"🔄 Entry switch reset - Price moved too far ({price_to_fib_percent:.2%}) above WMA Fib 0 after arming.")
            return

        if self.entry_switch_armed and self._entry_conditions_met(candle, five_min_data, wma_fib_0_level):
            success, result = self.position_manager.add_position(current_candle_close, candle.name)
            if success:
                self.logger.print_info(f"🟢 Entry triggered at {current_candle_close:.8f}.")
                self.entry_switch_armed = False
            else:
                self.logger.print_warning(f"❌ Failed to add position: {result}. Entry not completed.")
        else:
            self.logger.print_info(f"DEBUG: Entry conditions not met or not armed for current candle. Armed: {self.entry_switch_armed}")

    def _entry_conditions_met(self, candle, five_min_data, wma_fib_0_level):
        current_candle_close = candle['close']
        
        price_closed_above_fib_0 = current_candle_close > wma_fib_0_level

        previous_candle_was_below_fib_0 = self.last_candle_below_fib

        self.logger.print_info(f"DEBUG: Entry Conds Check: Current Close > WMA Fib 0: {price_closed_above_fib_0}, Prev Candle Below WMA Fib 0: {previous_candle_was_below_fib_0}")

        current_range = (candle['high'] - candle['low']) / candle['low']
        if current_range > self.config['trading']['max_candle_range']:
            self.logger.print_info(f"DEBUG: Candle range ({current_range:.4f}) too high (> {self.config['trading']['max_candle_range']:.4f}). Not entering.")
            return False
        
        if self.position_manager.positions:
            lowest_entry = min(pos['entry_price'] for pos in self.position_manager.positions)
            price_difference_from_lowest = (current_candle_close - lowest_entry) / lowest_entry
            
            required_spacing_pct = self.config['trading']['position_spacing_pct'] 

            if price_difference_from_lowest > required_spacing_pct:
                self.logger.print_info(f"DEBUG: Positions too close ({price_difference_from_lowest:.4f} diff). Not entering. Required diff: {required_spacing_pct:.4f}")
                return False
        
        return price_closed_above_fib_0 and previous_candle_was_below_fib_0

    def _check_exit_conditions(self, current_price, current_time, five_min_data):
        """Check if any positions should be closed"""
        if not self.position_manager.positions:
            return

        # Check cumulative profit target first (highest priority exit)
        metrics = self.position_manager.get_position_metrics(current_price)

        should_trigger_cumulative_exit = False
        if metrics['cumulative_profit'] >= self.config['trading']['cumulative_profit_target']:
            if self.config['trading']['require_fib50_cross']:
                # If require_fib50_cross is true, only exit if price is NOT above Fib 50
                # (meaning it has crossed below or is at Fib 50)
                if not self._is_price_above_fib50_for_holding(five_min_data):
                    should_trigger_cumulative_exit = True
                    self.logger.print_info(f"📈 Cumulative profit target met ({metrics['cumulative_profit']:.2%}) AND price is at/below WMA Fib 50. Closing all positions.")
                else:
                    self.logger.print_info(f"DEBUG: Cumulative profit target met, but price is ABOVE WMA Fib 50. Holding all positions due to require_fib50_cross.")
            else:
                # If require_fib50_cross is FALSE, exit purely on cumulative target
                should_trigger_cumulative_exit = True
                self.logger.print_info(f"📈 Cumulative profit target met ({metrics['cumulative_profit']:.2%}). Closing all positions.")

        if should_trigger_cumulative_exit:
            self.position_manager.close_all_positions(current_price, current_time, 'cumulative_target')
            return # Exit after closing all positions

        # Check individual positions for their specific targets
        for position in self.position_manager.positions[:]:
            if self._should_close_position(position, current_price, current_time, five_min_data):
                exit_type = self._determine_exit_type(position, five_min_data)
                self.logger.print_info(f"📉 Closing position {position['id']} via {exit_type} at {current_price:.8f}.")
                self.position_manager.close_position(position, current_price, current_time, exit_type)

    def _should_close_position(self, position, current_price, current_time, five_min_data):
        """
        Determine if a single position should be closed.
        This version adds a special "strong trend" check for profit_target_2.
        """
        # --- 1. Calculate Time and Determine the Active Profit Target ---
        entry_time = pd.to_datetime(position['entry_time'], utc=True)
        candles_elapsed_df = five_min_data[five_min_data.index > entry_time]
        candles_elapsed = len(candles_elapsed_df)
        time_remaining = max(0, self.config['trading']['countdown_periods'] - candles_elapsed)

        current_profit_pct = (current_price - position['entry_price']) / position['entry_price']

        current_target = (self.config['trading']['profit_target_1']
                          if time_remaining > 0
                          else self.config['trading']['profit_target_2'])

        self.logger.print_info(f"DEBUG: Pos {position['id']} - Profit: {current_profit_pct:.2%}, Target: {current_target:.2%}, T1 Time Left: {time_remaining}")

        # --- 2. Primary Gate: Is the Active Profit Target Met? ---
        if current_profit_pct < current_target:
            return False # If not, we don't need to check anything else.

        # --- 3. NEW: Special Check for Target 2 "Strong Trend" ---
        # This logic runs ONLY if the position is old enough to be on profit_target_2.
        if time_remaining == 0:
            self.logger.print_info(f"DEBUG: Pos {position['id']} on Target 2. Checking for strong trend hold...")
            wma_fib_0 = five_min_data['wma_fib_0'].iloc[-1]

            # Check if WMA Fib 0 is available and above the entry price
            if not pd.isna(wma_fib_0) and wma_fib_0 > position['entry_price']:
                self.logger.print_info(f"📈 DEBUG: STRONG TREND DETECTED for Pos {position['id']}.")
                self.logger.print_info(f"       WMA Fib 0 ({wma_fib_0:.8f}) > Entry ({position['entry_price']:.8f})")
                
                # If the strong trend is active, HOLD as long as price is above WMA Fib 0.
                if current_price > wma_fib_0:
                    self.logger.print_info(f"       HOLDING, as Price ({current_price:.8f}) > WMA Fib 0.")
                    return False # HOLD the position to ride the trend
                else:
                    self.logger.print_info(f"       CLOSING, as Price ({current_price:.8f}) crossed below WMA Fib 0.")
                    return True # SELL, price fell below the new dynamic support

        # --- 4. Default Behavior (for Target 1, or if Target 2's strong trend check isn't met) ---
        # This is the "hold above Fib 50" logic. It's the fallback for all other cases.
        if not self.config['trading']['require_fib50_cross']:
            self.logger.print_info(f"DEBUG: Profit target met. Closing (require_fib50_cross is False).")
            return True

        if self._is_price_above_fib50_for_holding(five_min_data):
            self.logger.print_info(f"DEBUG: Profit target met, but holding as price is ABOVE WMA Fib 50.")
            return False # HOLD the position
        else:
            self.logger.print_info(f"DEBUG: Profit target met and price is at/below WMA Fib 50. CLOSING.")
            return True # CLOSE the position
            
    def _is_price_above_fib50_for_holding(self, five_min_data):
        """
        Helper: Checks if the current candle's close price is strictly ABOVE WMA Fib 50.
        Used to determine if a trade should be HELD despite hitting PT1.
        If WMA Fib 50 is not available, it defaults to holding (returning True) to prevent premature close.
        """
        if 'wma_fib_50' not in five_min_data.columns or five_min_data['wma_fib_50'].isnull().iloc[-1]:
            self.logger.print_warning("🚨 WMA Fib 50 not available for current candle. Defaulting to HOLDING position (returning True) to prevent premature close.")
            return True # If Fib 50 is not available, assume it's still above or unconfirmed, so continue holding.

        current_candle_close = five_min_data['close'].iloc[-1]
        wma_fib50_level = five_min_data['wma_fib_50'].iloc[-1]
        
        is_above = current_candle_close > wma_fib50_level
        self.logger.print_info(f"DEBUG: _is_price_above_fib50_for_holding check: Current Close ({current_candle_close:.8f}) > WMA Fib 50 ({wma_fib50_level:.8f}) = {is_above}")
        return is_above


    def _check_fib50_condition(self, five_min_data):
        """
        Original _check_fib50_condition logic.
        This now specifically checks if the *previous candle closed below* WMA Fib 50.
        It is used for cumulative profit target exit confirmation.
        """
        if 'wma_fib_50' not in five_min_data.columns or five_min_data['wma_fib_50'].isnull().iloc[-1]:
            self.logger.print_warning("🚨 WMA Fib 50 not available for current candle or is NaN. Cannot check Fib 50 cross condition.")
            return False 
            
        if len(five_min_data) < 2:
            self.logger.print_info("DEBUG: Not enough data for Fib 50 cross check (less than 2 candles).")
            return False 

        last_completed_close = five_min_data['close'].iloc[-2]
        wma_fib50_level_prev = five_min_data['wma_fib_50'].iloc[-2]

        if pd.isna(wma_fib50_level_prev):
             self.logger.print_info("DEBUG: Previous WMA Fib 50 is NaN. Cannot confirm previous cross below.")
             return False

        condition_met = last_completed_close < wma_fib50_level_prev
        self.logger.print_info(f"DEBUG: _check_fib50_condition (Prev Close < Prev WMA Fib 50): Last Close ({last_completed_close:.8f}) < Prev WMA Fib 50 ({wma_fib50_level_prev:.8f}) = {condition_met}")
        return condition_met

    def _determine_exit_type(self, position, five_min_data):
        entry_time = pd.to_datetime(position['entry_time'], utc=True)
        candles_elapsed_df = five_min_data[five_min_data.index > entry_time]
        candles_elapsed = len(candles_elapsed_df)
        time_remaining = max(0, self.config['trading']['countdown_periods'] - candles_elapsed)
        
        return 'target_1' if time_remaining > 0 else 'target_2'
