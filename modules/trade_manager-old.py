from modules.logger import BotLogger

class TradeManager:
    def __init__(self, config, position_manager, fib_calculator):
        self.config = config
        self.position_manager = position_manager
        self.fib_calculator = fib_calculator
        self.entry_switch_armed = False
        self.last_candle_below_fib = True
        self.logger = BotLogger()

    def check_for_trades(self, candle, five_min_data):
        """Check for potential trades based on current market conditions"""
        if len(five_min_data) < 24:  # Need enough data for indicators
            return

        current_price = candle['close']
        current_time = candle.name

        # First check exit conditions for existing positions
        self._check_exit_conditions(current_price, current_time, five_min_data)

        # Then check entry conditions
        if len(self.position_manager.positions) < self.config['trading']['max_positions']:
            self._check_entry_conditions(candle, five_min_data)

    def _check_entry_conditions(self, candle, five_min_data):
        """Check if new position should be opened"""
        if 'wma_fib_0' not in five_min_data.columns:
            return

        current_price = candle['low']
        fib_level = five_min_data['wma_fib_0'].iloc[-1]
        threshold_level = fib_level * (1 + self.config['trading']['fib_entry_threshold'])
        
        price_to_fib_percent = (current_price - fib_level) / fib_level

        # Log price and threshold information
        self.logger.print_info(f"Current Price: {current_price:.8f}, Threshold: {threshold_level:.8f}, Price Below threshold: {current_price <= threshold_level}")

        # Check if price touches threshold
        if not self.entry_switch_armed and current_price <= threshold_level:
            self.entry_switch_armed = True
            self.logger.print_info("🔔 Entry switch ARMED - Price touched threshold")
        else:
            self.logger.print_info("Entry switch NOT ARMED")

        # Reset entry switch if price moves too far above fib
        if self.entry_switch_armed and price_to_fib_percent > 0.03:
            self.entry_switch_armed = False
            self.logger.print_info("🔄 Entry switch reset - Price moved too far above fib")
            return

        # Check entry conditions
        if self.entry_switch_armed and self._entry_conditions_met(candle, five_min_data):
            success, result = self.position_manager.add_position(current_price, candle.name)
            if success:
                self.logger.print_info(f"🟢 Entry triggered at {current_price:.8f}")
                self.entry_switch_armed = False

    def _entry_conditions_met(self, candle, five_min_data):
        """Check if all entry conditions are met"""
        current_price = candle['low']
        fib_level = five_min_data['wma_fib_0'].iloc[-1]
        
        # Basic price conditions
        price_above_fib = current_price > fib_level
        previous_below_fib = self.last_candle_below_fib

        # Log basic conditions
        self.logger.print_info(f"Price above fib: {price_above_fib}, Previous below fib: {previous_below_fib}")

        # Candle range check
        current_range = (candle['high'] - candle['low']) / candle['low']
        if current_range > self.config['trading']['max_candle_range']:
            self.logger.print_info("Candle range too high, not entering.")
            return False
        
        # Position spacing check
        if self.position_manager.positions:
            lowest_entry = min(pos['entry_price'] for pos in self.position_manager.positions)
            price_difference = (current_price - lowest_entry) / lowest_entry
            if price_difference > -0.03:  # Require 3% spacing between positions
                self.logger.print_info("Positions too close, not entering.")
                return False
        
        return price_above_fib and previous_below_fib

  
    
    def _check_exit_conditions(self, current_price, current_time, five_min_data):
        """Check if any positions should be closed"""
        if not self.position_manager.positions:
            return

        # Check cumulative profit target
        metrics = self.position_manager.get_position_metrics(current_price)
        if (metrics['cumulative_profit'] >= self.config['trading']['cumulative_profit_target'] and
            self._check_fib50_condition(five_min_data)):
            self.position_manager.close_all_positions(current_price, current_time, 'cumulative_target')
            return

        # Check individual positions
        for position in self.position_manager.positions[:]:
            if self._should_close_position(position, current_price, current_time, five_min_data):
                self.position_manager.close_position(position, current_price, current_time, 
                self._determine_exit_type(position, five_min_data))


    def _should_close_position(self, position, current_price, current_time, five_min_data):
        """Determine if a position should be closed"""
        entry_time = position['entry_time']
        candles_elapsed = len(five_min_data[five_min_data.index > entry_time])
        time_remaining = max(0, self.config['trading']['countdown_periods'] - candles_elapsed)
        
        current_profit = (current_price - position['entry_price']) / position['entry_price']
        current_target = (self.config['trading']['profit_target_1'] 
                         if time_remaining > 0 
                         else self.config['trading']['profit_target_2'])
        
        return current_profit >= current_target and self._check_fib50_condition(five_min_data)

    def _check_fib50_condition(self, five_min_data):
        """Check if price is below fib 50 level"""
        if not self.config['trading']['require_fib50_cross']:
            return True
            
        if 'wma_fib_50' not in five_min_data.columns:
            return True
            
        last_close = five_min_data['close'].iloc[-2]
        fib50_level = five_min_data['wma_fib_50'].iloc[-1]
        
        return last_close < fib50_level

    def _determine_exit_type(self, position, five_min_data):
        """Determine the type of exit for a position"""
        entry_time = position['entry_time']
        candles_elapsed = len(five_min_data[five_min_data.index > entry_time])
        time_remaining = max(0, self.config['trading']['countdown_periods'] - candles_elapsed)
        
        return 'target_1' if time_remaining > 0 else 'target_2'