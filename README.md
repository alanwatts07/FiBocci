# SOL/USDC Live Trading Bot

Welcome to the **SOL/USDC Live Trading Bot** project! This bot is designed to automate trading of the SOL/USDC pair (or any pair) using a sophisticated Fibonacci-based strategy. It includes robust features such as live charting with real-time Weighted Moving Average (WMA) Fibonacci levels, dynamic trade management, and comprehensive performance statistics reported in USDC.

![Live Trading Chart](Screenshot1.png)

## Quick Start Guide: Fibonacci Bot (Minimalist)

This guide provides the essential steps to get your price tracker and trading bot running.

### What You'll Need

* Your `sol_usd_tracker.py` script.
* Your `main.py` script (with the `start` command).
* All required Python packages installed in your virtual environment.
* (You can do a `pip install -r requirements.txt` once you activate the venv)
* Your virtual environment activated.

### Step 1: Start the Price Tracker

Open your terminal, navigate to your project directory, activate your virtual environment, and run the price tracker in the background. This populates `sol_usdc_price_history_jupiter.csv`:

```bash
# Optional: Clean old price data for a fresh start
rm sol_usdc_price_history_jupiter.csv

# Start the price tracker in the background
python3 sol_usd_tracker.py &
````

  * **Wait \~15-20 minutes** to allow the tracker to generate enough initial 5-minute candle data (at least 65 candles are needed for full WMA Fibonacci calculations).

### Step 2: Start the Trading Bot & Live Chart

Open a **new terminal tab** (or window), navigate to your project directory, activate your virtual environment, and run the bot. This will launch the trading logic and the web-based live chart:

```bash
# Make sure you are in a NEW terminal tab/window
python3 main.py start
```

  * Your live chart will open in your web browser, typically at `http://localhost:8051`.

### How to Stop

1.  **Stop the Trading Bot:** Go to the terminal tab running `main.py start` and press `Ctrl + C`. The bot will attempt a graceful shutdown and save trade history.
2.  **Stop the Price Tracker:**
      * Find its process ID: `ps aux | grep "sol_usd_tracker.py"`
      * Terminate it: `kill <PID>` (replace `<PID>` with the ID you found).
      * *Alternatively (if all else fails):* Close your WSL terminal or run `wsl --shutdown` from Windows PowerShell/CMD to stop all WSL processes.

-----

## Table of Contents

  - [Features](https://www.google.com/search?q=%23features)
  - [Getting Started](https://www.google.com/search?q=%23getting-started)
      - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      - [Installation](https://www.google.com/search?q=%23installation)
  - [Configuration](https://www.google.com/search?q=%23configuration)
  - [Usage](https://www.google.com/search?q=%23usage)
      - [Running the Bot](https://www.google.com/search?q=%23running-the-bot)
      - [Simulating Live Data](https://www.google.com/search?q=%23simulating-live-data)
  - [Modules Overview](https://www.google.com/search?q=%23modules-overview)
      - [Main Components](https://www.google.com/search?q=%23main-components)
      - [Live Chart](https://www.google.com/search?q=%23live-chart)
  - [Trading Strategy](https://www.google.com/search?q=%23trading-strategy)
  - [FAQ](https://www.google.com/search?q=%23faq)
  - [Contributing](https://www.google.com/search?q=%23contributing)
  - [License](https://www.google.com/search?q=%23license)
  - [Acknowledgments](https://www.google.com/search?q=%23acknowledgments)
  - [Contact](https://www.google.com/search?q=%23contact)

-----

## Features

This bot is designed to maximize capital (USDC) by strategically accumulating more Solana (SOL) during market dips, while managing risk.

  * **Automated Trading Strategy**: Implements a refined Fibonacci-based strategy tailored for **SOL/USDC** trading.
  * **Live Charting (Web UI)**:
      * Real-time candlestick chart.
      * **Dynamic Weighted Moving Average (WMA) Fibonacci levels** plotted with customizable colors and dash styles.
      * **Real-time price line**, **open position lines** (with live profit/loss percentages), and **closed trade markers** (entry, exit, profit/loss path).
      * Displays real-time trade statistics and bot status (e.g., "Entry Switch Armed").
  * **Intelligent Trade Management**:
      * **USDC-centric Trading**: Manages capital and calculates profits primarily in USDC.
      * **Position Sizing**: Configurable trade size (in USDC) and maximum number of concurrent open positions.
      * **Dip Accumulation**: Implements a **position spacing** feature, requiring new concurrent trades to be a configurable percentage below the lowest existing entry, facilitating dollar-cost averaging on dips.
      * **Automated Entry**: Based on price dipping below a Fibonacci threshold and then re-crossing the WMA Fib 0 level (buy signal).
      * **Automated Exit**:
          * **Tiered Profit Targets**: Uses `profit_target_1` (higher, short-term) and `profit_target_2` (lower, time-based fallback).
          * **Fib 50 Holding Logic**: Can hold trades that meet `profit_target_1` if the price remains above WMA Fib 50, only closing when it drops back below, to maximize gains.
          * **Cumulative Profit Target**: Closes all positions if an overall portfolio profit target (in percentage of initial capital) is reached.
  * **Configurable**: All key trading parameters, chart styles (colors, dash types, widths), and paths are easily adjustable via `config/config.yaml`.
  * **Robust Logging**: Provides detailed console logs for tracking bot actions, calculations, and debugging.
  * **Historical Data Processing**: Initializes with existing market data from CSV for faster startup and indicator calculation.
  * **Graceful Shutdown**: Handles `Ctrl+C` for clean exits and trade history saving.

-----

## Getting Started

### Prerequisites

  * Python 3.7 or higher
  * [Pipenv](https://pipenv.pypa.io/en/latest/) or `pip` for managing Python packages
  * Git (optional, for cloning the repository)

### Installation

1.  **Clone the Repository**:

    ```bash
    git clone [https://github.com/yourusername/wif-sol-trading-bot.git](https://github.com/yourusername/wif-sol-trading-bot.git)
    cd wif-sol-trading-bot
    ```

2.  **Set Up a Virtual Environment**:

    Using Pipenv:

    ```bash
    pipenv install
    pipenv shell
    ```

    Or using `pip` and `venv`:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**:

    The required packages are listed in `requirements.txt`. Install them with:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Files**:

      * Ensure your `sol_usd_tracker.py` is correctly generating `sol_usdc_price_history_jupiter.csv` with real-time (or simulated) 1-minute price data in `price_data/` (or your configured path). The bot reads this file.

-----

## Configuration

The bot is configured using the `config/config.yaml` file. This file contains critical settings for:

  * **Trading Parameters**:
      * `initial_balance`: Your starting capital, **implicitly in USDC**.
      * `trade_size`: The amount of **USDC** to spend per individual trade.
      * `max_positions`: Maximum number of open SOL positions concurrently.
      * `fib_entry_threshold`: How far below WMA Fib 0 the price must dip to arm.
      * `fib_reset_threshold`: How far above WMA Fib 0 the price must go to disarm.
      * `profit_target_1`: Higher profit percentage for initial exit attempts (e.g., `0.01` for 1%).
      * `profit_target_2`: Lower profit percentage if `countdown_periods` are exceeded.
      * `cumulative_profit_target`: Overall portfolio profit percentage target to close all trades.
      * `countdown_periods`: Number of 5-minute candles before switching from `profit_target_1` to `profit_target_2`.
      * `require_fib50_cross`: Boolean to enable/disable specific Fib 50 conditions for exits.
      * `position_spacing_pct`: Minimum percentage below the lowest open position required for a new entry (e.g., `-0.03` for 3% dip).
  * **Chart Styles**: Customizable `color`, `dash` type (`"solid"`, `"dot"`, `"dash"`, etc.), and `width` for all plotted lines (Fibonacci levels, current price, open positions, trade paths).
  * **Paths**: Location of live data CSV, and where to save trade history.
  * **Logging**: Logging levels and file output.

**Example `config.yaml` (Excerpts):**

```yaml
trading:
  initial_balance: 1000.0          # Starting capital in USDC
  trade_size: 100.0               # Amount of USDC to spend per trade
  max_positions: 4
  profit_target_1: 0.01           # 1% profit target
  profit_target_2: 0.007          # 0.7% fallback profit target
  cumulative_profit_target: 0.035 # 3.5% overall portfolio profit target
  fib_entry_threshold: -0.003
  fib_reset_threshold: 0.035
  max_candle_range: 0.016
  countdown_periods: 10           # Max 10 5-min candles to hit profit_target_1
  require_fib50_cross: true
  position_spacing_pct: -0.03     # Require 3% dip for new concurrent entry

paths:
  live_data: sol_usdc_price_history_jupiter.csv
  trade_history: trade_history.csv

chart_styles:
  fib_0_line:
    color: "#FFFFFF" # Solid White
    dash: "solid"
    width: 1
  fib_50_line:
    color: "#FFFF00" # Dotted Yellow
    dash: "dot"
    width: 1
  entry_threshold_line:
    color: "#FF0000" # Solid Red
    dash: "solid"
    width: 1
  # ... other line styles
```

-----

## Usage

### Running the Bot

To start the trading bot, follow the [Quick Start Guide](https://www.google.com/search?q=%23quick-start-guide-fibonacci-bot-minimalist).

**Optional Arguments for `python3 main.py start`**:

  * `--config` or `-c`: Specify a custom configuration file.

    ```bash
    python3 main.py start --config config/my_custom_config.yaml
    ```

  * `--interval` or `-i`: Set the check interval (in seconds) for new live data in the CSV.

    ```bash
    python3 main.py start --interval 5 # Check CSV file for updates every 5 seconds
    ```

### Simulating Live Data

If you want to simulate live data (e.g., for testing without a live price tracker), you can use a pre-populated CSV file and introduce delays in `main.py`'s `process_new_data` loop. This allows you to observe the bot's behavior and live charts in a controlled manner.

**Example**: (Modify `main.py` accordingly for testing purposes only)

```python
# In main.py, inside async def process_new_data(self, csv_path):
    # ... existing code ...
    for idx, row in new_data_df.iterrows():
        # ... process data point ...
        await asyncio.sleep(1)  # Pause for 1 second between processing 1-minute price points
```

-----

## Modules Overview

### Main Components

  * **`main.py`**: The bot's core entry point. Manages initialization of all components, orchestrates the market monitoring loop, and handles graceful shutdown.
  * **`data_handler.py`**: Processes raw 1-minute price data from the CSV, aggregates it into 5-minute candlestick data, and maintains the historical data for indicators.
  * **`indicators.py`**: Contains the `FibonacciCalculator` class, responsible for calculating raw Fibonacci levels and their Weighted Moving Averages (WMAs).
  * **`trade_manager.py`**: Houses the `TradeManager` class, which evaluates the sophisticated entry and exit conditions based on market data and indicators. It decides *when* to execute a trade.
  * **`position_manager.py`**: Manages all open positions, maintains the trade history, calculates individual and overall portfolio profits (in USDC), and tracks the bot's USDC balance and SOL holdings.
  * **`live_chart.py`**: A Dash application that provides a real-time, interactive web interface for visualizing market data, Fibonacci levels, trade actions, and performance statistics.
  * **`logger.py`**: Provides structured console logging for bot events, status updates, and debugging information.
  * **`backtest_manager.py`**: (Currently commented out) Manages the backtesting process on historical data.

### Live Chart

The live chart is a Dash application that displays:

  * **Candlestick Chart**: Real-time price movements with completed and current forming candles.
  * **WMA Fibonacci Levels**: Plots the `WMA Fib 0`, `WMA Fib 50`, and `Entry Threshold` lines dynamically, with customizable appearance.
  * **Trade Markers**: Visual indicators for trade entries, exits, and their profit/loss paths.
  * **Trade Statistics**: Displays comprehensive metrics such as Total Trades, Win Rate, Average Profit per Trade, Best/Worst Trade, **Total USDC Profit**, and **Overall Capital Gain Percentage**.
  * **Bot Status**: Shows whether the "Entry Switch" is armed.

**Accessing the Live Chart**:

By default, the live chart runs on `http://localhost:8051`. Open this URL in your web browser to view the chart and statistics.

-----

## Trading Strategy

The bot employs a dynamic Fibonacci-based strategy with precise entry, exit, and risk management rules:

  * **Entry Conditions**:

    1.  The bot's "Entry Switch" is **armed** when the current candle's `low` price touches or drops below a configurable `fib_entry_threshold` relative to the `WMA Fib 0` level. This identifies significant pullbacks.
    2.  An actual **entry** occurs (i.e., a buy order is placed) only if the switch is armed **AND** the *current candle's close price* re-crosses **above** the `WMA Fib 0` level.
    3.  Crucially, the *previous candle* must have closed **below** the `WMA Fib 0` level to confirm a true "re-cross from below" reversal.
    4.  The current candle's range must be within `max_candle_range` to avoid volatile entries.
    5.  **Position Spacing**: If other positions are open, the new entry must be at least `position_spacing_pct` (e.g., 3%) **lower** than the lowest existing entry price, promoting dollar-cost averaging.
    6.  The bot must have sufficient USDC balance and not exceed `max_positions`.

  * **Exit Conditions**:

    1.  **Profit Target 1**: If the trade hits a specified `profit_target_1` (e.g., 1%) within `countdown_periods` (e.g., 10 5-min candles):
          * If `require_fib50_cross` is `true` **AND** the current price is *above* the `WMA Fib 50` level, the bot **holds the trade**, aiming for further gains.
          * If the price is *not* above `WMA Fib 50` (i.e., at or below), the trade is closed for `profit_target_1`.
    2.  **Profit Target 2**: If `countdown_periods` expire and `profit_target_1` hasn't been met (or wasn't closed due to Fib 50 hold), the bot attempts to close the trade at a lower `profit_target_2` (e.g., 0.7%).
    3.  **Cumulative Profit Target**: If the overall portfolio's cumulative profit (realized + unrealized) as a percentage of initial capital (`cumulative_profit_target`) is reached, all open positions are closed, provided the `_check_fib50_condition` (previous candle closed below WMA Fib 50) is met. This ensures a strategic full portfolio exit.
    4.  *(Self-correction):* The "Entry Switch" will automatically disarm if the price moves too far above `WMA Fib 0` after being armed, preventing stale signals.

  * **Risk Management**:

      * Strict limits on `max_positions` control total exposure.
      * Trade sizes are fixed in **USDC**, ensuring predictable capital deployment.
      * Balance tracking prevents overspending.
      * The position spacing strategy inherently reduces risk by averaging down.

-----

## FAQ

**Q**: *Can I use this bot for other trading pairs?*

**A**: Yes, but you'll need to adjust the `sol_usd_tracker.py` script, `config.yaml` paths, and verify the strategy parameters are suitable for the new pair's volatility.

**Q**: *Is this bot ready for live trading with real money?*

**A**: This bot is provided for **educational and simulated trading purposes**. While robust features are implemented, real-money trading involves significant risk. **Thorough testing (backtesting, paper trading), professional financial advice, and understanding of exchange APIs/fees are essential** before deploying with real funds.

**Q**: *How do I add more indicators or modify the strategy?*

**A**: You can extend the `indicators.py` and `trade_manager.py` modules to incorporate additional indicators and adjust the trading logic. The `config.yaml` provides a centralized place for parameter tuning.

**Q**: *The live chart doesn't display correctly. What should I do?*

**A**: Ensure all dependencies are installed, the `python3 sol_usd_tracker.py` is running in a separate terminal and generating data, and `main.py start` is running. Check both terminals for any error messages. Verify the chart port (`http://localhost:8051`) is accessible.

-----

## Contributing

Contributions are welcome\! If you'd like to improve the bot, add new features, or fix bugs, please follow these steps:

1.  **Fork the Repository**:
    Click the "Fork" button at the top right of this page to create a copy of the repository on your account.

2.  **Clone Your Fork**:

    ```bash
    git clone [https://github.com/yourusername/wif-sol-trading-bot.git](https://github.com/yourusername/wif-sol-trading-bot.git)
    ```

3.  **Create a New Branch**:

    ```bash
    git checkout -b my-feature
    ```

4.  **Make Your Changes**:
    Add new features or fix issues as needed.

5.  **Commit Your Changes**:

    ```bash
    git commit -am 'Add new feature'
    ```

6.  **Push to Your Fork**:

    ```bash
    git push origin my-feature
    ```

7.  **Submit a Pull Request**:
    Go back to the original repository and open a pull request with a description of your changes.

-----

## License

This project is licensed under the **MIT License** - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

-----

## Acknowledgments

  * **Libraries Used**:
      * [Pandas](https://pandas.pydata.org/): Data manipulation and analysis.
      * [NumPy](https://numpy.org/): Numerical computing.
      * [Plotly Dash](https://plotly.com/dash/): Interactive web applications and live charting.
      * [Rich](https://rich.readthedocs.io/en/stable/): For beautiful console logging.
      * [PyYAML](https://pyyaml.org/): YAML parsing and configuration handling.
      * [Asyncio](https://docs.python.org/3/library/asyncio.html): Asynchronous I/O.
      * [Typer](https://typer.tiangolo.com/): Building the command-line interface.
      * [pytz](https://www.google.com/search?q=https://python.pypi.org/pypi/pytz): Timezone handling.

-----

## Contact

For any questions or support, please open an issue on the repository or contact the maintainer.

-----

*Happy Trading\!* 🚀