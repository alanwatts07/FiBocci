import requests
import csv
import time
from datetime import datetime
from colorama import Fore, Style, init

# Initialize Colorama for cross-platform ANSI support
init(autoreset=True)

# --- Configuration ---
PAIR = "solana/usd-coin"  # CoinGecko ID for SOL vs USDC
API_URL = f"https://api.coingecko.com/api/v3/simple/price?ids={PAIR.split('/')[0]}&vs_currencies={PAIR.split('/')[1]}"
CSV_FILENAME = "sol_usdc_price_history.csv"
INTERVAL_SECONDS = 60 # 1 minute

# --- ANSI Text Colors ---
GREEN = Fore.GREEN
RED = Fore.RED
YELLOW = Fore.YELLOW
CYAN = Fore.CYAN
MAGENTA = Fore.MAGENTA
BLUE = Fore.BLUE
RESET = Style.RESET_ALL
BRIGHT = Style.BRIGHT
DIM = Style.DIM

def get_sol_usdc_price():
    """Fetches the current SOL/USDC price from CoinGecko."""
    try:
        response = requests.get(API_URL)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        sol_id = PAIR.split('/')[0]
        usdc_id = PAIR.split('/')[1]
        price = data.get(sol_id, {}).get(usdc_id)
        return float(price) if price else None
    except requests.exceptions.RequestException as e:
        print(f"{RED}Error fetching price: {e}{RESET}")
        return None

def append_to_csv(timestamp, price):
    """Appends a new price entry to the CSV file."""
    with open(CSV_FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is empty (first run)
        if f.tell() == 0:
            writer.writerow(["Timestamp", "Price"])
        writer.writerow([timestamp, price])

def display_price(timestamp, price):
    """Displays the price with ANSI text."""
    price_str = f"{price:.4f}" if price is not None else "N/A"

    # Simple color logic based on the last known price (can be enhanced)
    # For a true "up/down" indicator, you'd need to store the previous price

    display_color = GREEN if price is not None and len(price_history) > 1 and price > price_history[-2] else \
                    RED if price is not None and len(price_history) > 1 and price < price_history[-2] else \
                    YELLOW

    print(f"\n{CYAN}{BRIGHT}╔═════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BRIGHT}║    Cyril Wexler Price Tracker       ║{RESET}")
    print(f"{CYAN}{BRIGHT}╠═════════════════════════════════════╣{RESET}")
    print(f"{CYAN}{BRIGHT}║ {MAGENTA}Pair:{RESET} {BRIGHT}{YELLOW}SOL/USDC               {RESET}║")
    print(f"{CYAN}{BRIGHT}║ {MAGENTA}Time:{RESET} {BRIGHT}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET} ║")
    print(f"{CYAN}{BRIGHT}║ {MAGENTA}Price:{RESET} {BRIGHT}{display_color}{price_str:<26}{RESET}║") # Padded for alignment
    print(f"{CYAN}{BRIGHT}╚═════════════════════════════════════╝{RESET}")
    print(f"{DIM}Next update in {INTERVAL_SECONDS} seconds...{RESET}")

# --- Main Loop ---
if __name__ == "__main__":
    print(f"{BRIGHT}{BLUE}Starting SOL/USDC Price Tracker...{RESET}")
    price_history = [] # To store recent prices for basic up/down indication

    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            price = get_sol_usdc_price()

            if price is not None:
                append_to_csv(current_time, price)
                price_history.append(price)
                # Keep history to a reasonable size
                if len(price_history) > 10: 
                    price_history.pop(0) 

            display_price(current_time, price)
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tracker stopped by user.{RESET}")
    except Exception as e:
        print(f"\n{RED}An unexpected error occurred: {e}{RESET}")
