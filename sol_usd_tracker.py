import requests
import csv
import time
from datetime import datetime
from colorama import Fore, Style, init

# Initialize Colorama for cross-platform ANSI support
init(autoreset=True)

# --- Configuration ---
# Jupiter API Endpoint (use lite-api for free tier, api.jup.ag with API key for higher limits)
JUPITER_API_BASE_URL = "https://lite-api.jup.ag/" 
PRICE_API_PATH = "price/v3"

# Token Mint Addresses (important for Jupiter API)
SOL_MINT_ADDRESS = "So11111111111111111111111111111111111111112"
USDC_MINT_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

CSV_FILENAME = "sol_usdc_price_history_jupiter.csv"
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

def get_sol_usdc_price_jupiter():
    """Fetches the current SOL/USDC price from Jupiter Exchange Price API V3."""
    url = f"{JUPITER_API_BASE_URL}{PRICE_API_PATH}"
    params = {
        "ids": SOL_MINT_ADDRESS,
        # The documentation implies 'vsToken' might not be strictly necessary if you're getting USD price,
        # but including it doesn't hurt and clarifies intent.
        # It's also possible to query multiple IDs like:
        # "ids": f"{SOL_MINT_ADDRESS},{JUP_MINT_ADDRESS}"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        
        data = response.json()
        
        # --- REMOVE OR COMMENT OUT THE TEMPORARY DEBUGGING PRINTS AFTER THIS FIX ---
        # print(f"\n{CYAN}--- Jupiter API Parsed JSON Data: ---{RESET}")
        # import json
        # print(json.dumps(data, indent=2))
        # print(f"{CYAN}-------------------------------------{RESET}")
        
        # CORRECTED: Access the 'usdPrice' directly from the token's mint address key
        sol_price_data = data.get(SOL_MINT_ADDRESS) # No "data" key at the top level
        
        if sol_price_data and "usdPrice" in sol_price_data: # Look for "usdPrice"
            price = float(sol_price_data["usdPrice"])
            return price
        else:
            print(f"{YELLOW}Warning: Price data for SOL/USDC not found in response.{RESET}")
            # print(f"{YELLOW}Keys found in top-level object: {data.keys()}{RESET}") # for further debug if needed
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"{RED}Error fetching price from Jupiter API: {e}{RESET}")
        return None
    except json.JSONDecodeError as e:
        print(f"{RED}Error decoding JSON response from Jupiter API: {e}{RESET}")
        # If response.text exists and is not too large, print it for debugging
        if 'response' in locals() and response.text:
            print(f"{RED}Raw response that caused error: {response.text[:500]}...{RESET}") # Print first 500 chars
        return None
    except Exception as e:
        print(f"{RED}An unexpected error occurred during API call: {e}{RESET}")
        return None

def append_to_csv(timestamp, price):
    """Appends a new price entry to the CSV file."""
    # Using 'a' mode creates the file if it doesn't exist
    with open(CSV_FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)
        # Check if file is empty to write header
        if f.tell() == 0:
            writer.writerow(["timestamp", "price"])
        writer.writerow([timestamp, price])

def display_price(timestamp, price, price_history):
    """Displays the price with ANSI text."""
    price_str = f"{price:.4f}" if price is not None else "N/A"
    
    # Simple color logic based on the last known price
    display_color = YELLOW # Default to yellow
    if price is not None and len(price_history) > 1:
        # Compare current price to the previous one in the history list
        if price > price_history[-2]:
            display_color = GREEN
        elif price < price_history[-2]:
            display_color = RED
    
    print(f"\n{CYAN}{BRIGHT}╔═════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BRIGHT}║    Jupiter SOL/USDC Price Tracker   ║{RESET}")
    print(f"{CYAN}{BRIGHT}╠═════════════════════════════════════╣{RESET}")
    print(f"{CYAN}{BRIGHT}║ {MAGENTA}Pair:{RESET} {BRIGHT}{YELLOW}SOL/USDC (Jupiter){RESET}    ║")
    print(f"{CYAN}{BRIGHT}║ {MAGENTA}Time:{RESET} {BRIGHT}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET} ║")
    print(f"{CYAN}{BRIGHT}║ {MAGENTA}Price:{RESET} {BRIGHT}{display_color}{price_str:<26}{RESET}║") # Padded for alignment
    print(f"{CYAN}{BRIGHT}╚═════════════════════════════════════╝{RESET}")
    print(f"{DIM}Next update in {INTERVAL_SECONDS} seconds...{RESET}")

# --- Main Loop ---
if __name__ == "__main__":
    print(f"{BRIGHT}{BLUE}Starting Jupiter SOL/USDC Price Tracker...{RESET}")
    price_history = [] # To store recent prices for basic up/down indication

    try:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            price = get_sol_usdc_price_jupiter()

            if price is not None:
                append_to_csv(current_time, price)
                price_history.append(price)
                # Keep history to a reasonable size to avoid excessive memory usage
                if len(price_history) > 10: 
                    price_history.pop(0) 
            
            display_price(current_time, price, price_history)
            time.sleep(INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tracker stopped by user.{RESET}")
    except Exception as e:
        print(f"\n{RED}An unexpected error occurred: {e}{RESET}")
