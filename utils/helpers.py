import yaml
from datetime import datetime

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def format_price(price, decimals=8):
    """Format price with specified decimal places"""
    return f"{price:.{decimals}f}"

def get_current_timestamp():
    """Get current timestamp in UTC"""
    return datetime.utcnow()

def calculate_profit_percentage(entry_price, current_price):
    """Calculate profit percentage"""
    return (current_price - entry_price) / entry_price

def format_timestamp(timestamp):
    """Format timestamp for display"""
    return timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')