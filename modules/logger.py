from rich.console import Console
from rich.panel import Panel
from rich.live import Live # Still import, but won't be used for persistent log panel
from rich.table import Table # Still import, but won't be used for persistent status table in a Live UI
from rich.box import DOUBLE # Still import, but not directly used for Panels
from datetime import datetime
import time
import traceback

class BotLogger:
    def __init__(self):
        self.console = Console()
        self.status_data = {
            'price': None,
            'positions': 0,
            'balance': 0.0,
            'status': 'Initializing'
        }
        # In this logger design, self.log_panel and self.live are NOT used for continuous updates.
        # They were causing the AttributeError because they're part of a different Rich TUI pattern.
        # Removing them as instance variables in __init__ is the correct fix for this logger.
        # self.log_panel = None  <-- REMOVE THIS
        # self.live = None       <-- REMOVE THIS

    def start_ui(self):
        """Initialize the logger (prints initial status/info)"""
        self.print_info("[bold green]Bot Started[/bold green]")
        self.print_status() # This prints the current status panel

    def update_status(self, current_price=None, positions=None, balance=None):
        """Update status data and print a new status panel."""
        try:
            if current_price is not None:
                self.status_data['price'] = current_price
            if positions is not None:
                # Assuming positions is a list of objects, so len() is appropriate
                self.status_data['positions'] = len(positions) 
            if balance is not None:
                self.status_data['balance'] = balance
            self.status_data['status'] = 'Active'

            self.print_status()

        except Exception as e:
            self.print_error(f"Error updating status: {str(e)}")

    def print_warning(self, message):
        """Print warning message as a console panel."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Direct print to console as per your existing print_info/error methods
        self.console.print(Panel(
            f"[white]{timestamp}[/white] [yellow]WARNING[/yellow] {message}",
            border_style="yellow", # Use yellow border for warning
            title="Warning"
        ))
            
    def print_status(self):
        """Print current status as a formatted console panel."""
        try:
            status_text = f"""
[bold cyan]═══ Trading Bot Status ═══[/bold cyan]
Time: {datetime.now().strftime('%H:%M:%S')}
Status: [green]{self.status_data['status']}[/green]
Price: {f"${self.status_data['price']:.8f}" if self.status_data['price'] is not None else '-'}
Positions: {self.status_data['positions']}
Balance: SOL {self.status_data['balance']:.3f}
[bold cyan]═══════════════════════[/bold cyan]
            """
            
            self.console.print(Panel(status_text, border_style="cyan"))

        except Exception as e:
            self.print_error(f"Error printing status: {str(e)}")

    def print_trade(self, trade_type, price, amount):
        """Print trade information as a console panel."""
        try:
            color = "green" if trade_type == "BUY" else "red"
            self.console.print(Panel(
                f"{trade_type} Order\nPrice: {price:.8f}\nAmount: {amount:.3f}",
                border_style=color,
                title=f"New {trade_type} Trade"
            ))
        except Exception as e:
            self.print_error(f"Error printing trade: {str(e)}")

    def print_error(self, error_msg):
        """Print error message as a console panel."""
        try:
            self.console.print(Panel(
                str(error_msg),
                border_style="red",
                title="Error"
            ))
        except Exception as e:
            # Fallback print if even console.print fails (very rare)
            print(f"Critical error in print_error: {str(e)}\nOriginal error: {error_msg}")

    def print_info(self, info_msg):
        """Print info message as a console panel."""
        try:
            self.console.print(Panel(
                str(info_msg),
                border_style="blue",
                title="Info"
            ))
        except Exception as e:
            self.print_error(f"Error printing info: {str(e)}") # Use print_error for self-logging

    def stop(self):
        """Stop the logger (prints final status/info)"""
        try:
            self.status_data['status'] = 'Stopped'
            self.print_status()
            self.print_info("[bold red]Bot Stopped[/bold red]")
        except Exception as e:
            self.print_error(f"Error stopping logger: {str(e)}")