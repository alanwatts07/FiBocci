

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.box import DOUBLE
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

    def start_ui(self):
        """Initialize the logger"""
        self.print_info("[bold green]Bot Started[/bold green]")
        self.print_status()

    def update_status(self, current_price=None, positions=None, balance=None):
        """Update status and print"""
        try:
            # Update status data
            if current_price is not None:
                self.status_data['price'] = current_price
            if positions is not None:
                self.status_data['positions'] = len(positions)
            if balance is not None:
                self.status_data['balance'] = balance
            self.status_data['status'] = 'Active'

            # Print status
            self.print_status()

        except Exception as e:
            self.print_error(f"Error updating status: {str(e)}")

    def print_status(self):
        """Print current status"""
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
        """Print trade information"""
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
        """Print error message"""
        try:
            self.console.print(Panel(
                str(error_msg),
                border_style="red",
                title="Error"
            ))
        except Exception as e:
            print(f"Critical error in print_error: {str(e)}")

    def print_info(self, info_msg):
        """Print info message"""
        try:
            self.console.print(Panel(
                str(info_msg),
                border_style="blue",
                title="Info"
            ))
        except Exception as e:
            self.print_error(f"Error printing info: {str(e)}")

    def stop(self):
        """Stop the logger"""
        try:
            self.status_data['status'] = 'Stopped'
            self.print_status()
            self.print_info("[bold red]Bot Stopped[/bold red]")
        except Exception as e:
            print(f"Error stopping logger: {str(e)}")
