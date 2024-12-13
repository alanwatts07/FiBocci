import websockets
import asyncio
import json
from modules.logger import BotLogger

class WebsocketHandler:
    def __init__(self, config):
        self.config = config
        self.connected_clients = set()
        self.server = None
        self.logger = BotLogger()

    async def start_server(self):
        """Start websocket server"""
        try:
            self.server = await websockets.serve(
                self.handle_connection,
                self.config['websocket']['host'],
                self.config['websocket']['port']
            )
            self.logger.print_info(f"Websocket server started on {self.config['websocket']['host']}:"
                  f"{self.config['websocket']['port']}")
            await self.server.wait_closed()
        except Exception as e:
            self.logger.print_error(f"Error starting websocket server: {str(e)}")

    async def stop_server(self):
        """Stop the websocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.logger.print_info("Websocket server stopped")

    async def handle_connection(self, websocket, path):
        """Handle new websocket connections"""
        self.connected_clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.remove(websocket)

    async def broadcast_price(self, price_data):
        """Broadcast price updates to all connected clients"""
        if not self.connected_clients:
            return
            
        message = json.dumps(price_data)
        await asyncio.gather(
            *[client.send(message) for client in self.connected_clients],
            return_exceptions=True
        )