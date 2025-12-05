"""
WebSocket Server for Real-time AR Visualization
Enables real-time updates to AR clients with compression and optimization
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, Set, Any, Optional
import gzip
import base64
from datetime import datetime

from .ar_interface import ARInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARWebSocketServer:
    """WebSocket server for AR visualization updates"""
    
    def __init__(self, ar_interface: ARInterface, host: str = "localhost", port: int = 8765):
        self.ar_interface = ar_interface
        self.host = host
        self.port = port
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()
        self.is_running = False
        
        logger.info(f"✅ AR WebSocket Server initialized on {host}:{port}")
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register a new client connection"""
        self.connected_clients.add(websocket)
        client_id = f"client_{len(self.connected_clients)}"
        
        # Register client with AR interface
        self.ar_interface.register_client(client_id, {
            "camera_position": [0, 0, 0],
            "camera_orientation": [0, 0, 0],
            "viewport_size": (1920, 1080)
        })
        
        logger.info(f"🔌 Client connected: {client_id} (Total: {len(self.connected_clients)})")
        return client_id
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol, client_id: str):
        """Unregister a client connection"""
        self.connected_clients.discard(websocket)
        self.ar_interface.unregister_client(client_id)
        logger.info(f"🔌 Client disconnected: {client_id} (Total: {len(self.connected_clients)})")
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connection"""
        client_id = None
        try:
            # Register client
            client_id = await self.register_client(websocket)
            
            # Send initial state
            initial_view = self.ar_interface.get_client_view(client_id, compressed=True)
            await websocket.send(json.dumps(initial_view))
            
            # Handle client messages
            async for message in websocket:
                try:
                    # Parse client message
                    data = json.loads(message)
                    
                    # Handle different message types
                    if data.get("type") == "update_camera":
                        # Update client camera position
                        self.ar_interface.update_client_state(client_id, {
                            "camera_position": data.get("position", [0, 0, 0]),
                            "camera_orientation": data.get("orientation", [0, 0, 0]),
                            "viewport_size": data.get("viewport", [1920, 1080])
                        })
                    elif data.get("type") == "request_update":
                        # Send current view
                        client_view = self.ar_interface.get_client_view(client_id, compressed=True)
                        await websocket.send(json.dumps(client_view))
                        
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON from client {client_id}")
                except Exception as e:
                    logger.error(f"❌ Error handling client message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 Client {client_id} connection closed")
        except Exception as e:
            logger.error(f"❌ Error handling client {client_id}: {e}")
        finally:
            # Unregister client
            if client_id:
                await self.unregister_client(websocket, client_id)
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast updates to all connected clients"""
        if not self.connected_clients:
            return
            
        try:
            # Update AR interface with new data
            self.ar_interface.model.update_realtime_data(data)
            
            # Send updates to all clients
            disconnected_clients = set()
            
            for websocket in self.connected_clients.copy():
                try:
                    # Get optimized view for this client
                    # Extract client_id from the websocket (simplified approach)
                    client_id = f"client_{id(websocket) % 10000}"  # Simplified ID generation
                    
                    client_view = self.ar_interface.get_client_view(client_id, compressed=True)
                    await websocket.send(json.dumps(client_view))
                    
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(websocket)
                except Exception as e:
                    logger.error(f"❌ Error sending update to client: {e}")
            
            # Remove disconnected clients
            for websocket in disconnected_clients:
                self.connected_clients.discard(websocket)
                
        except Exception as e:
            logger.error(f"❌ Error broadcasting update: {e}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"🚀 Starting AR WebSocket Server on {self.host}:{self.port}")
        
        self.is_running = True
        server = await websockets.serve(self.handle_client, self.host, self.port)
        
        try:
            await server.wait_closed()
        except asyncio.CancelledError:
            logger.info("🛑 WebSocket server cancelled")
        finally:
            self.is_running = False
            logger.info("🔴 AR WebSocket Server stopped")
    
    def stop_server(self):
        """Stop the WebSocket server"""
        self.is_running = False
        logger.info("🔴 Stopping AR WebSocket Server")


class WebSocketARInterface(ARInterface):
    """AR Interface with integrated WebSocket support"""
    
    def __init__(self, websocket_server: ARWebSocketServer = None):
        super().__init__()
        self.websocket_server = websocket_server
        logger.info("✅ WebSocket-enabled AR Interface initialized")
    
    async def broadcast_update(self, data: Dict):
        """Broadcast updates to all clients via WebSocket"""
        # Update local model
        self.model.update_realtime_data(data)
        
        # Broadcast via WebSocket if available
        if self.websocket_server and self.websocket_server.is_running:
            await self.websocket_server.broadcast_update(data)
        else:
            # Fallback to original method
            super().broadcast_update(data)


def create_websocket_ar_interface(host: str = "localhost", port: int = 8765) -> WebSocketARInterface:
    """Factory function to create WebSocket-enabled AR interface"""
    # Create WebSocket server
    websocket_server = ARWebSocketServer(ARInterface(), host, port)
    
    # Create WebSocket-enabled AR interface
    ar_interface = WebSocketARInterface(websocket_server)
    
    logger.info("✅ WebSocket AR Interface created")
    return ar_interface


# Example usage and server runner
async def run_websocket_server():
    """Run the WebSocket server"""
    print("🌐 Starting AR WebSocket Server...")
    
    # Create AR interface
    ar_interface = ARInterface()
    
    # Create WebSocket server
    server = ARWebSocketServer(ar_interface, "localhost", 8765)
    
    # Start server
    await server.start_server()


async def simulate_real_time_updates():
    """Simulate real-time updates for testing"""
    print("🔄 Starting simulation of real-time updates...")
    
    # Create AR interface
    ar_interface = ARInterface()
    
    # Register test client
    ar_interface.register_client("test_client", {
        "camera_position": [50, 10, 5],
        "viewport_size": (1920, 1080)
    })
    
    # Simulate updates
    for i in range(100):
        # Generate test data
        test_data = {
            "furnace_temperature": 1500 + (i % 100) * 2,
            "belt_speed": 150 + (i % 50),
            "mold_temperature": 320 + (i % 20),
            "defects": {
                "crack": 0.1 + (i % 10) * 0.01,
                "bubble": 0.05 + (i % 5) * 0.01,
                "chip": 0.02 + (i % 3) * 0.005
            }
        }
        
        # Update AR interface
        ar_interface.model.update_realtime_data(test_data)
        
        # Get client view
        client_view = ar_interface.get_client_view("test_client", compressed=True)
        
        if i % 20 == 0:
            print(f"📊 Update {i}: Payload size = {client_view['performance']['payload_size_kb']:.2f} KB")
        
        # Wait before next update
        await asyncio.sleep(0.5)
    
    print("✅ Simulation completed!")


if __name__ == "__main__":
    print("🧪 Testing WebSocket AR Server...")
    print("=" * 50)
    
    # Run simulation
    asyncio.run(simulate_real_time_updates())
    
    print("\n✅ WebSocket AR Server test completed!")
    print("\nTo run the actual WebSocket server, use:")
    print("  asyncio.run(run_websocket_server())")