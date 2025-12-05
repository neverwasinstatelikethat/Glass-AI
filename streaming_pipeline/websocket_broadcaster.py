"""
WebSocket Broadcaster for Real-Time Dashboard Updates
Broadcasts sensor data, predictions, alerts, and recommendations to frontend clients
"""

import asyncio
import logging
from typing import Dict, Any, List, Set
from datetime import datetime
import json
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketBroadcaster:
    """Manages WebSocket connections and broadcasts data to connected clients"""
    
    def __init__(self):
        # Store active connections by channel
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.all_connections: Set[WebSocket] = set()
        
        # Channels
        self.channels = [
            "sensors",      # Real-time sensor data
            "predictions",  # ML model predictions
            "alerts",       # Alert notifications
            "recommendations",  # RL agent recommendations
            "quality",      # Quality metrics and KPIs
            "all"           # All channels combined
        ]
        
        # Message counters for stats
        self.message_stats = defaultdict(int)
        
    async def connect(self, websocket: WebSocket, channels: List[str] = None):
        """
        Accept a new WebSocket connection and subscribe to channels
        
        Args:
            websocket: FastAPI WebSocket instance
            channels: List of channel names to subscribe to. If None, subscribe to 'all'
        """
        await websocket.accept()
        
        if channels is None:
            channels = ["all"]
        
        # Add to channel subscriptions
        for channel in channels:
            if channel in self.channels:
                self.active_connections[channel].add(websocket)
        
        # Add to all connections
        self.all_connections.add(websocket)
        
        logger.info(f"📡 New WebSocket connection. Channels: {channels}. Total connections: {len(self.all_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        # Remove from all channels
        for channel in self.channels:
            self.active_connections[channel].discard(websocket)
        
        # Remove from all connections
        self.all_connections.discard(websocket)
        
        logger.info(f"📴 WebSocket disconnected. Total connections: {len(self.all_connections)}")
    
    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """
        Broadcast a message to all clients subscribed to a channel
        
        Args:
            channel: Channel name
            message: Message data to broadcast
        """
        if channel not in self.channels:
            logger.warning(f"⚠️ Unknown channel: {channel}")
            return
        
        # Add metadata
        message_with_metadata = {
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **message
        }
        
        # Serialize to JSON
        try:
            message_json = json.dumps(message_with_metadata)
        except Exception as e:
            logger.error(f"❌ Error serializing message: {e}")
            return
        
        # Get connections for this channel and 'all' channel
        target_connections = self.active_connections[channel].union(
            self.active_connections["all"]
        )
        
        # Broadcast to all connections
        disconnected = []
        for connection in target_connections:
            try:
                await connection.send_text(message_json)
                self.message_stats[channel] += 1
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"❌ Error sending message to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
        
        if target_connections:
            logger.debug(f"📤 Broadcasted to {len(target_connections)} clients on channel '{channel}'")
    
    async def broadcast_sensor_update(self, sensor_data: Dict[str, Any]):
        """Broadcast sensor data update"""
        await self.broadcast_to_channel("sensors", {
            "type": "sensor_update",
            "data": sensor_data
        })
    
    async def broadcast_prediction_update(self, predictions: Dict[str, Any]):
        """Broadcast ML model predictions"""
        await self.broadcast_to_channel("predictions", {
            "type": "prediction_update",
            "data": predictions
        })
    
    async def broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert notification"""
        await self.broadcast_to_channel("alerts", {
            "type": "alert",
            "data": alert
        })
    
    async def broadcast_recommendation(self, recommendation: Dict[str, Any]):
        """Broadcast RL agent recommendation"""
        await self.broadcast_to_channel("recommendations", {
            "type": "recommendation",
            "data": recommendation
        })
    
    async def broadcast_quality_metrics(self, metrics: Dict[str, Any]):
        """Broadcast quality metrics and KPIs"""
        await self.broadcast_to_channel("quality", {
            "type": "quality_metrics",
            "data": metrics
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics"""
        return {
            "total_connections": len(self.all_connections),
            "connections_by_channel": {
                channel: len(connections)
                for channel, connections in self.active_connections.items()
            },
            "messages_sent_by_channel": dict(self.message_stats),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global broadcaster instance
broadcaster = WebSocketBroadcaster()


async def sensor_data_stream_task(broadcaster: WebSocketBroadcaster, data_generator):
    """
    Background task to stream sensor data
    
    Args:
        broadcaster: WebSocketBroadcaster instance
        data_generator: GlassProductionDataGenerator instance
    """
    logger.info("🔄 Starting sensor data stream task...")
    
    try:
        while True:
            # Generate sensor readings
            readings = []
            for sensor_key in data_generator.sensors.keys():
                reading = data_generator.generate_sensor_reading(sensor_key)
                readings.append(reading)
            
            # Aggregate by production line
            aggregated = {
                "production_line": "Line_A",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sensors": readings,
                "state_summary": data_generator.get_current_state_summary()
            }
            
            # Broadcast sensor update
            await broadcaster.broadcast_sensor_update(aggregated)
            
            # Wait before next update (5-10 seconds for dashboard)
            await asyncio.sleep(5)
            
    except asyncio.CancelledError:
        logger.info("⏹️ Sensor data stream task cancelled")
    except Exception as e:
        logger.error(f"❌ Error in sensor data stream task: {e}")


async def defect_detection_stream_task(broadcaster: WebSocketBroadcaster, data_generator):
    """
    Background task to stream defect detection events
    
    Args:
        broadcaster: WebSocketBroadcaster instance
        data_generator: GlassProductionDataGenerator instance
    """
    logger.info("🔄 Starting defect detection stream task...")
    
    try:
        while True:
            # Generate defect event
            defect = data_generator.generate_defect_event()
            
            if defect:
                # Create alert from defect
                alert = {
                    "severity": defect["severity"],
                    "type": "defect_detected",
                    "message": f"{defect['defect_type'].capitalize()} detected on {defect['production_line']}",
                    "defect_type": defect["defect_type"],
                    "confidence": defect["confidence"],
                    "production_line": defect["production_line"],
                    "position": {
                        "x": defect["position_x"],
                        "y": defect["position_y"]
                    },
                    "timestamp": defect["timestamp"]
                }
                
                # Broadcast alert
                await broadcaster.broadcast_alert(alert)
                logger.info(f"⚠️ Alert broadcasted: {alert['message']}")
            
            # Check every 3-5 seconds (MIK-1 inspection rate)
            await asyncio.sleep(4)
            
    except asyncio.CancelledError:
        logger.info("⏹️ Defect detection stream task cancelled")
    except Exception as e:
        logger.error(f"❌ Error in defect detection stream task: {e}")


async def quality_metrics_stream_task(broadcaster: WebSocketBroadcaster, data_generator):
    """
    Background task to stream quality metrics
    
    Args:
        broadcaster: WebSocketBroadcaster instance
        data_generator: GlassProductionDataGenerator instance
    """
    logger.info("🔄 Starting quality metrics stream task...")
    
    try:
        while True:
            # Calculate quality metrics (simplified for demo)
            state = data_generator.current_state
            
            # Estimate defect rate based on parameter deviations
            temp_deviation = abs(state["furnace_temperature"] - 1520) / 30
            speed_deviation = abs(state["belt_speed"] - 150) / 10
            overall_deviation = (temp_deviation + speed_deviation) / 2
            
            defect_rate = min(2 + overall_deviation * 3, 15)  # 2-15% range
            quality_score = max(100 - defect_rate, 70)  # 70-98% range
            
            metrics = {
                "production_line": "Line_A",
                "defect_rate": round(defect_rate, 2),
                "quality_score": round(quality_score, 2),
                "production_rate": round(150 + np.random.normal(0, 5), 1),  # units/hour
                "total_defects": int(defect_rate * 10),  # Defects in last hour
                "critical_alerts": len([a for a in data_generator.current_anomalies if True]),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            # Broadcast quality metrics
            await broadcaster.broadcast_quality_metrics(metrics)
            
            # Update every 30 seconds
            await asyncio.sleep(30)
            
    except asyncio.CancelledError:
        logger.info("⏹️ Quality metrics stream task cancelled")
    except Exception as e:
        logger.error(f"❌ Error in quality metrics stream task: {e}")


# Import numpy for quality metrics calculation
import numpy as np


async def start_background_tasks(broadcaster: WebSocketBroadcaster, data_generator):
    """
    Start all background streaming tasks
    
    Args:
        broadcaster: WebSocketBroadcaster instance
        data_generator: GlassProductionDataGenerator instance
    """
    tasks = [
        asyncio.create_task(sensor_data_stream_task(broadcaster, data_generator)),
        asyncio.create_task(defect_detection_stream_task(broadcaster, data_generator)),
        asyncio.create_task(quality_metrics_stream_task(broadcaster, data_generator)),
    ]
    
    return tasks


if __name__ == "__main__":
    # Test the broadcaster
    from data_ingestion.synthetic_data_generator import GlassProductionDataGenerator
    
    async def test_broadcaster():
        broadcaster = WebSocketBroadcaster()
        generator = GlassProductionDataGenerator()
        
        logger.info("🧪 Testing WebSocket broadcaster...")
        
        # Start background tasks
        tasks = await start_background_tasks(broadcaster, generator)
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Cancel tasks
        for task in tasks:
            task.cancel()
        
        # Show stats
        stats = broadcaster.get_stats()
        logger.info(f"📊 Broadcaster stats: {stats}")
    
    asyncio.run(test_broadcaster())
