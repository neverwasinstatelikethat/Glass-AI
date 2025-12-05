"""
MQTT Broker Connector for IoT Device Communication
Supports secure MQTT connections with TLS and authentication
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import json
import ssl
from aiomqtt import Client, MqttError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MQTTBrokerConnector:
    """Asynchronous MQTT client for industrial IoT communication"""
    
    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: str = "glass_production_client",
        use_tls: bool = False,
        tls_ca_cert: Optional[str] = None,
        tls_cert: Optional[str] = None,
        tls_key: Optional[str] = None,
        keepalive: int = 60,
        callback: Optional[Callable] = None
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        self.client_id = client_id
        self.use_tls = use_tls
        self.tls_ca_cert = tls_ca_cert
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.keepalive = keepalive
        self.callback = callback
        self.client: Optional[Client] = None
        self.connected = False
        self.subscriptions: Dict[str, Callable] = {}
        self.running = False
        
        # Topic mappings for glass production
        self.topic_mappings = {
            "sensors": "glass/production/sensors/#",
            "defects": "glass/production/defects/#",
            "quality": "glass/production/quality/#",
            "alarms": "glass/production/alarms/#",
            "control": "glass/production/control/#"
        }
    
    async def connect(self) -> bool:
        """Connect to MQTT broker with retry logic"""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Configure TLS if requested
                tls_context = None
                if self.use_tls:
                    tls_context = ssl.create_default_context()
                    if self.tls_ca_cert:
                        tls_context.load_verify_locations(self.tls_ca_cert)
                    if self.tls_cert and self.tls_key:
                        tls_context.load_cert_chain(self.tls_cert, self.tls_key)
                
                # Store connection parameters for later use
                self._connection_params = {
                    "hostname": self.broker_host,
                    "port": self.broker_port,
                    "username": self.username,
                    "password": self.password,
                    "identifier": self.client_id,
                    "tls_context": tls_context,
                    "keepalive": self.keepalive
                }
                
                # Test connection by creating a temporary client
                # Use a longer timeout for Windows compatibility
                test_params = self._connection_params.copy()
                if "timeout" not in test_params:
                    test_params["timeout"] = 10
                
                async with Client(**test_params) as test_client:
                    logger.info(f"✅ Successfully connected to MQTT broker: {self.broker_host}:{self.broker_port}")
                    # Just test the connection, don't do anything else
                
                # Mark as connected
                self.connected = True
                
                logger.info(f"✅ MQTT client ready for broker: {self.broker_host}:{self.broker_port}")
                if self.use_tls:
                    logger.info("🔒 TLS encryption enabled")
                
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ MQTT connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"❌ Error connecting to MQTT broker after {max_retries} attempts: {e}")
                    self.connected = False
                    return False
    
    async def disconnect(self):
        """Disconnect from MQTT broker"""
        self.running = False
        
        # In aiomqtt, disconnection happens automatically when exiting context managers
        # So we just need to mark as disconnected
        self.connected = False
        logger.info("✅ Disconnected from MQTT broker")
    
    async def subscribe(self, topic: str, qos: int = 1, handler: Optional[Callable] = None) -> bool:
        """Subscribe to a topic"""
        if not self.connected:
            logger.error("❌ MQTT client not connected")
            return False
        
        try:
            # Create a temporary client for subscription
            # Use a longer timeout for Windows compatibility
            temp_params = self._connection_params.copy()
            if "timeout" not in temp_params:
                temp_params["timeout"] = 10
                
            async with Client(**temp_params) as temp_client:
                await temp_client.subscribe(topic, qos=qos)
            
            # Store handler for this topic
            if handler:
                self.subscriptions[topic] = handler
            elif self.callback:
                self.subscriptions[topic] = self.callback
            
            logger.info(f"✅ Subscribed to topic: {topic} (QoS {qos})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error subscribing to {topic}: {e}")
            return False
    
    async def publish(self, topic: str, payload: Any, qos: int = 1, retain: bool = False) -> bool:
        """Publish message to a topic"""
        if not self.connected:
            logger.error("❌ MQTT client not connected")
            return False
        
        try:
            # Convert payload to JSON if it's a dict
            if isinstance(payload, dict):
                payload_str = json.dumps(payload, ensure_ascii=False)
            else:
                payload_str = str(payload)
            
            # Create a temporary client for publishing
            # Use a longer timeout for Windows compatibility
            temp_params = self._connection_params.copy()
            if "timeout" not in temp_params:
                temp_params["timeout"] = 10
                
            async with Client(**temp_params) as temp_client:
                await temp_client.publish(topic, payload_str, qos=qos, retain=retain)
            
            logger.debug(f"📤 Published to {topic}: {payload_str[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error publishing to {topic}: {e}")
            return False
    
    async def start_listening(self):
        """Start listening for messages on subscribed topics"""
        if not self.connected:
            logger.error("❌ MQTT client not connected")
            return
        
        self.running = True
        logger.info("👂 Starting MQTT message listener...")
        
        try:
            # Create a new client instance for listening
            # Use a longer timeout for Windows compatibility
            listen_params = self._connection_params.copy()
            if "timeout" not in listen_params:
                listen_params["timeout"] = 10
                
            listening_client = Client(**listen_params)
            
            # Use the new client within context for proper connection handling
            async with listening_client:
                # Subscribe to all topics first
                for topic in self.subscriptions.keys():
                    await listening_client.subscribe(topic)
                    logger.debug(f"📡 Subscribed to {topic} for listening")
                
                # Check if we're using the newer aiomqtt version with filtered_messages
                if hasattr(listening_client, 'filtered_messages'):
                    # Newer version - use filtered_messages
                    async with listening_client.filtered_messages("#") as messages:
                        await listening_client.subscribe("#")
                        async for message in messages:
                            if not self.running:
                                break
                            
                            await self._process_message(message)
                else:
                    # Older version - use the messages iterator directly
                    async with listening_client.messages() as messages:
                        await listening_client.subscribe("#")
                        async for message in messages:
                            if not self.running:
                                break
                            
                            await self._process_message(message)
        
        except Exception as e:
            logger.error(f"❌ Error in MQTT listener: {e}")
        finally:
            self.running = False
            logger.info("👂 MQTT message listener stopped")
    
    async def _process_message(self, message):
        """Process an incoming MQTT message"""
        try:
            # Parse message
            topic = message.topic.value
            payload = message.payload.decode()
            
            # Try to parse as JSON
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"raw_payload": payload}
            
            # Add metadata
            enriched_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "topic": topic,
                "data": data
            }
            
            # Handle message based on topic
            handled = False
            for subscription_topic, handler in self.subscriptions.items():
                if self._topic_matches(subscription_topic, topic):
                    try:
                        await handler(enriched_data)
                        handled = True
                    except Exception as e:
                        logger.error(f"❌ Error in handler for {topic}: {e}")
            
            # If no specific handler, use default callback
            if not handled and self.callback:
                try:
                    await self.callback(enriched_data)
                except Exception as e:
                    logger.error(f"❌ Error in default callback for {topic}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    def _topic_matches(self, subscription: str, topic: str) -> bool:
        """Check if a topic matches a subscription pattern (including wildcards)"""
        if subscription == topic:
            return True
        
        # Handle wildcards
        if subscription.endswith("#"):
            prefix = subscription[:-1]  # Remove #
            return topic.startswith(prefix)
        elif "+" in subscription:
            # Single level wildcard
            sub_parts = subscription.split("/")
            topic_parts = topic.split("/")
            
            if len(sub_parts) != len(topic_parts):
                return False
            
            for s_part, t_part in zip(sub_parts, topic_parts):
                if s_part != "+" and s_part != t_part:
                    return False
            
            return True
        
        return False
    
    async def subscribe_to_production_topics(self):
        """Subscribe to all standard production topics"""
        for topic_name, topic_pattern in self.topic_mappings.items():
            try:
                success = await self.subscribe(topic_pattern)
                if not success:
                    logger.warning(f"⚠️ Failed to subscribe to {topic_name} topic: {topic_pattern}")
            except Exception as e:
                logger.error(f"❌ Error subscribing to {topic_name} topic: {e}")
    
    async def publish_sensor_data(self, sensor_data: Dict[str, Any], production_line: str = "Line_A"):
        """Publish sensor data to the appropriate topic"""
        topic = f"glass/production/sensors/{production_line}"
        return await self.publish(topic, sensor_data)
    
    async def publish_defect_data(self, defect_data: Dict[str, Any], production_line: str = "Line_A"):
        """Publish defect data to the appropriate topic"""
        topic = f"glass/production/defects/{production_line}/{defect_data.get('defect_type', 'unknown')}"
        return await self.publish(topic, defect_data)
    
    async def publish_quality_data(self, quality_data: Dict[str, Any], production_line: str = "Line_A"):
        """Publish quality data to the appropriate topic"""
        topic = f"glass/production/quality/{production_line}"
        return await self.publish(topic, quality_data)
    
    async def publish_alarm(self, alarm_data: Dict[str, Any], priority: str = "MEDIUM"):
        """Publish alarm data to the appropriate topic"""
        topic = f"glass/production/alarms/{priority.lower()}"
        return await self.publish(topic, alarm_data)
    
    async def send_control_command(self, command: str, target: str, parameters: Dict[str, Any]):
        """Send control command to a target device"""
        topic = f"glass/production/control/{target}"
        payload = {
            "command": command,
            "parameters": parameters,
            "timestamp": datetime.utcnow().isoformat()
        }
        return await self.publish(topic, payload)


class MQTTSimulator:
    """Simulator for MQTT broker for testing purposes"""
    
    def __init__(self):
        self.topics = {}
        self.messages = []
    
    async def simulate_sensor_data(self, production_line: str = "Line_A") -> Dict[str, Any]:
        """Generate simulated sensor data"""
        import numpy as np
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "production_line": production_line,
            "sensors": {
                "furnace": {
                    "temperature": 1500 + (50 * (0.5 - np.random.random())),
                    "pressure": 15 + (3 * (0.5 - np.random.random())),
                    "melt_level": 2500 + (200 * (0.5 - np.random.random()))
                },
                "forming": {
                    "belt_speed": 150 + (30 * (0.5 - np.random.random())),
                    "mold_temp": 320 + (40 * (0.5 - np.random.random())),
                    "quality_score": 0.95 + (0.05 * (0.5 - np.random.random()))
                }
            }
        }
    
    async def simulate_defect_data(self) -> Dict[str, Any]:
        """Generate simulated defect data"""
        import numpy as np
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "defect_type": np.random.choice(["crack", "bubble", "chip", "cloudiness", "deformation"]),
            "severity": np.random.choice(["LOW", "MEDIUM", "HIGH"]),
            "position": {
                "x": float(np.random.randint(0, 1000)),
                "y": float(np.random.randint(0, 500))
            },
            "size_mm": float(np.random.uniform(0.1, 10.0)),
            "confidence": float(np.random.uniform(0.7, 0.99))
        }


async def main_example():
    """Example usage of MQTT Broker Connector"""
    
    async def message_callback(data):
        """Callback for incoming MQTT messages"""
        print(f"📥 MQTT Message on {data['topic']}")
        print(f"   Timestamp: {data['timestamp']}")
        print(f"   Data: {json.dumps(data['data'], indent=2, ensure_ascii=False)}")
    
    # Create MQTT client
    mqtt_client = MQTTBrokerConnector(
        broker_host="localhost",
        broker_port=1883,
        client_id="glass_production_demo",
        callback=message_callback
    )
    
    try:
        # Connect to broker
        if await mqtt_client.connect():
            print("✅ Connected to MQTT broker")
            
            # Subscribe to production topics
            await mqtt_client.subscribe_to_production_topics()
            
            # Start listening
            listen_task = asyncio.create_task(mqtt_client.start_listening())
            
            # Simulate publishing some data
            simulator = MQTTSimulator()
            
            print("🔄 Publishing sample data (will run for 30 seconds)...")
            for i in range(10):
                # Publish sensor data
                sensor_data = await simulator.simulate_sensor_data()
                await mqtt_client.publish_sensor_data(sensor_data)
                
                # Occasionally publish defect data
                if i % 3 == 0:
                    defect_data = await simulator.simulate_defect_data()
                    await mqtt_client.publish_defect_data(defect_data)
                
                await asyncio.sleep(3)
            
            # Stop listening
            mqtt_client.running = False
            listen_task.cancel()
            
        else:
            print("❌ Failed to connect to MQTT broker")
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error in demo: {e}")
    finally:
        await mqtt_client.disconnect()


if __name__ == "__main__":
    import numpy as np
    asyncio.run(main_example())