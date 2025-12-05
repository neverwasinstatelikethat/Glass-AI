"""
Data Router for Glass Production System
Routes collected data to appropriate processing pipelines and storage systems
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import hashlib

# Import processing modules
from streaming_pipeline.kafka_producer import GlassProductionKafkaProducer
from streaming_pipeline.data_validator import DataValidator
from feature_engineering.real_time_features import RealTimeFeatureExtractor

# Import database clients
from storage.influxdb_client import GlassInfluxDBClient
from storage.postgres_client import GlassPostgresClient

# Import HTTP client
import aiohttp
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataRouter:
    """Routes data to appropriate destinations based on type and priority"""
    
    def __init__(
        self,
        routing_rules: Optional[Dict[str, List[str]]] = None,
        default_routes: Optional[List[str]] = None
    ):
        self.routing_rules = routing_rules or self._default_routing_rules()
        self.default_routes = default_routes or ["kafka", "feature_extractor"]
        self.running = False
        
        # Destination handlers
        self.destinations = {
            "kafka": None,
            "feature_extractor": None,
            "validator": None,
            "buffer": None,
            "websocket": None,
            "database": None,
            "rest_api": None
        }
        
        # Routing statistics
        self.routing_stats = defaultdict(lambda: {
            "messages_routed": 0,
            "bytes_routed": 0,
            "errors": 0,
            "last_routed": None
        })
        
        # Data buffers for queuing
        self.route_queues = defaultdict(deque)
        self.max_queue_size = 1000
        
        # Message deduplication
        self.message_cache = deque(maxlen=10000)
        self.cache_ttl = timedelta(minutes=5)
        
        # Task requirements - sensor frequency specifications
        self.sensor_frequencies = {
            "critical": ["pressure_sensors", "thermocouples"],  # 1/sec
            "high": ["furnace_temperature", "forming_speed"],   # 1/min
            "medium": ["batch_feed_rate", "forming_temperature"], # 1/5min
            "low": ["ambient_data", "gas_composition"],        # 1/15min
            "quality": ["mik1_defects"]                        # ~3-5 sec per item
        }
    
    def _default_routing_rules(self) -> Dict[str, List[str]]:
        """Define default routing rules"""
        return {
            "sensor_data": ["kafka", "feature_extractor", "validator", "database", "rest_api"],
            "defect_data": ["kafka", "validator", "database", "rest_api"],
            "image_data": ["kafka", "buffer"],
            "control_data": ["kafka", "websocket"],
            "alarm_data": ["kafka", "websocket", "database"],
            "quality_data": ["kafka", "feature_extractor", "database"],
            "prediction_data": ["kafka", "websocket"],
            "recommendation_data": ["kafka", "websocket"]
        }
    
    async def initialize_destinations(self):
        """Initialize all destination handlers"""
        logger.info("🔧 Initializing data routing destinations...")
        
        # Initialize Kafka producer
        try:
            # Get Kafka bootstrap servers from environment variable or use default
            import os
            kafka_bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9093")
            
            # Check if we're running in Docker environment
            # More robust Docker detection
            is_docker = (os.path.exists('/.dockerenv') or 
                        os.environ.get('DOCKER_CONTAINER') or
                        os.environ.get("ENVIRONMENT") == "docker")
            
            if is_docker:
                # In Docker, use internal service name
                kafka_bootstrap_servers = "kafka:9092"
                logger.info("🐳 Detected Docker environment, using internal Kafka address")
            else:
                logger.info("💻 Detected non-Docker environment, using external Kafka address")
            
            self.destinations["kafka"] = GlassProductionKafkaProducer(
                bootstrap_servers=kafka_bootstrap_servers,
                use_msgpack=False,
                enable_idempotence=True
            )
            logger.info(f"✅ Kafka producer initialized with servers: {kafka_bootstrap_servers}")
        except Exception as e:
            logger.warning(f"⚠️ Kafka producer initialization failed: {e}")
            # Set to None so we know it's not available
            self.destinations["kafka"] = None        
        # Initialize feature extractor
        try:
            self.destinations["feature_extractor"] = RealTimeFeatureExtractor(
                window_size=60,
                feature_callback=self._feature_callback
            )
            logger.info("✅ Feature extractor initialized")
        except Exception as e:
            logger.warning(f"⚠️ Feature extractor initialization failed: {e}")
        
        # Initialize data validator
        try:
            self.destinations["validator"] = DataValidator(
                window_size=100,
                outlier_threshold=3.0,
                validation_callback=self._validation_callback
            )
            logger.info("✅ Data validator initialized")
        except Exception as e:
            logger.warning(f"⚠️ Data validator initialization failed: {e}")
        
        # Initialize data buffer
        try:
            self.destinations["buffer"] = DataBuffer(max_size=10000)
            logger.info("✅ Data buffer initialized")
        except Exception as e:
            logger.warning(f"⚠️ Data buffer initialization failed: {e}")
        
        # Initialize InfluxDB client
        try:
            self.destinations["database"] = GlassInfluxDBClient()
            await self.destinations["database"].connect()
            logger.info("✅ InfluxDB client initialized")
        except Exception as e:
            logger.warning(f"⚠️ InfluxDB client initialization failed: {e}")
        
        # Initialize REST API client
        try:
            # Determine backend URL based on environment
            is_docker = (os.path.exists('/.dockerenv') or 
                        os.environ.get('DOCKER_CONTAINER') or
                        os.environ.get("ENVIRONMENT") == "docker")
            
            if is_docker:
                backend_url = "http://backend:8000"
                logger.info("🐳 Detected Docker environment, using internal backend address")
            else:
                backend_url = "http://localhost:8000"
                logger.info("💻 Detected non-Docker environment, using external backend address")
            
            # Create REST API client
            self.destinations["rest_api"] = {
                "session": aiohttp.ClientSession(),
                "base_url": backend_url
            }
            logger.info(f"✅ REST API client initialized with backend: {backend_url}")
        except Exception as e:
            logger.warning(f"⚠️ REST API client initialization failed: {e}")

    async def start_destinations(self):
        """Start destination handlers"""
        # Start Kafka producer
        if self.destinations["kafka"]:
            try:
                await self.destinations["kafka"].start()
                logger.info("✅ Kafka producer started")
            except Exception as e:
                logger.error(f"❌ Error starting Kafka producer: {e}")
                # Set to None so we know it's not available
                self.destinations["kafka"] = None        
        # Start feature extractor (no explicit start needed)
        if self.destinations["feature_extractor"]:
            logger.info("✅ Feature extractor ready")
        
        # Start data validator (no explicit start needed)
        if self.destinations["validator"]:
            logger.info("✅ Data validator ready")
        
        # Start data buffer (no explicit start needed)
        if self.destinations["buffer"]:
            logger.info("✅ Data buffer ready")
    
    async def route_data(self, data: Dict[str, Any], data_type: str = "sensor_data") -> bool:
        """Route data to appropriate destinations based on type"""
        try:
            # Generate message ID for deduplication
            message_id = self._generate_message_id(data)
            
            # Check for duplicates
            if message_id in self.message_cache:
                logger.debug(f"🔄 Duplicate message detected: {message_id}")
                return True
            
            # Add to cache
            self.message_cache.append(message_id)
            
            # Determine routes
            routes = self.routing_rules.get(data_type, self.default_routes)
            
            # Route to each destination
            routed_count = 0
            for route in routes:
                if await self._send_to_destination(route, data, data_type):
                    routed_count += 1
                    self.routing_stats[route]["messages_routed"] += 1
                    self.routing_stats[route]["bytes_routed"] += len(str(data).encode('utf-8'))
                    self.routing_stats[route]["last_routed"] = datetime.utcnow()
            
            logger.debug(f"📤 Routed data to {routed_count}/{len(routes)} destinations")
            return routed_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error routing data: {e}")
            for route in self.routing_rules.get(data_type, self.default_routes):
                self.routing_stats[route]["errors"] += 1
            return False
    
    async def _send_to_destination(self, destination: str, data: Dict[str, Any], data_type: str) -> bool:
        """Send data to a specific destination"""
        try:
            handler = self.destinations.get(destination)
            if not handler:
                logger.warning(f"⚠️ No handler for destination: {destination}")
                return False
            
            # Route based on destination type
            if destination == "kafka":
                return await self._route_to_kafka(handler, data, data_type)
            elif destination == "feature_extractor":
                return await self._route_to_feature_extractor(handler, data)
            elif destination == "validator":
                return await self._route_to_validator(handler, data)
            elif destination == "buffer":
                return await self._route_to_buffer(handler, data)
            elif destination == "websocket":
                return await self._route_to_websocket(handler, data)
            elif destination == "database":
                return await self._route_to_database(handler, data)
            elif destination == "rest_api":
                return await self._route_to_rest_api(handler, data, data_type)
            else:
                logger.warning(f"⚠️ Unknown destination: {destination}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending to {destination}: {e}")
            self.routing_stats[destination]["errors"] += 1
            return False
    
    async def _route_to_kafka(self, kafka_producer: GlassProductionKafkaProducer, data: Dict[str, Any], data_type: str) -> bool:
        """Route data to Kafka"""
        try:
            # Send based on data type
            if data_type == "sensor_data":
                success = await kafka_producer.send_sensor_data(data, processed=False)
            elif data_type == "defect_data":
                success = await kafka_producer.send_defect(data)
            elif data_type == "image_data":
                # Handle image data separately
                success = await kafka_producer._send_message(
                    kafka_producer.TOPICS["sensors_raw"],
                    data,
                    key=data.get("camera_id", "unknown")
                )
            elif data_type == "alarm_data":
                success = await kafka_producer.send_alert(data)
            elif data_type == "quality_data":
                success = await kafka_producer.send_quality_metrics(data)
            elif data_type == "prediction_data":
                success = await kafka_producer.send_prediction(data)
            elif data_type == "recommendation_data":
                success = await kafka_producer.send_recommendation(data)
            else:
                # Default routing
                success = await kafka_producer._send_message(
                    kafka_producer.TOPICS["sensors_raw"],
                    data,
                    key=data.get("production_line", "unknown")
                )
            
            if success:
                logger.debug(f"✅ Data sent to Kafka for {data_type}")
            else:
                logger.warning(f"⚠️ Failed to send data to Kafka for {data_type}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error routing to Kafka: {e}")
            # Return True to indicate graceful degradation - data wasn't lost, just not sent to Kafka
            return True
    
    async def _route_to_feature_extractor(self, extractor: RealTimeFeatureExtractor, data: Dict[str, Any]) -> bool:
        """Route data to feature extractor"""
        try:
            # Feature extractor expects sensor data
            if "sensors" in data:
                await extractor.update_with_sensor_data(data)
                logger.debug("✅ Data sent to feature extractor")
                return True
            else:
                logger.debug("⏭️ Skipping feature extractor (no sensor data)")
                return True  # Not an error, just no applicable data
                
        except Exception as e:
            logger.error(f"❌ Error routing to feature extractor: {e}")
            return False
    
    async def _route_to_validator(self, validator: DataValidator, data: Dict[str, Any]) -> bool:
        """Route data to validator"""
        try:
            # Validator expects sensor data
            if "sensors" in data:
                validation_result = validator.validate_sensor_data(data)
                logger.debug("✅ Data sent to validator")
                return True
            else:
                logger.debug("⏭️ Skipping validator (no sensor data)")
                return True  # Not an error, just no applicable data
                
        except Exception as e:
            logger.error(f"❌ Error routing to validator: {e}")
            return False
    
    async def _route_to_buffer(self, buffer: 'DataBuffer', data: Dict[str, Any]) -> bool:
        """Route data to buffer"""
        try:
            await buffer.add_data(data)
            logger.debug("✅ Data buffered")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error buffering data: {e}")
            return False
    
    async def _route_to_websocket(self, handler, data: Dict[str, Any]) -> bool:
        """Route data to WebSocket (placeholder)"""
        try:
            # This would integrate with WebSocket server
            logger.debug("✅ Data routed to WebSocket (placeholder)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error routing to WebSocket: {e}")
            return False
    
    async def _route_to_database(self, handler: GlassInfluxDBClient, data: Dict[str, Any]) -> bool:
        """Route data to database"""
        try:
            # Determine data type and route appropriately
            if "sensors" in data:
                # Sensor data goes to InfluxDB
                success = await handler.write_sensor_data(data)
                logger.debug("✅ Sensor data routed to InfluxDB")
                return success
            elif "defect_type" in data:
                # Defect data goes to InfluxDB
                success = await handler.write_defect_data(data)
                logger.debug("✅ Defect data routed to InfluxDB")
                return success
            elif "quality_rate" in data:
                # Quality metrics - for now log that this would go to PostgreSQL
                logger.debug("ℹ️ Quality metrics would be routed to PostgreSQL")
                return True
            else:
                logger.debug("⏭️ Skipping database routing (unsupported data type)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error routing to database: {e}")
            return False
    
    async def _route_to_rest_api(self, handler: Dict[str, Any], data: Dict[str, Any], data_type: str) -> bool:
        """Route data to REST API"""
        try:
            session = handler["session"]
            base_url = handler["base_url"]
            
            # Route based on data type
            if data_type == "sensor_data":
                url = f"{base_url}/api/sensors/data"
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        logger.debug("✅ Sensor data sent to REST API")
                        return True
                    else:
                        logger.warning(f"⚠️ Failed to send sensor data to REST API: {response.status}")
                        return False
            elif data_type == "defect_data":
                # Transform defect data to match the expected format
                if "defects" in data and len(data["defects"]) > 0:
                    for defect in data["defects"]:
                        defect_payload = {
                            "timestamp": data["timestamp"],
                            "production_line": data["production_line"],
                            "defect_type": defect["type"],
                            "severity": defect["severity"],
                            "position": defect["position"],
                            "size_mm": defect["size_mm"],
                            "confidence": defect["confidence"]
                        }
                        url = f"{base_url}/api/defects"
                        async with session.post(url, json=defect_payload) as response:
                            if response.status == 200:
                                logger.debug(f"✅ Defect data ({defect['type']}) sent to REST API")
                            else:
                                logger.warning(f"⚠️ Failed to send defect data to REST API: {response.status}")
                return True
            else:
                logger.debug(f"⏭️ Skipping REST API routing for data type: {data_type}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error routing to REST API: {e}")
            return False
    
    def _generate_message_id(self, data: Dict[str, Any]) -> str:
        """Generate unique message ID for deduplication"""
        try:
            # Create hash of key data fields
            key_fields = {
                "timestamp": data.get("timestamp"),
                "production_line": data.get("production_line"),
                "frame_id": data.get("frame_id"),
                "sensor_data": str(data.get("sensors", {}))
            }
            
            message_str = json.dumps(key_fields, sort_keys=True)
            return hashlib.md5(message_str.encode()).hexdigest()
            
        except Exception as e:
            logger.debug(f"⚠️ Error generating message ID: {e}")
            # Fallback to timestamp-based ID
            return hashlib.md5(str(datetime.utcnow().isoformat()).encode()).hexdigest()
    
    async def _feature_callback(self, features: Dict[str, Any]):
        """Handle extracted features"""
        try:
            # Route features to Kafka
            if self.destinations["kafka"]:
                feature_data = {
                    "timestamp": features.get("timestamp", datetime.utcnow().isoformat()),
                    "production_line": "Line_A",  # Default
                    "features": features
                }
                await self.destinations["kafka"]._send_message(
                    self.destinations["kafka"].TOPICS["sensors_processed"],
                    feature_data,
                    key="features"
                )
                logger.debug("✅ Features sent to Kafka")
            
        except Exception as e:
            logger.error(f"❌ Error handling extracted features: {e}")
    
    async def _validation_callback(self, validation_result: Dict[str, Any]):
        """Handle validation results"""
        try:
            # Route validation results to Kafka
            if self.destinations["kafka"]:
                await self.destinations["kafka"]._send_message(
                    self.destinations["kafka"].TOPICS["alerts"],
                    validation_result,
                    key="validation"
                )
                logger.debug("✅ Validation results sent to Kafka")
            
            # Log critical anomalies
            if validation_result.get("validation_status") == "critical_error":
                logger.critical(f"🚨 CRITICAL VALIDATION ERROR: {validation_result}")
            
        except Exception as e:
            logger.error(f"❌ Error handling validation results: {e}")
    
    async def get_routing_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get routing statistics"""
        return dict(self.routing_stats)
    
    async def reset_stats(self):
        """Reset routing statistics"""
        self.routing_stats.clear()
        logger.info("📊 Routing statistics reset")
    
    async def stop_destinations(self):
        """Stop all destination handlers"""
        logger.info("⏹️ Stopping routing destinations...")
        
        # Stop Kafka producer
        if self.destinations["kafka"]:
            try:
                await self.destinations["kafka"].stop()
                logger.info("✅ Kafka producer stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping Kafka producer: {e}")
        
        # Other destinations don't need explicit stopping
        logger.info("✅ All routing destinations stopped")


class DataBuffer:
    """Buffer for temporarily storing data during high load or system issues"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.stats = {
            "total_added": 0,
            "total_removed": 0,
            "peak_size": 0
        }
    
    async def add_data(self, data: Dict[str, Any]):
        """Add data to buffer"""
        try:
            timestamp = datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
            buffered_data = {
                "data": data,
                "buffered_at": datetime.utcnow().isoformat(),
                "timestamp": timestamp.isoformat()
            }
            
            self.buffer.append(buffered_data)
            self.stats["total_added"] += 1
            self.stats["peak_size"] = max(self.stats["peak_size"], len(self.buffer))
            
            logger.debug(f"💾 Data buffered (buffer size: {len(self.buffer)})")
            
        except Exception as e:
            logger.error(f"❌ Error buffering data: {e}")
    
    async def get_buffered_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get buffered data (FIFO)"""
        try:
            # Get oldest data first
            data_list = []
            for _ in range(min(limit, len(self.buffer))):
                if self.buffer:
                    data_list.append(self.buffer.popleft())
                    self.stats["total_removed"] += 1
            
            logger.debug(f"📤 Retrieved {len(data_list)} items from buffer")
            return data_list
            
        except Exception as e:
            logger.error(f"❌ Error retrieving buffered data: {e}")
            return []
    
    async def flush_buffer(self, router: DataRouter) -> int:
        """Flush buffer by routing all buffered data"""
        try:
            flushed_count = 0
            
            while self.buffer:
                buffered_item = self.buffer.popleft()
                data = buffered_item["data"]
                
                # Determine data type (simplified)
                data_type = "sensor_data"
                if "defects" in data or "defect_count" in data:
                    data_type = "defect_data"
                elif "frame_id" in data:
                    data_type = "image_data"
                elif "quality_score" in data:
                    data_type = "quality_data"
                
                # Route the data
                if await router.route_data(data, data_type):
                    flushed_count += 1
                    self.stats["total_removed"] += 1
                else:
                    # Put back in buffer if routing failed
                    self.buffer.appendleft(buffered_item)
                    break  # Stop if routing fails to prevent infinite loop
            
            if flushed_count > 0:
                logger.info(f"🔄 Flushed {flushed_count} items from buffer")
            
            return flushed_count
            
        except Exception as e:
            logger.error(f"❌ Error flushing buffer: {e}")
            return 0
    
    def get_buffer_stats(self) -> Dict[str, int]:
        """Get buffer statistics"""
        return {
            **self.stats,
            "current_size": len(self.buffer),
            "capacity": self.max_size,
            "utilization": len(self.buffer) / self.max_size if self.max_size > 0 else 0
        }


async def main_example():
    """Example usage of Data Router"""
    
    # Create data router
    router = DataRouter()
    
    try:
        # Initialize destinations
        await router.initialize_destinations()
        await router.start_destinations()
        
        print("🔄 Testing data routing with sample data...")
        
        # Generate sample data
        sample_sensor_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "production_line": "Line_A",
            "sensors": {
                "furnace_temperature": 1500.0,
                "furnace_pressure": 15.0,
                "melt_level": 2500.0,
                "forming_belt_speed": 150.0,
                "quality_score": 0.95
            }
        }
        
        # Route sensor data
        success = await router.route_data(sample_sensor_data, "sensor_data")
        print(f"✅ Sensor data routing: {'Success' if success else 'Failed'}")
        
        # Generate sample defect data
        sample_defect_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "production_line": "Line_A",
            "defect_type": "bubble",
            "severity": "MEDIUM",
            "position": {"x": 100.0, "y": 50.0},
            "size_mm": 2.5,
            "confidence": 0.85
        }
        
        # Route defect data
        success = await router.route_data(sample_defect_data, "defect_data")
        print(f"✅ Defect data routing: {'Success' if success else 'Failed'}")
        
        # Show routing statistics
        stats = await router.get_routing_stats()
        print(f"\n📊 Routing Statistics:")
        for destination, dest_stats in stats.items():
            print(f"   {destination}: {dest_stats['messages_routed']} messages routed")
        
        # Show buffer stats if available
        if router.destinations["buffer"]:
            buffer_stats = router.destinations["buffer"].get_buffer_stats()
            print(f"   Buffer: {buffer_stats['current_size']}/{buffer_stats['capacity']} items")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
    finally:
        await router.stop_destinations()


if __name__ == "__main__":
    asyncio.run(main_example())