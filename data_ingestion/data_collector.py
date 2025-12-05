"""
Data Collector for Glass Production System
Centralized data collection from all industrial sources
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque

# Import connector modules
from industrial_connectors.opc_ua_client import OPCUAClient
from industrial_connectors.modbus_driver import ModbusDriver
from industrial_connectors.mik1_camera_stream import MIK1CameraStream
from streaming_pipeline.mqtt_broker import MQTTBrokerConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Centralized data collector orchestrating all industrial data sources"""
    
    def __init__(
        self,
        collection_interval: float = 1.0,  # seconds
        buffer_size: int = 1000,
        data_callback: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.collection_interval = collection_interval
        self.buffer_size = buffer_size
        self.data_callback = data_callback
        self.config = config or {}
        self.running = False
        
        # Data buffers for each source type
        self.data_buffers = {
            "opc_ua": deque(maxlen=buffer_size),
            "modbus": deque(maxlen=buffer_size),
            "mik1_camera": deque(maxlen=buffer_size),
            "mqtt": deque(maxlen=buffer_size)
        }
        
        # Connector instances
        self.connectors = {
            "opc_ua": None,
            "modbus": None,
            "mik1_camera": None,
            "mqtt": None
        }
        
        # Collection statistics
        self.stats = {
            "total_collections": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "data_points_collected": 0,
            "start_time": datetime.utcnow()
        }
        
        # Source status tracking
        self.source_status = defaultdict(lambda: {
            "connected": False,
            "last_update": None,
            "error_count": 0,
            "data_rate": 0.0
        })
    
    async def initialize_connectors(self):
        """Initialize all data source connectors"""
        logger.info("🔧 Initializing data source connectors...")
        
        # Get source configurations
        sources_config = self.config.get("collector", {}).get("sources", {})
        
        # Initialize OPC UA client
        try:
            opc_ua_config = sources_config.get("opc_ua", {})
            server_url = opc_ua_config.get("server_url", "opc.tcp://opcua_server:4840")
            namespace = opc_ua_config.get("namespace", "http://glass.factory/UA/")
            
            self.connectors["opc_ua"] = OPCUAClient(
                server_url=server_url,
                namespace=namespace,
                callback=self._opc_ua_data_handler
            )
            logger.info("✅ OPC UA client initialized")
        except Exception as e:
            logger.warning(f"⚠️ OPC UA client initialization failed: {e}")
        
        # Initialize Modbus driver
        try:
            modbus_config = sources_config.get("modbus", {})
            protocol = modbus_config.get("protocol", "tcp")
            host = modbus_config.get("host", "modbus_server")
            port = modbus_config.get("port", 5020)
            
            self.connectors["modbus"] = ModbusDriver(
                protocol=protocol,
                host=host,
                port=port,
                callback=self._modbus_data_handler
            )
            logger.info("✅ Modbus driver initialized")
        except Exception as e:
            logger.warning(f"⚠️ Modbus driver initialization failed: {e}")
        
        # Initialize MIK-1 camera
        try:
            mik1_config = sources_config.get("mik1_camera", {})
            camera_source = mik1_config.get("camera_source", "0")
            
            self.connectors["mik1_camera"] = MIK1CameraStream(
                camera_source=camera_source,
                callback=self._mik1_data_handler
            )
            logger.info("✅ MIK-1 camera stream initialized")
        except Exception as e:
            logger.warning(f"⚠️ MIK-1 camera initialization failed: {e}")
        
        # Initialize MQTT broker
        try:
            mqtt_config = sources_config.get("mqtt", {})
            broker_host = mqtt_config.get("broker_host", "mosquitto")
            broker_port = mqtt_config.get("broker_port", 1883)
            client_id = mqtt_config.get("client_id", "glass_production_collector")
            
            self.connectors["mqtt"] = MQTTBrokerConnector(
                broker_host=broker_host,
                broker_port=broker_port,
                client_id=client_id,
                callback=self._mqtt_data_handler
            )
            logger.info("✅ MQTT broker connector initialized")
        except Exception as e:
            logger.warning(f"⚠️ MQTT broker initialization failed: {e}")
    
    async def connect_sources(self) -> Dict[str, bool]:
        """Connect to all initialized data sources"""
        connection_results = {}
        
        # Connect OPC UA
        if self.connectors["opc_ua"]:
            try:
                result = await self.connectors["opc_ua"].connect()
                connection_results["opc_ua"] = result
                self.source_status["opc_ua"]["connected"] = result
                if result:
                    logger.info("✅ Connected to OPC UA server")
                else:
                    logger.warning("⚠️ Failed to connect to OPC UA server, using simulation mode")
            except Exception as e:
                logger.error(f"❌ OPC UA connection error: {e}")
                connection_results["opc_ua"] = False
                self.source_status["opc_ua"]["connected"] = False
        
        # Connect Modbus
        if self.connectors["modbus"]:
            try:
                result = await self.connectors["modbus"].connect()
                connection_results["modbus"] = result
                self.source_status["modbus"]["connected"] = result
                if result:
                    logger.info("✅ Connected to Modbus device")
                else:
                    logger.warning("⚠️ Failed to connect to Modbus device, using simulation mode")
            except Exception as e:
                logger.error(f"❌ Modbus connection error: {e}")
                connection_results["modbus"] = False
                self.source_status["modbus"]["connected"] = False
        
        # Connect MIK-1 camera
        if self.connectors["mik1_camera"]:
            try:
                result = await self.connectors["mik1_camera"].connect()
                connection_results["mik1_camera"] = result
                self.source_status["mik1_camera"]["connected"] = result
                if result:
                    logger.info("✅ Connected to MIK-1 camera")
                else:
                    logger.info("✅ Using MIK-1 camera simulator")
            except Exception as e:
                logger.error(f"❌ MIK-1 camera connection error: {e}")
                connection_results["mik1_camera"] = False
                self.source_status["mik1_camera"]["connected"] = False
        
        # Connect MQTT
        if self.connectors["mqtt"]:
            try:
                result = await self.connectors["mqtt"].connect()
                connection_results["mqtt"] = result
                self.source_status["mqtt"]["connected"] = result
                if result:
                    logger.info("✅ Connected to MQTT broker")
                    # Add a small delay to ensure connection is fully established
                    await asyncio.sleep(0.5)
                    # Subscribe to production topics
                    await self.connectors["mqtt"].subscribe_to_production_topics()
                else:
                    logger.warning("⚠️ Failed to connect to MQTT broker, using simulation mode")
            except Exception as e:
                logger.error(f"❌ MQTT connection error: {e}")
                connection_results["mqtt"] = False
                self.source_status["mqtt"]["connected"] = False
        
        # If no sources are connected, we'll generate synthetic data
        connected_count = sum(1 for result in connection_results.values() if result)
        if connected_count == 0:
            logger.info("⚠️ No industrial sources connected, will use synthetic data generation")
        
        return connection_results
    
    async def start_collection(self):
        """Start collecting data from all sources"""
        if not any(self.connectors.values()):
            logger.error("❌ No connectors initialized")
            return
        
        self.running = True
        tasks = []
        
        # Start OPC UA data collection
        if self.connectors["opc_ua"]:
            try:
                # Try subscription first, fall back to polling
                await self.connectors["opc_ua"].subscribe_to_changes(interval=1000)
                # Check if subscription was actually successful
                if (hasattr(self.connectors["opc_ua"], 'running') and 
                    self.connectors["opc_ua"].running):
                    logger.info("🔄 OPC UA subscription started")
                else:
                    logger.warning("⚠️ OPC UA subscription not active, using polling")
                    task = asyncio.create_task(self._opc_ua_polling_task())
                    tasks.append(task)
            except Exception as e:
                logger.warning(f"⚠️ OPC UA subscription failed, using polling: {e}")
                task = asyncio.create_task(self._opc_ua_polling_task())
                tasks.append(task)
        
        # Start Modbus polling
        if self.connectors["modbus"] and self.source_status["modbus"]["connected"]:
            task = asyncio.create_task(self._modbus_polling_task())
            tasks.append(task)
        elif self.connectors["modbus"]:
            logger.warning("⚠️ Modbus not connected, skipping polling task")
        
        # Start MIK-1 camera streaming
        if self.connectors["mik1_camera"]:
            task = asyncio.create_task(self._mik1_streaming_task())
            tasks.append(task)
        
        # Start MQTT listening
        if self.connectors["mqtt"] and self.source_status["mqtt"]["connected"]:
            task = asyncio.create_task(self.connectors["mqtt"].start_listening())
            tasks.append(task)
        elif self.connectors["mqtt"]:
            logger.warning("⚠️ MQTT broker not connected, skipping listener task")
        
        # Start main collection loop
        collection_task = asyncio.create_task(self._collection_loop())
        tasks.append(collection_task)
        
        logger.info("🟢 All data collection tasks started")
        
        # Wait for tasks
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("⏹️ Data collection cancelled")
        except Exception as e:
            logger.error(f"❌ Error in data collection: {e}")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect data from all sources
                await self._collect_all_sources()
                
                # Update statistics
                self.stats["total_collections"] += 1
                
                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in collection loop: {e}")
                self.stats["failed_collections"] += 1
                await asyncio.sleep(self.collection_interval * 2)  # Wait longer on error
    
    async def _collect_all_sources(self):
        """Collect data from all connected sources"""
        try:
            collected_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "sources": {},
                "data_points": 0
            }
            
            # Check if any sources are connected
            connected_sources = sum(1 for status in self.source_status.values() if status["connected"])
            
            if connected_sources > 0:
                # Collect from each connected source
                for source_name, connector in self.connectors.items():
                    if connector and self.source_status[source_name]["connected"]:
                        try:
                            # Different collection methods for different sources
                            if source_name == "opc_ua":
                                # Check if OPC UA client is still valid
                                if hasattr(connector, 'client') and connector.client:
                                    try:
                                        _ = await connector.client.get_namespace_array()
                                        data = await connector.read_sensor_data()
                                        if data and "sensors" in data:
                                            collected_data["sources"][source_name] = data
                                            collected_data["data_points"] += len(data.get("sensors", {}))
                                    except Exception as e:
                                        logger.error(f"❌ OPC UA client error: {e}")
                                        # Try to reconnect
                                        reconnect_result = await connector.connect()
                                        self.source_status[source_name]["connected"] = reconnect_result
                                        if reconnect_result:
                                            data = await connector.read_sensor_data()
                                            if data and "sensors" in data:
                                                collected_data["sources"][source_name] = data
                                                collected_data["data_points"] += len(data.get("sensors", {}))
                                else:
                                    logger.warning("⚠️ OPC UA client not initialized")
                                
                            elif source_name == "modbus":
                                # Check if Modbus client is still connected
                                if hasattr(connector, 'connected') and connector.connected:
                                    data = await connector.read_all_sensors()
                                    if data and "sensors" in data:
                                        collected_data["sources"][source_name] = data
                                        collected_data["data_points"] += len(data.get("sensors", {}))
                                else:
                                    logger.warning("⚠️ Modbus client not connected")
                            
                            # Update source status
                            self.source_status[source_name]["last_update"] = datetime.utcnow()
                            self.source_status[source_name]["data_rate"] = collected_data["data_points"]
                            
                        except Exception as e:
                            logger.error(f"❌ Error collecting from {source_name}: {e}")
                            self.source_status[source_name]["error_count"] += 1
            else:
                # No real sources connected, generate synthetic data
                logger.info("🔄 No real sources connected, generating synthetic data")
                simulator = DataCollectionSimulator()
                synthetic_data = await simulator.generate_sample_data()
                collected_data = synthetic_data
                
                # Update statistics for synthetic data
                self.stats["successful_collections"] += 1
                self.stats["data_points_collected"] += collected_data.get("data_points", 0)
                
                # Call callback if provided
                if self.data_callback:
                    await self.data_callback(collected_data)
                
                logger.debug(f"📊 Generated synthetic data with {collected_data.get('data_points', 0)} data points")
                return
            
            # Add to buffers
            for source_name, data in collected_data["sources"].items():
                if data:
                    self.data_buffers[source_name].append(data)
            
            # Update statistics
            self.stats["successful_collections"] += 1
            self.stats["data_points_collected"] += collected_data["data_points"]
            
            # Call callback if provided
            if self.data_callback:
                await self.data_callback(collected_data)
            
            logger.debug(f"📊 Collected {collected_data['data_points']} data points from {len(collected_data['sources'])} sources")
            
        except Exception as e:
            logger.error(f"❌ Error in data collection: {e}")
            self.stats["failed_collections"] += 1
    
    async def _opc_ua_data_handler(self, data: Dict[str, Any]):
        """Handle incoming OPC UA data"""
        try:
            self.data_buffers["opc_ua"].append(data)
            self.source_status["opc_ua"]["last_update"] = datetime.utcnow()
            logger.debug(f"📥 OPC UA data received: {len(data.get('sensors', {}))} sensors")
        except Exception as e:
            logger.error(f"❌ Error handling OPC UA data: {e}")
    
    async def _modbus_data_handler(self, data: Dict[str, Any]):
        """Handle incoming Modbus data"""
        try:
            self.data_buffers["modbus"].append(data)
            self.source_status["modbus"]["last_update"] = datetime.utcnow()
            logger.debug(f"📥 Modbus data received: {len(data.get('sensors', {}))} registers")
        except Exception as e:
            logger.error(f"❌ Error handling Modbus data: {e}")
    
    async def _mik1_data_handler(self, data: Dict[str, Any]):
        """Handle incoming MIK-1 camera data"""
        try:
            self.data_buffers["mik1_camera"].append(data)
            self.source_status["mik1_camera"]["last_update"] = datetime.utcnow()
            logger.debug(f"📥 MIK-1 data received: {data.get('defects_detected', 0)} defects")
        except Exception as e:
            logger.error(f"❌ Error handling MIK-1 data: {e}")
    
    async def _mqtt_data_handler(self, data: Dict[str, Any]):
        """Handle incoming MQTT data"""
        try:
            self.data_buffers["mqtt"].append(data)
            self.source_status["mqtt"]["last_update"] = datetime.utcnow()
            logger.debug(f"📥 MQTT data received from {data.get('topic', 'unknown')}")
        except Exception as e:
            logger.error(f"❌ Error handling MQTT data: {e}")
    
    async def _opc_ua_polling_task(self):
        """Task to poll OPC UA data periodically"""
        while self.running:
            try:
                if self.connectors["opc_ua"]:
                    data = await self.connectors["opc_ua"].read_sensor_data()
                    await self._opc_ua_data_handler(data)
                await asyncio.sleep(5)  # Poll every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ OPC UA polling error: {e}")
                await asyncio.sleep(10)
    
    async def _modbus_polling_task(self):
        """Task to poll Modbus data periodically"""
        while self.running:
            try:
                if self.connectors["modbus"]:
                    data = await self.connectors["modbus"].read_all_sensors()
                    if data:
                        await self._modbus_data_handler(data)
                await asyncio.sleep(10)  # Poll every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Modbus polling error: {e}")
                await asyncio.sleep(15)
    
    async def _mik1_streaming_task(self):
        """Task to stream MIK-1 camera data"""
        try:
            if self.connectors["mik1_camera"]:
                await self.connectors["mik1_camera"].start_streaming(process_frames=True)
        except Exception as e:
            logger.error(f"❌ MIK-1 streaming error: {e}")
    
    async def get_buffered_data(self, source: str = "all", limit: int = 100) -> List[Dict[str, Any]]:
        """Get buffered data from specified source(s)"""
        try:
            if source == "all":
                # Return data from all sources
                all_data = []
                for buffer in self.data_buffers.values():
                    all_data.extend(list(buffer)[-limit:])
                return sorted(all_data, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]
            elif source in self.data_buffers:
                # Return data from specific source
                return list(self.data_buffers[source])[-limit:]
            else:
                logger.warning(f"⚠️ Unknown source: {source}")
                return []
        except Exception as e:
            logger.error(f"❌ Error retrieving buffered data: {e}")
            return []
    
    async def get_source_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all data sources"""
        return dict(self.source_status)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        stats = self.stats.copy()
        stats["uptime"] = (datetime.utcnow() - stats["start_time"]).total_seconds()
        stats["success_rate"] = (
            stats["successful_collections"] / max(1, stats["total_collections"])
            if stats["total_collections"] > 0 else 0
        )
        return stats
    
    async def stop_collection(self):
        """Stop all data collection tasks"""
        self.running = False
        logger.info("⏹️ Stopping data collection...")
        
        # Stop individual connectors
        for source_name, connector in self.connectors.items():
            if connector:
                try:
                    if source_name == "opc_ua":
                        await connector.disconnect()
                    elif source_name == "modbus":
                        await connector.disconnect()
                    elif source_name == "mik1_camera":
                        await connector.disconnect()
                    elif source_name == "mqtt":
                        await connector.disconnect()
                except Exception as e:
                    logger.error(f"❌ Error stopping {source_name}: {e}")
        
        logger.info("✅ Data collection stopped")


class DataCollectionSimulator:
    """Simulator for data collection for testing purposes"""
    
    def __init__(self):
        self.base_timestamp = datetime.utcnow()
        import random
        self.random = random
    
    def generate_sensor_data(self) -> Dict[str, Any]:
        """Generate synthetic sensor data in the correct format"""
        current_time = datetime.utcnow()
        
        # Furnace data (critical for defects)
        furnace_temp = 1500 + self.random.uniform(-50, 50)  # 1450-1550°C
        furnace_pressure = 15 + self.random.uniform(-3, 3)   # kPa
        melt_level = 2500 + self.random.uniform(-200, 200)   # mm
        
        # Forming data
        mold_temp = 320 + self.random.uniform(-30, 30)       # °C
        belt_speed = 150 + self.random.uniform(-20, 20)      # m/min
        forming_pressure = 50 + self.random.uniform(-10, 10) # bar
        
        # Annealing data
        annealing_temp = 600 + self.random.uniform(-50, 50)  # °C
        
        # Process data
        batch_flow = 2000 + self.random.uniform(-300, 300)   # kg/h
        
        return {
            "timestamp": current_time.isoformat() + "Z",
            "production_line": "Line_A",
            "sensors": {
                "furnace": {
                    "temperature": round(furnace_temp, 2),
                    "pressure": round(furnace_pressure, 2),
                    "melt_level": round(melt_level, 2),
                    "o2_percent": round(5.0 + self.random.uniform(-0.5, 0.5), 2),
                    "co2_percent": round(10.0 + self.random.uniform(-1.0, 1.0), 2)
                },
                "forming": {
                    "mold_temperature": round(mold_temp, 2),
                    "belt_speed": round(belt_speed, 2),
                    "pressure": round(forming_pressure, 2)
                },
                "annealing": {
                    "temperature": round(annealing_temp, 2)
                },
                "process": {
                    "batch_flow": round(batch_flow, 2)
                }
            }
        }
    
    def generate_defect_data(self) -> Dict[str, Any]:
        """Generate synthetic defect data in the correct format"""
        current_time = datetime.utcnow()
        defect_types = ["crack", "bubble", "chip", "cloudiness", "deformation", 
                       "inclusion", "stress", "surface_defect"]
        
        # Generate realistic defect probabilities based on sensor data
        defect_count = self.random.randint(0, 5)
        defect_list = self.random.sample(defect_types, k=min(defect_count, len(defect_types)))
        
        return {
            "timestamp": current_time.isoformat() + "Z",
            "production_line": "Line_A",
            "defects": [
                {
                    "type": defect_type,
                    "severity": self.random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                    "position": {
                        "x": round(self.random.uniform(0, 1000), 2),
                        "y": round(self.random.uniform(0, 500), 2)
                    },
                    "size_mm": round(self.random.uniform(0.1, 10.0), 2),
                    "confidence": round(self.random.uniform(0.7, 0.99), 4)
                } for defect_type in defect_list
            ]
        }
    
    async def generate_sample_data(self) -> Dict[str, Any]:
        """Generate sample industrial data"""
        # Return sensor data in the format expected by the data collector
        sensor_data = self.generate_sensor_data()
        defect_data = self.generate_defect_data()
        
        return {
            "timestamp": sensor_data["timestamp"],
            "sources": {
                "opc_ua": sensor_data,
                "mik1_camera": defect_data
            },
            "data_points": len(sensor_data.get("sensors", {}))
        }


async def main_example():
    """Example usage of Data Collector"""
    
    async def data_callback(collected_data):
        """Callback for collected data"""
        print(f"\n📊 Data Collection at {collected_data['timestamp']}")
        print(f"   Sources: {list(collected_data['sources'].keys())}")
        print(f"   Data points: {collected_data['data_points']}")
    
    # Create data collector
    collector = DataCollector(
        collection_interval=2.0,
        data_callback=data_callback
    )
    
    try:
        # Initialize and connect sources
        await collector.initialize_connectors()
        connection_results = await collector.connect_sources()
        
        print("🔌 Connection Results:")
        for source, result in connection_results.items():
            status = "✅ Connected" if result else "❌ Failed"
            print(f"   {source}: {status}")
        
        if any(connection_results.values()):
            print("\n🔄 Starting data collection (will run for 30 seconds)...")
            
            # Start collection
            collection_task = asyncio.create_task(collector.start_collection())
            
            # Let it run for 30 seconds
            await asyncio.sleep(30)
            
            # Stop collection
            await collector.stop_collection()
            collection_task.cancel()
            
            # Show final statistics
            stats = await collector.get_collection_stats()
            print(f"\n📈 Collection Statistics:")
            print(f"   Total collections: {stats['total_collections']}")
            print(f"   Successful: {stats['successful_collections']}")
            print(f"   Failed: {stats['failed_collections']}")
            print(f"   Data points collected: {stats['data_points_collected']}")
            print(f"   Success rate: {stats['success_rate']:.2%}")
            
        else:
            print("❌ No sources could be connected")
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        await collector.stop_collection()
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        await collector.stop_collection()


if __name__ == "__main__":
    import numpy as np
    asyncio.run(main_example())