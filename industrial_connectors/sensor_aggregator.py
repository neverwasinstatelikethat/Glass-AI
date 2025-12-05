"""
Sensor Aggregator for Industrial Data Fusion
Combines data from OPC UA, Modbus, MIK-1 Camera, and other sources
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict, deque

# Local imports
from industrial_connectors.opc_ua_client import OPCUAClient
from industrial_connectors.modbus_driver import ModbusDriver
from industrial_connectors.mik1_camera_stream import MIK1CameraStream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SensorAggregator:
    """Aggregates data from multiple industrial sensor sources"""
    
    def __init__(
        self,
        aggregation_window_seconds: int = 60,
        max_buffer_size: int = 1000,
        callback: Optional[Callable] = None
    ):
        self.aggregation_window = timedelta(seconds=aggregation_window_seconds)
        self.max_buffer_size = max_buffer_size
        self.callback = callback
        self.running = False
        
        # Data buffers for each source
        self.data_buffers = {
            "opc_ua": deque(maxlen=max_buffer_size),
            "modbus": deque(maxlen=max_buffer_size),
            "mik1_camera": deque(maxlen=max_buffer_size),
            "mqtt": deque(maxlen=max_buffer_size)
        }
        
        # Timestamp tracking
        self.last_aggregation_time = datetime.utcnow()
        self.source_timestamps = defaultdict(datetime)
        
        # Sensor clients
        self.opc_ua_client: Optional[OPCUAClient] = None
        self.modbus_driver: Optional[ModbusDriver] = None
        self.mik1_camera: Optional[MIK1CameraStream] = None
        
        # Aggregation statistics
        self.stats = {
            "total_readings": 0,
            "aggregated_samples": 0,
            "last_aggregation": None
        }
    
    async def initialize_sources(self):
        """Initialize all sensor sources"""
        logger.info("🔧 Initializing sensor sources...")
        
        # Initialize OPC UA client
        try:
            self.opc_ua_client = OPCUAClient(
                server_url="opc.tcp://localhost:4840",
                namespace="http://glass.factory/UA/",
                callback=self._opc_ua_data_handler
            )
            logger.info("✅ OPC UA client initialized")
        except Exception as e:
            logger.warning(f"⚠️ OPC UA client initialization failed: {e}")
            self.opc_ua_client = None
        
        # Initialize Modbus driver
        try:
            self.modbus_driver = ModbusDriver(
                protocol="tcp",
                host="localhost",
                port=502,
                callback=self._modbus_data_handler
            )
            logger.info("✅ Modbus driver initialized")
        except Exception as e:
            logger.warning(f"⚠️ Modbus driver initialization failed: {e}")
            self.modbus_driver = None
        
        # Initialize MIK-1 camera
        try:
            self.mik1_camera = MIK1CameraStream(
                camera_source="0",
                callback=self._mik1_data_handler
            )
            logger.info("✅ MIK-1 camera stream initialized")
        except Exception as e:
            logger.warning(f"⚠️ MIK-1 camera initialization failed: {e}")
            self.mik1_camera = None
    
    async def connect_sources(self) -> bool:
        """Connect to all initialized sensor sources"""
        success_count = 0
        total_sources = 0
        
        # Connect OPC UA
        if self.opc_ua_client:
            total_sources += 1
            try:
                if await self.opc_ua_client.connect():
                    success_count += 1
                    logger.info("✅ Connected to OPC UA server")
                else:
                    logger.error("❌ Failed to connect to OPC UA server")
            except Exception as e:
                logger.error(f"❌ OPC UA connection error: {e}")
        
        # Connect Modbus
        if self.modbus_driver:
            total_sources += 1
            try:
                if await self.modbus_driver.connect():
                    success_count += 1
                    logger.info("✅ Connected to Modbus device")
                else:
                    logger.error("❌ Failed to connect to Modbus device")
            except Exception as e:
                logger.error(f"❌ Modbus connection error: {e}")
        
        # Connect MIK-1 camera
        if self.mik1_camera:
            total_sources += 1
            try:
                if await self.mik1_camera.connect():
                    success_count += 1
                    logger.info("✅ Connected to MIK-1 camera")
                else:
                    logger.error("❌ Failed to connect to MIK-1 camera")
            except Exception as e:
                logger.error(f"❌ MIK-1 camera connection error: {e}")
        
        logger.info(f"🔌 Connection summary: {success_count}/{total_sources} sources connected")
        return success_count > 0
    
    async def start_data_collection(self):
        """Start collecting data from all sources"""
        if not self.opc_ua_client and not self.modbus_driver and not self.mik1_camera:
            logger.error("❌ No sensor sources initialized")
            return
        
        self.running = True
        tasks = []
        
        # Start OPC UA subscription or polling
        if self.opc_ua_client:
            try:
                # Try subscription first, fall back to polling
                await self.opc_ua_client.subscribe_to_changes(interval=1000)
                logger.info("🔄 OPC UA subscription started")
            except Exception as e:
                logger.warning(f"⚠️ OPC UA subscription failed, using polling: {e}")
                task = asyncio.create_task(self._opc_ua_polling_task())
                tasks.append(task)
        
        # Start Modbus polling
        if self.modbus_driver:
            task = asyncio.create_task(self._modbus_polling_task())
            tasks.append(task)
        
        # Start MIK-1 camera streaming
        if self.mik1_camera:
            task = asyncio.create_task(self._mik1_streaming_task())
            tasks.append(task)
        
        # Start aggregation timer
        aggregation_task = asyncio.create_task(self._aggregation_timer())
        tasks.append(aggregation_task)
        
        logger.info("🟢 All data collection tasks started")
        
        # Wait for tasks (they should run indefinitely)
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("⏹️ Data collection cancelled")
        except Exception as e:
            logger.error(f"❌ Error in data collection: {e}")
    
    async def _opc_ua_data_handler(self, data: Dict[str, Any]):
        """Handle incoming OPC UA data"""
        try:
            timestamp = datetime.fromisoformat(data["timestamp"])
            self.data_buffers["opc_ua"].append({
                "timestamp": timestamp,
                "data": data,
                "source": "opc_ua"
            })
            self.source_timestamps["opc_ua"] = timestamp
            self.stats["total_readings"] += 1
            logger.debug(f"📥 OPC UA data received: {len(data.get('sensors', {}))} sensors")
        except Exception as e:
            logger.error(f"❌ Error handling OPC UA data: {e}")
    
    async def _modbus_data_handler(self, data: Dict[str, Any]):
        """Handle incoming Modbus data"""
        try:
            timestamp = datetime.fromisoformat(data["timestamp"])
            self.data_buffers["modbus"].append({
                "timestamp": timestamp,
                "data": data,
                "source": "modbus"
            })
            self.source_timestamps["modbus"] = timestamp
            self.stats["total_readings"] += 1
            logger.debug(f"📥 Modbus data received: {len(data.get('sensors', {}))} registers")
        except Exception as e:
            logger.error(f"❌ Error handling Modbus data: {e}")
    
    async def _mik1_data_handler(self, data: Dict[str, Any]):
        """Handle incoming MIK-1 camera data"""
        try:
            timestamp = datetime.fromisoformat(data["timestamp"])
            self.data_buffers["mik1_camera"].append({
                "timestamp": timestamp,
                "data": data,
                "source": "mik1_camera"
            })
            self.source_timestamps["mik1_camera"] = timestamp
            self.stats["total_readings"] += 1
            logger.debug(f"📥 MIK-1 data received: {data.get('defects_detected', 0)} defects")
        except Exception as e:
            logger.error(f"❌ Error handling MIK-1 data: {e}")
    
    async def _opc_ua_polling_task(self):
        """Task to poll OPC UA data periodically"""
        while self.running:
            try:
                if self.opc_ua_client:
                    data = await self.opc_ua_client.read_sensor_data()
                    await self._opc_ua_data_handler(data)
                await asyncio.sleep(5)  # Poll every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ OPC UA polling error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _modbus_polling_task(self):
        """Task to poll Modbus data periodically"""
        while self.running:
            try:
                if self.modbus_driver:
                    data = await self.modbus_driver.read_all_sensors()
                    if data:
                        await self._modbus_data_handler(data)
                await asyncio.sleep(10)  # Poll every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Modbus polling error: {e}")
                await asyncio.sleep(15)  # Wait longer on error
    
    async def _mik1_streaming_task(self):
        """Task to stream MIK-1 camera data"""
        try:
            if self.mik1_camera:
                await self.mik1_camera.start_streaming(process_frames=True)
        except Exception as e:
            logger.error(f"❌ MIK-1 streaming error: {e}")
    
    async def _aggregation_timer(self):
        """Timer task to trigger periodic aggregation"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                if current_time - self.last_aggregation_time >= self.aggregation_window:
                    await self.aggregate_data()
                    self.last_aggregation_time = current_time
                
                await asyncio.sleep(1)  # Check every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Aggregation timer error: {e}")
                await asyncio.sleep(5)
    
    async def aggregate_data(self) -> Dict[str, Any]:
        """Aggregate data from all sources within the time window"""
        try:
            current_time = datetime.utcnow()
            window_start = current_time - self.aggregation_window
            
            # Collect data from all buffers within the window
            aggregated_data = {
                "timestamp": current_time.isoformat(),
                "window_start": window_start.isoformat(),
                "sources": {},
                "sensors": {},
                "quality": {},
                "statistics": {}
            }
            
            # Process each data source
            for source_name, buffer in self.data_buffers.items():
                if not buffer:
                    continue
                
                # Filter data within time window
                window_data = [entry for entry in buffer 
                              if entry["timestamp"] >= window_start]
                
                if not window_data:
                    continue
                
                aggregated_data["sources"][source_name] = {
                    "count": len(window_data),
                    "latest_timestamp": max(entry["timestamp"] for entry in window_data).isoformat()
                }
                
                # Aggregate based on source type
                if source_name == "opc_ua":
                    await self._aggregate_opc_ua_data(window_data, aggregated_data)
                elif source_name == "modbus":
                    await self._aggregate_modbus_data(window_data, aggregated_data)
                elif source_name == "mik1_camera":
                    await self._aggregate_mik1_data(window_data, aggregated_data)
            
            # Add statistics
            aggregated_data["statistics"] = {
                "total_sources": len(aggregated_data["sources"]),
                "total_readings": self.stats["total_readings"],
                "aggregation_window_seconds": self.aggregation_window.total_seconds()
            }
            
            # Update stats
            self.stats["aggregated_samples"] += 1
            self.stats["last_aggregation"] = current_time
            
            # Call callback if provided
            if self.callback:
                await self.callback(aggregated_data)
            
            logger.info(f"📊 Aggregated data from {len(aggregated_data['sources'])} sources")
            return aggregated_data
            
        except Exception as e:
            logger.error(f"❌ Error aggregating data: {e}")
            return {}
    
    async def _aggregate_opc_ua_data(self, window_data: List[Dict], aggregated_data: Dict):
        """Aggregate OPC UA sensor data"""
        try:
            # Collect all sensor readings
            all_sensor_readings = defaultdict(list)
            
            for entry in window_data:
                sensor_data = entry["data"].get("sensors", {})
                for sensor_id, reading in sensor_data.items():
                    if reading.get("value") is not None:
                        all_sensor_readings[sensor_id].append(reading["value"])
            
            # Calculate statistics for each sensor
            for sensor_id, readings in all_sensor_readings.items():
                if readings:
                    numeric_readings = [r for r in readings if isinstance(r, (int, float))]
                    if numeric_readings:
                        aggregated_data["sensors"][f"opc_ua_{sensor_id}"] = {
                            "mean": float(np.mean(numeric_readings)),
                            "std": float(np.std(numeric_readings)),
                            "min": float(np.min(numeric_readings)),
                            "max": float(np.max(numeric_readings)),
                            "count": len(numeric_readings),
                            "latest": float(numeric_readings[-1]) if numeric_readings else None
                        }
        except Exception as e:
            logger.error(f"❌ Error aggregating OPC UA data: {e}")
    
    async def _aggregate_modbus_data(self, window_data: List[Dict], aggregated_data: Dict):
        """Aggregate Modbus register data"""
        try:
            # Collect all register readings
            all_register_readings = defaultdict(list)
            
            for entry in window_data:
                register_data = entry["data"].get("sensors", {})
                for register_name, value in register_data.items():
                    if value is not None:
                        all_register_readings[register_name].append(value)
            
            # Calculate statistics for each register
            for register_name, readings in all_register_readings.items():
                if readings:
                    numeric_readings = [r for r in readings if isinstance(r, (int, float))]
                    if numeric_readings:
                        aggregated_data["sensors"][f"modbus_{register_name}"] = {
                            "mean": float(np.mean(numeric_readings)),
                            "std": float(np.std(numeric_readings)),
                            "min": float(np.min(numeric_readings)),
                            "max": float(np.max(numeric_readings)),
                            "count": len(numeric_readings),
                            "latest": float(numeric_readings[-1]) if numeric_readings else None
                        }
        except Exception as e:
            logger.error(f"❌ Error aggregating Modbus data: {e}")
    
    async def _aggregate_mik1_data(self, window_data: List[Dict], aggregated_data: Dict):
        """Aggregate MIK-1 camera data"""
        try:
            total_defects = 0
            defect_details = []
            image_stats = []
            
            for entry in window_data:
                data = entry["data"]
                total_defects += data.get("defects_detected", 0)
                defect_details.extend(data.get("defects", []))
                
                # Collect image statistics
                img_stats = data.get("image_stats", {})
                if img_stats:
                    image_stats.append(img_stats)
            
            # Aggregate quality metrics
            aggregated_data["quality"] = {
                "total_defects": total_defects,
                "defect_rate": total_defects / max(1, len(window_data)),  # Defects per frame
                "defect_details": defect_details[-10:] if defect_details else [],  # Last 10 defects
                "total_frames": len(window_data)
            }
            
            # Aggregate image statistics
            if image_stats:
                brightness_values = [stat.get("mean_brightness", 0) for stat in image_stats]
                if brightness_values:
                    aggregated_data["quality"]["image_brightness"] = {
                        "mean": float(np.mean(brightness_values)),
                        "std": float(np.std(brightness_values)),
                        "min": float(np.min(brightness_values)),
                        "max": float(np.max(brightness_values))
                    }
        except Exception as e:
            logger.error(f"❌ Error aggregating MIK-1 data: {e}")
    
    async def get_current_snapshot(self) -> Dict[str, Any]:
        """Get current snapshot of all sensor data"""
        try:
            snapshot = {
                "timestamp": datetime.utcnow().isoformat(),
                "sources": {},
                "sensors": {},
                "quality": {}
            }
            
            # Get latest data from each source
            for source_name, buffer in self.data_buffers.items():
                if buffer:
                    latest_entry = buffer[-1]
                    snapshot["sources"][source_name] = {
                        "timestamp": latest_entry["timestamp"].isoformat(),
                        "data": latest_entry["data"]
                    }
            
            return snapshot
        except Exception as e:
            logger.error(f"❌ Error getting snapshot: {e}")
            return {}
    
    async def stop_collection(self):
        """Stop all data collection tasks"""
        self.running = False
        logger.info("⏹️ Stopping sensor data collection...")
        
        # Stop individual sources
        if self.opc_ua_client:
            try:
                await self.opc_ua_client.disconnect()
            except Exception as e:
                logger.error(f"❌ Error stopping OPC UA client: {e}")
        
        if self.modbus_driver:
            try:
                await self.modbus_driver.disconnect()
            except Exception as e:
                logger.error(f"❌ Error stopping Modbus driver: {e}")
        
        if self.mik1_camera:
            try:
                await self.mik1_camera.disconnect()
            except Exception as e:
                logger.error(f"❌ Error stopping MIK-1 camera: {e}")
        
        logger.info("✅ Sensor data collection stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        return {
            **self.stats,
            "buffer_sizes": {name: len(buffer) for name, buffer in self.data_buffers.items()},
            "source_timestamps": {name: ts.isoformat() for name, ts in self.source_timestamps.items()}
        }


class DataValidator:
    """Validates and cleans sensor data"""
    
    def __init__(self):
        # Define valid ranges for different sensors
        self.valid_ranges = {
            "furnace_temperature": {"min": 1000, "max": 1800, "unit": "°C"},
            "furnace_pressure": {"min": 0, "max": 50, "unit": "bar"},
            "furnace_melt_level": {"min": 1000, "max": 4000, "unit": "mm"},
            "forming_belt_speed": {"min": 50, "max": 300, "unit": "m/min"},
            "forming_mold_temp": {"min": 200, "max": 500, "unit": "°C"},
            "quality_score": {"min": 0, "max": 1, "unit": "ratio"},
            "defect_count": {"min": 0, "max": 100, "unit": "count"}
        }
    
    def validate_sensor_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sensor data against known ranges"""
        validated_data = {}
        validation_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "validated_fields": [],
            "out_of_range_fields": [],
            "missing_fields": []
        }
        
        for field_name, value in data.items():
            if field_name in self.valid_ranges:
                range_info = self.valid_ranges[field_name]
                if range_info["min"] <= value <= range_info["max"]:
                    validated_data[field_name] = value
                    validation_report["validated_fields"].append(field_name)
                else:
                    validation_report["out_of_range_fields"].append({
                        "field": field_name,
                        "value": value,
                        "expected_range": f"{range_info['min']}-{range_info['max']} {range_info['unit']}"
                    })
                    # Use last known good value or median of valid range
                    validated_data[field_name] = np.clip(value, range_info["min"], range_info["max"])
            else:
                validated_data[field_name] = value
        
        # Check for missing critical fields
        critical_fields = ["furnace_temperature", "forming_belt_speed", "quality_score"]
        for field in critical_fields:
            if field not in data:
                validation_report["missing_fields"].append(field)
        
        validated_data["_validation_report"] = validation_report
        return validated_data
    
    def interpolate_missing_data(self, data_series: List[Dict], field_name: str) -> List[Dict]:
        """Interpolate missing values in a time series"""
        if not data_series:
            return data_series
        
        # Extract timestamps and values
        timestamps = []
        values = []
        valid_indices = []
        
        for i, data_point in enumerate(data_series):
            value = data_point.get("sensors", {}).get(field_name)
            if value is not None and not np.isnan(value):
                timestamps.append(datetime.fromisoformat(data_point["timestamp"]))
                values.append(value)
                valid_indices.append(i)
        
        if len(valid_indices) < 2:
            return data_series  # Not enough data to interpolate
        
        # Perform linear interpolation
        for i in range(len(data_series)):
            if i not in valid_indices:
                # Find nearest valid points for interpolation
                prev_idx = None
                next_idx = None
                
                for idx in reversed(valid_indices):
                    if idx < i:
                        prev_idx = idx
                        break
                
                for idx in valid_indices:
                    if idx > i:
                        next_idx = idx
                        break
                
                if prev_idx is not None and next_idx is not None:
                    # Linear interpolation
                    prev_time = datetime.fromisoformat(data_series[prev_idx]["timestamp"])
                    next_time = datetime.fromisoformat(data_series[next_idx]["timestamp"])
                    current_time = datetime.fromisoformat(data_series[i]["timestamp"])
                    
                    time_ratio = (current_time - prev_time).total_seconds() / (next_time - prev_time).total_seconds()
                    interpolated_value = values[valid_indices.index(prev_idx)] + \
                                       time_ratio * (values[valid_indices.index(next_idx)] - values[valid_indices.index(prev_idx)])
                    
                    # Update the data point
                    if "sensors" not in data_series[i]:
                        data_series[i]["sensors"] = {}
                    data_series[i]["sensors"][field_name] = interpolated_value
        
        return data_series


async def main_example():
    """Example usage of Sensor Aggregator"""
    
    async def aggregation_callback(data):
        """Callback for aggregated data"""
        print(f"\n📊 Aggregated data at {data['timestamp']}")
        print(f"   Sources: {list(data['sources'].keys())}")
        print(f"   Sensors: {len(data['sensors'])} aggregated readings")
        if data['quality']:
            print(f"   Quality: {data['quality'].get('total_defects', 0)} defects in window")
    
    # Create aggregator
    aggregator = SensorAggregator(
        aggregation_window_seconds=30,
        callback=aggregation_callback
    )
    
    try:
        # Initialize and connect sources
        await aggregator.initialize_sources()
        connected = await aggregator.connect_sources()
        
        if connected:
            print("✅ Sensor aggregator initialized with connected sources")
            print("🔄 Starting data collection (will run for 2 minutes)...")
            
            # Start collection for a limited time for demo
            collection_task = asyncio.create_task(aggregator.start_data_collection())
            
            # Let it run for 2 minutes
            await asyncio.sleep(120)
            
            # Stop collection
            await aggregator.stop_collection()
            collection_task.cancel()
            
            # Show final stats
            stats = aggregator.get_stats()
            print(f"\n📈 Final Statistics:")
            print(f"   Total readings: {stats['total_readings']}")
            print(f"   Aggregated samples: {stats['aggregated_samples']}")
            print(f"   Buffer sizes: {stats['buffer_sizes']}")
        else:
            print("❌ No sources could be connected")
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        await aggregator.stop_collection()
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        await aggregator.stop_collection()


if __name__ == "__main__":
    asyncio.run(main_example())