"""
InfluxDB Client for Glass Production System
Handles time-series data storage for sensor readings
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlassInfluxDBClient:
    """InfluxDB client for storing time-series sensor data"""
    
    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        org: Optional[str] = None,
        bucket: Optional[str] = None
    ):
        self.url = url or os.getenv("INFLUXDB_URL", "http://localhost:8086")
        self.token = token or os.getenv("INFLUXDB_TOKEN", "my-super-secret-auth-token")
        self.org = org or os.getenv("INFLUXDB_ORG", "glass_factory")
        self.bucket = bucket or os.getenv("INFLUXDB_BUCKET", "sensors")
        
        # Initialize client
        self.client = None
        self.write_api = None
        self.query_api = None
        self.connected = False
        
        # Task requirements - sensor specifications
        self.sensor_specs = {
            "furnace_temperature": {"unit": "°C", "range": (1200, 1700), "critical_range": (1400, 1600)},
            "furnace_pressure": {"unit": "кПа", "range": (0, 50)},
            "melt_level": {"unit": "мм", "range": (0, 5000)},
            "forming_temperature": {"unit": "°C", "range": (20, 600)},
            "forming_pressure": {"unit": "МПа", "range": (0, 100)},
            "forming_speed": {"unit": "м/мин", "range": (0, 200)},
            "annealing_temperature": {"unit": "°C", "range": (20, 1200)},
            "batch_feed_rate": {"unit": "кг/ч", "range": (0, 5000)}
        }
        
    async def connect(self):
        """Establish connection to InfluxDB"""
        try:
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            self.connected = True
            logger.info("✅ Connected to InfluxDB")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to InfluxDB (working in simulation mode): {e}")
            self.connected = False
            return False
    
    async def write_sensor_data(self, data: Dict[str, Any]) -> bool:
        """Write sensor data to InfluxDB"""
        # If not connected, work in simulation mode
        if not self.connected:
            logger.debug("⏭️ InfluxDB not connected, working in simulation mode")
            return True
            
        try:
            if not self.client or not self.write_api:
                logger.warning("⚠️ InfluxDB client not initialized")
                return False
            
            # Extract data fields
            timestamp = data.get("timestamp", datetime.utcnow().isoformat())
            production_line = data.get("production_line", "Line_A")
            sensors = data.get("sensors", {})
            
            # Validate sensor data against task requirements
            validated_sensors = self._validate_sensor_data(sensors)
            
            # Create data point
            point = Point("sensor_readings") \
                .tag("production_line", production_line) \
                .time(timestamp)
            
            # Add sensor values
            for sensor_name, sensor_data in validated_sensors.items():
                if isinstance(sensor_data, dict):
                    value = sensor_data.get("value")
                    status = sensor_data.get("status", "UNKNOWN")
                    point = point.tag(f"{sensor_name}_status", status)
                else:
                    value = sensor_data
                
                if value is not None:
                    point = point.field(sensor_name, float(value))
            
            # Write data
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            logger.debug(f"✅ Wrote sensor data to InfluxDB: {production_line}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error writing sensor data to InfluxDB: {e}")
            return False
    
    def _validate_sensor_data(self, sensors: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sensor data against task specifications"""
        validated = {}
        for sensor_name, sensor_data in sensors.items():
            # Check if sensor is in our specifications
            if sensor_name in self.sensor_specs:
                spec = self.sensor_specs[sensor_name]
                
                # Extract value
                if isinstance(sensor_data, dict):
                    value = sensor_data.get("value")
                else:
                    value = sensor_data
                
                # Validate range if value is numeric
                if isinstance(value, (int, float)):
                    min_val, max_val = spec["range"]
                    if not (min_val <= value <= max_val):
                        logger.warning(f"⚠️ Sensor {sensor_name} value {value} out of range ({min_val}-{max_val})")
                
                validated[sensor_name] = sensor_data
            else:
                # Log unknown sensors but still include them
                logger.debug(f"ℹ️ Unknown sensor: {sensor_name}")
                validated[sensor_name] = sensor_data
        
        return validated
    
    async def write_defect_data(self, data: Dict[str, Any]) -> bool:
        """Write defect data to InfluxDB"""
        # If not connected, work in simulation mode
        if not self.connected:
            logger.debug("⏭️ InfluxDB not connected, working in simulation mode")
            return True
            
        try:
            if not self.client or not self.write_api:
                logger.warning("⚠️ InfluxDB client not initialized")
                return False
            
            # Extract data fields
            timestamp = data.get("timestamp", datetime.utcnow().isoformat())
            production_line = data.get("production_line", "Line_A")
            defect_type = data.get("defect_type", "unknown")
            severity = data.get("severity", "LOW")
            size_mm = data.get("size_mm", 0.0)
            confidence = data.get("confidence", 0.0)
            
            # Validate defect data
            if defect_type not in ["bubble", "crack", "scratch", "stain", "deformation", "chip", "cloudiness"]:
                logger.warning(f"⚠️ Unknown defect type: {defect_type}")
            
            # Create data point
            point = Point("defects") \
                .tag("production_line", production_line) \
                .tag("defect_type", defect_type) \
                .tag("severity", severity) \
                .time(timestamp) \
                .field("size_mm", float(size_mm)) \
                .field("confidence", float(confidence))
            
            # Add position if available
            position = data.get("position", {})
            if position:
                point = point.field("position_x", float(position.get("x", 0.0))) \
                           .field("position_y", float(position.get("y", 0.0)))
            
            # Write data
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)
            logger.debug(f"✅ Wrote defect data to InfluxDB: {defect_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error writing defect data to InfluxDB: {e}")
            return False
    
    async def query_sensor_data(self, production_line: str, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Query recent sensor data from InfluxDB"""
        # If not connected, return empty list
        if not self.connected:
            logger.debug("⏭️ InfluxDB not connected, returning empty results")
            return []
            
        try:
            if not self.client or not self.query_api:
                logger.warning("⚠️ InfluxDB client not initialized")
                return []
            
            # Flux query for sensor data
            flux_query = f'''
                from(bucket: "{self.bucket}")
                  |> range(start: -{hours_back}h)
                  |> filter(fn: (r) => r["_measurement"] == "sensor_readings")
                  |> filter(fn: (r) => r["production_line"] == "{production_line}")
                  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''
            
            # Execute query
            tables = self.query_api.query(flux_query, org=self.org)
            
            # Process results
            results = []
            for table in tables:
                for record in table.records:
                    results.append(dict(record.values))
            
            logger.debug(f"✅ Queried {len(results)} sensor records from InfluxDB")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error querying sensor data from InfluxDB: {e}")
            return []
    
    async def close(self):
        """Close InfluxDB connection"""
        try:
            if self.client:
                self.client.close()
                self.connected = False
                logger.info("✅ Closed InfluxDB connection")
        except Exception as e:
            logger.error(f"❌ Error closing InfluxDB connection: {e}")