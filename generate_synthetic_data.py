#!/usr/bin/env python3
"""
Script to generate synthetic data for the Glass Production Predictive Analytics System
and feed it directly into the backend REST API.
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Dict, Any
import logging
import aiohttp

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MQTT and Kafka imports removed - using REST API instead


class SyntheticDataGenerator:
    """Generates realistic synthetic data for glass production"""
    
    def __init__(self):
        self.base_time = datetime.utcnow()
        
    def generate_sensor_data(self) -> Dict[str, Any]:
        """Generate synthetic sensor data"""
        current_time = datetime.utcnow()
        
        # Furnace data (critical for defects)
        furnace_temp = 1500 + random.uniform(-50, 50)  # 1450-1550°C
        furnace_pressure = 15 + random.uniform(-3, 3)   # kPa
        melt_level = 2500 + random.uniform(-200, 200)   # mm
        
        # Forming data
        mold_temp = 320 + random.uniform(-30, 30)       # °C
        belt_speed = 150 + random.uniform(-20, 20)      # m/min
        forming_pressure = 50 + random.uniform(-10, 10) # bar
        
        # Annealing data
        annealing_temp = 600 + random.uniform(-50, 50)  # °C
        
        # Process data
        batch_flow = 2000 + random.uniform(-300, 300)   # kg/h
        
        return {
            "timestamp": current_time.isoformat() + "Z",
            "production_line": "Line_A",
            "sensors": {
                "furnace": {
                    "temperature": round(furnace_temp, 2),
                    "pressure": round(furnace_pressure, 2),
                    "melt_level": round(melt_level, 2),
                    "o2_percent": round(5.0 + random.uniform(-0.5, 0.5), 2),
                    "co2_percent": round(10.0 + random.uniform(-1.0, 1.0), 2)
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
        """Generate synthetic defect data"""
        current_time = datetime.utcnow()
        defect_types = ["crack", "bubble", "chip", "cloudiness", "deformation", 
                       "inclusion", "stress", "surface_defect"]
        
        # Generate realistic defect probabilities based on sensor data
        defect_count = random.randint(0, 5)
        defect_list = random.sample(defect_types, k=min(defect_count, len(defect_types)))
        
        return {
            "timestamp": current_time.isoformat() + "Z",
            "production_line": "Line_A",
            "defects": [
                {
                    "type": defect_type,
                    "severity": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                    "position": {
                        "x": round(random.uniform(0, 1000), 2),
                        "y": round(random.uniform(0, 500), 2)
                    },
                    "size_mm": round(random.uniform(0.1, 10.0), 2),
                    "confidence": round(random.uniform(0.7, 0.99), 4)
                } for defect_type in defect_list
            ]
        }
    
    def generate_quality_data(self) -> Dict[str, Any]:
        """Generate synthetic quality metrics"""
        current_time = datetime.utcnow()
        
        # Generate realistic quality score based on defect data
        quality_score = 0.95 + random.uniform(-0.15, 0.05)  # 80-100% quality
        quality_score = max(0.0, min(1.0, quality_score))  # Clamp between 0 and 1
        
        # Use dynamic units produced based on quality score rather than hardcoded range
        total_units = max(100, int(quality_score * 1200))  # Scale with quality
        defective_units = int(total_units * (1 - quality_score))
        
        return {
            "timestamp": current_time.isoformat() + "Z",
            "production_line": "Line_ёA",
            "metrics": {
                "total_units": total_units,
                "defective_units": defective_units,
                "quality_rate": round(quality_score * 100, 2),
                "first_pass_yield": round(quality_score * 100, 2),
                "oee": round(quality_score * 95, 2)  # Overall Equipment Effectiveness
            }
        }


class DataFeeder:
    """Feeds synthetic data into the system via REST API"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.generator = SyntheticDataGenerator()
        self.session = None
        
    async def initialize_session(self):
        """Initialize HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        
    async def close_session(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def send_sensor_data(self, data: Dict[str, Any]):
        """Send sensor data to backend API"""
        try:
            await self.initialize_session()
            url = f"{self.backend_url}/api/sensors/data"
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    logger.debug("📤 Sensor data sent successfully")
                    return True
                else:
                    logger.error(f"❌ Failed to send sensor data: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Error sending sensor data: {e}")
            return False
    
    async def send_defect_data(self, data: Dict[str, Any]):
        """Send defect data to backend API"""
        try:
            await self.initialize_session()
            url = f"{self.backend_url}/api/defects"
            # Transform defect data to match the expected format
            if "defects" in data and len(data["defects"]) > 0:
                # Send each defect individually
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
                    async with self.session.post(url, json=defect_payload) as response:
                        if response.status == 200:
                            logger.debug(f"📤 Defect data ({defect['type']}) sent successfully")
                        else:
                            logger.error(f"❌ Failed to send defect data: {response.status}")
            return True
        except Exception as e:
            logger.error(f"❌ Error sending defect data: {e}")
            return False
    
    async def feed_data_continuously(self, interval: float = 2.0):
        """Continuously feed synthetic data into the system via REST API"""
        logger.info(f"🔄 Starting continuous data feeding every {interval} seconds")
        
        try:
            await self.initialize_session()
            counter = 0
            while True:
                counter += 1
                logger.info(f"📊 Generating data batch #{counter}")
                
                # Generate different types of data
                sensor_data = self.generator.generate_sensor_data()
                defect_data = self.generator.generate_defect_data()
                quality_data = self.generator.generate_quality_data()
                
                # Send sensor data
                await self.send_sensor_data(sensor_data)
                
                # Send defect data (less frequently)
                if counter % 3 == 0:
                    await self.send_defect_data(defect_data)
                
                # Wait before next batch
                await asyncio.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Stopping data feeding...")
        finally:
            await self.close_session()


async def main():
    """Main function to run the synthetic data generator"""
    logger.info("🚀 Starting Synthetic Data Generator for Glass Production System")
    
    # Check if running in Docker environment
    import os
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
    backend_url = "http://backend:8000" if is_docker else "http://localhost:8000"
    
    feeder = DataFeeder(backend_url=backend_url)
    
    # Run for a specified duration or until interrupted
    try:
        await feeder.feed_data_continuously(interval=2.0)
    except KeyboardInterrupt:
        logger.info("👋 Data generation stopped by user")
    finally:
        await feeder.close_session()


if __name__ == "__main__":
    asyncio.run(main())