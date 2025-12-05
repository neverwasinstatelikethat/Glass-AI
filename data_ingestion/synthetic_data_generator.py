"""
Synthetic Data Generator for Glass Production Simulation
Generates realistic sensor data matching task.md specifications
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensorConfig:
    """Configuration for a sensor"""
    sensor_id: str
    parameter_name: str
    unit: str
    min_value: float
    max_value: float
    normal_mean: float
    normal_std: float
    frequency_seconds: float  # How often to generate reading


class GlassProductionDataGenerator:
    """Generate synthetic sensor data for glass production"""
    
    def __init__(self):
        self.sensors = self._initialize_sensors()
        self.defect_types = ["crack", "bubble", "chip", "cloudiness", "deformation", "stain"]
        self.production_lines = ["Line_A", "Line_B", "Line_C"]
        
        # State tracking for realistic data generation
        self.current_state = {
            "furnace_temperature": 1520.0,
            "furnace_pressure": 15.2,
            "melt_level": 2500.0,
            "belt_speed": 150.0,
            "mold_temp": 320.0,
            "forming_pressure": 45.0,
            "annealing_temp": 580.0,
            "cooling_rate": 3.5
        }
        
        # Anomaly injection state
        self.anomaly_probability = 0.05  # 5% chance of anomaly
        self.current_anomalies = []
    
    def _initialize_sensors(self) -> Dict[str, SensorConfig]:
        """Initialize sensor configurations matching task.md"""
        return {
            # Furnace sensors (1/min frequency)
            "FURNACE_01_TEMP": SensorConfig(
                "FURNACE_01", "Temperature", "°C", 
                1200, 1700, 1520, 30, 60
            ),
            "FURNACE_01_PRESSURE": SensorConfig(
                "FURNACE_01", "Pressure", "kPa",
                0, 50, 15, 3, 60
            ),
            "FURNACE_01_MELT_LEVEL": SensorConfig(
                "FURNACE_01", "MeltLevel", "mm",
                0, 5000, 2500, 200, 300
            ),
            
            # Forming sensors (1/sec frequency)
            "FORMING_03_SPEED": SensorConfig(
                "FORMING_03", "BeltSpeed", "m/min",
                0, 200, 150, 10, 1
            ),
            "FORMING_03_MOLD_TEMP": SensorConfig(
                "FORMING_03", "MoldTemp", "°C",
                20, 600, 320, 20, 1
            ),
            "FORMING_03_PRESSURE": SensorConfig(
                "FORMING_03", "FormingPressure", "MPa",
                0, 100, 45, 5, 1
            ),
            
            # Annealing sensors (1/2sec frequency)
            "ANNEALING_05_TEMP": SensorConfig(
                "ANNEALING_05", "ProductTemp", "°C",
                20, 1200, 580, 30, 2
            ),
            "ANNEALING_05_COOLING": SensorConfig(
                "ANNEALING_05", "CoolingRate", "°C/min",
                2, 5, 3.5, 0.5, 2
            ),
        }
    
    def _inject_anomaly(self) -> Optional[Dict[str, Any]]:
        """Randomly inject anomalies to simulate real production issues"""
        if np.random.random() < self.anomaly_probability:
            anomaly_types = [
                {
                    "type": "temperature_spike",
                    "parameter": "furnace_temperature",
                    "deviation": np.random.uniform(40, 80),  # +40 to +80°C spike
                    "duration": np.random.randint(5, 15)  # 5-15 minutes
                },
                {
                    "type": "pressure_drop",
                    "parameter": "furnace_pressure",
                    "deviation": -np.random.uniform(5, 10),  # -5 to -10 kPa drop
                    "duration": np.random.randint(3, 10)
                },
                {
                    "type": "speed_variation",
                    "parameter": "belt_speed",
                    "deviation": np.random.uniform(-30, 30),  # ±30 m/min
                    "duration": np.random.randint(2, 8)
                }
            ]
            return np.random.choice(anomaly_types)
        return None
    
    def _apply_anomalies(self, parameter: str, base_value: float) -> float:
        """Apply active anomalies to parameter value"""
        value = base_value
        for anomaly in self.current_anomalies:
            if anomaly["parameter"] == parameter:
                value += anomaly["deviation"]
                anomaly["remaining_duration"] -= 1
        
        # Remove expired anomalies
        self.current_anomalies = [
            a for a in self.current_anomalies 
            if a.get("remaining_duration", 0) > 0
        ]
        
        return value
    
    def generate_sensor_reading(self, sensor_key: str, production_line: str = "Line_A") -> Dict[str, Any]:
        """Generate a single sensor reading"""
        sensor = self.sensors[sensor_key]
        
        # Get base value from current state
        state_key = sensor_key.split("_")[1].lower() + "_" + sensor.parameter_name.lower().replace("temp", "temperature")
        if state_key == "01_temperature":
            state_key = "furnace_temperature"
        elif state_key == "01_pressure":
            state_key = "furnace_pressure"
        elif state_key == "01_meltlevel":
            state_key = "melt_level"
        elif state_key == "03_beltspeed":
            state_key = "belt_speed"
        elif state_key == "03_moldtemperature":
            state_key = "mold_temp"
        elif state_key == "03_formingpressure":
            state_key = "forming_pressure"
        elif state_key == "05_producttemperature":
            state_key = "annealing_temp"
        elif state_key == "05_coolingrate":
            state_key = "cooling_rate"
        
        # Generate value with realistic drift
        base_value = self.current_state.get(state_key, sensor.normal_mean)
        noise = np.random.normal(0, sensor.normal_std)
        drift = np.random.normal(0, sensor.normal_std * 0.1)  # Slow drift
        
        value = base_value + noise + drift
        
        # Apply anomalies if any
        value = self._apply_anomalies(state_key, value)
        
        # Clamp to sensor range
        value = np.clip(value, sensor.min_value, sensor.max_value)
        
        # Update state
        self.current_state[state_key] = value
        
        # Determine status based on deviation from normal
        deviation_factor = abs(value - sensor.normal_mean) / sensor.normal_std
        if deviation_factor > 3:
            status = "CRITICAL"
        elif deviation_factor > 2:
            status = "ERROR"
        elif deviation_factor > 1.5:
            status = "WARNING"
        else:
            status = "OK"
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "sensor_id": sensor.sensor_id,
            "parameter_name": sensor.parameter_name,
            "value": round(value, 2),
            "unit": sensor.unit,
            "status": status,
            "production_line": production_line
        }
    
    def generate_defect_event(self, production_line: str = "Line_A") -> Optional[Dict[str, Any]]:
        """Generate defect detection event from MIK-1 camera"""
        # Defect probability based on current parameter deviations
        defect_probability = 0.02  # Base 2% defect rate
        
        # Increase probability based on parameter deviations
        temp_deviation = abs(self.current_state["furnace_temperature"] - 1520) / 30
        speed_deviation = abs(self.current_state["belt_speed"] - 150) / 10
        pressure_deviation = abs(self.current_state["forming_pressure"] - 45) / 5
        
        defect_probability += (temp_deviation + speed_deviation + pressure_deviation) * 0.01
        defect_probability = min(defect_probability, 0.3)  # Cap at 30%
        
        if np.random.random() < defect_probability:
            defect_type = np.random.choice(self.defect_types)
            
            # Defect severity based on parameter state
            severity_score = (temp_deviation + speed_deviation + pressure_deviation) / 3
            severity = "HIGH" if severity_score > 2 else "MEDIUM" if severity_score > 1 else "LOW"
            
            return {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "production_line": production_line,
                "defect_type": defect_type,
                "severity": severity,
                "confidence": round(0.7 + np.random.random() * 0.25, 2),
                "position_x": round(np.random.random() * 100, 1),
                "position_y": round(np.random.random() * 100, 1),
                "sensor_id": "MIK1_CAM_01"
            }
        return None
    
    async def generate_continuous_data(self, duration_seconds: int = 60) -> List[Dict[str, Any]]:
        """Generate continuous stream of sensor data for specified duration"""
        start_time = datetime.utcnow()
        all_readings = []
        
        while (datetime.utcnow() - start_time).total_seconds() < duration_seconds:
            # Check for new anomaly
            new_anomaly = self._inject_anomaly()
            if new_anomaly:
                new_anomaly["remaining_duration"] = new_anomaly["duration"]
                self.current_anomalies.append(new_anomaly)
                logger.info(f"🔥 Anomaly injected: {new_anomaly['type']}")
            
            # Generate readings for all sensors based on their frequency
            current_time = datetime.utcnow()
            
            for sensor_key, sensor in self.sensors.items():
                # Generate reading if it's time (based on frequency)
                reading = self.generate_sensor_reading(sensor_key)
                all_readings.append(reading)
            
            # Generate defect event (every 3-5 seconds simulating MIK-1 inspection)
            if np.random.random() < 0.25:  # ~25% chance = ~4 seconds average
                defect = self.generate_defect_event()
                if defect:
                    all_readings.append(defect)
                    logger.info(f"⚠️ Defect detected: {defect['defect_type']} ({defect['severity']})")
            
            await asyncio.sleep(1)  # Generate batch every second
        
        return all_readings
    
    def get_current_state_summary(self) -> Dict[str, Any]:
        """Get current production state summary"""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "state": self.current_state.copy(),
            "active_anomalies": len(self.current_anomalies),
            "anomaly_details": [
                {
                    "type": a["type"],
                    "parameter": a["parameter"],
                    "remaining": a["remaining_duration"]
                }
                for a in self.current_anomalies
            ]
        }


async def main():
    """Test the synthetic data generator"""
    generator = GlassProductionDataGenerator()
    
    logger.info("🏭 Starting synthetic data generation...")
    logger.info(f"Initial state: {generator.get_current_state_summary()}")
    
    # Generate 30 seconds of data
    readings = await generator.generate_continuous_data(duration_seconds=30)
    
    logger.info(f"\n📊 Generated {len(readings)} readings")
    logger.info(f"Final state: {generator.get_current_state_summary()}")
    
    # Show sample readings
    logger.info("\n📋 Sample readings:")
    for reading in readings[:5]:
        logger.info(f"  {reading}")


if __name__ == "__main__":
    asyncio.run(main())
