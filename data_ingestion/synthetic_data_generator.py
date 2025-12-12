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
        
        # Temperature simulation state (for realistic fluctuations)
        self.base_temperature = 1520.0
        self.temperature_drift = 0.0
        self.last_anomaly_time = 0.0
        self.start_time = datetime.utcnow().timestamp()
        
        # Belt speed simulation state
        self.last_maintenance_stop = 0.0
        self.maintenance_in_progress = False
    
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
    
    def simulate_furnace_temperature(self) -> float:
        """Simulate realistic furnace temperature with diurnal patterns, drift, and anomalies"""
        current_time = datetime.utcnow().timestamp()
        elapsed_time = current_time - self.start_time
        
        # Calculate hour of day (0-24)
        hour_of_day = (elapsed_time % 86400) / 3600
        
        # Start with base temperature
        temperature = self.base_temperature
        
        # Diurnal pattern: ±15°C sinusoidal variation (simulates daily ambient changes)
        diurnal_variation = 15 * np.sin(2 * np.pi * hour_of_day / 24)
        temperature += diurnal_variation
        
        # Random drift (slowly changing bias)
        self.temperature_drift += np.random.normal(0, 0.5)
        self.temperature_drift *= 0.95  # Decay factor to prevent runaway drift
        temperature += self.temperature_drift
        
        # Gaussian noise (measurement noise + small fluctuations)
        temperature += np.random.normal(0, 5)
        
        # Periodic temperature cycles to create conditions for different defect types
        # Create temperature waves that favor different defects with more pronounced variations
        cycle_phase = (elapsed_time % 14400) / 14400  # 4-hour cycle
        if cycle_phase < 0.2:  # First period - high temperature for cracks
            temperature += 50 + 20 * np.sin(2 * np.pi * elapsed_time / 1800)  # Rapid fluctuations
        elif cycle_phase < 0.4:  # Second period - temperature fluctuations for bubbles
            temperature += 30 * np.sin(2 * np.pi * elapsed_time / 300)  # Very rapid fluctuations
        elif cycle_phase < 0.6:  # Third period - stable temperature
            temperature += np.random.normal(0, 3)  # Less variation
        elif cycle_phase < 0.8:  # Fourth period - temperature cycling
            temperature += 40 * np.sin(2 * np.pi * elapsed_time / 1200)  # Moderate fluctuations
        else:  # Fifth period - mixed conditions
            temperature += 20 * np.sin(2 * np.pi * elapsed_time / 600)  # Standard fluctuations
        
        # Anomaly events: 1% probability per update (simulates equipment issues)
        time_since_anomaly = current_time - self.last_anomaly_time
        if time_since_anomaly > 3600 and np.random.random() < 0.01:  # At least 1 hour since last
            temperature += 40  # +40°C spike
            self.last_anomaly_time = current_time
            logger.info("🔥 Temperature anomaly spike: +40°C")
        
        # Clamp to physical limits
        return np.clip(temperature, 1400, 1700)
    
    def simulate_belt_speed(self) -> float:
        """Simulate realistic belt speed with maintenance stops and variations"""
        current_time = datetime.utcnow().timestamp()
        elapsed_time = current_time - self.start_time
        
        # Base speed
        speed = 150.0
        
        # Maintenance stops: 0.5% probability per update, lasts 2-5 minutes
        if not self.maintenance_in_progress:
            if np.random.random() < 0.005:
                self.maintenance_in_progress = True
                self.last_maintenance_stop = current_time
                logger.info("🛠️ Belt maintenance stop initiated")
        else:
            # Check if maintenance is over (2-5 minutes)
            if (current_time - self.last_maintenance_stop) > np.random.uniform(120, 300):
                self.maintenance_in_progress = False
                logger.info("✅ Belt maintenance stop completed")
            else:
                return 0.0  # Belt stopped during maintenance
        
        # Normal variation ±10 m/min
        speed += np.random.normal(0, 10)
        
        # Periodic speed adjustments (simulates production rate changes)
        speed += 20 * np.sin(2 * np.pi * elapsed_time / 7200)  # 2-hour cycle
        
        # Create speed conditions that favor different defect types with more pronounced variations
        cycle_phase = (elapsed_time % 14400) / 14400  # 4-hour cycle
        if cycle_phase < 0.2:  # First period - normal conditions
            pass  # Keep base speed
        elif cycle_phase < 0.4:  # Second period - slow speed for cloudiness
            speed = max(0, speed - 30 + np.random.normal(0, 8))
        elif cycle_phase < 0.6:  # Third period - high speed for chips and deformations
            speed = min(200, speed + 40 + np.random.normal(0, 10))
        elif cycle_phase < 0.8:  # Fourth period - variable speed for stains
            speed += 25 * np.sin(2 * np.pi * elapsed_time / 180)  # Rapid oscillations
        else:  # Fifth period - mixed conditions
            speed += 15 * np.sin(2 * np.pi * elapsed_time / 300)  # Moderate variations
        
        # Clamp to operational range
        return np.clip(speed, 0, 200)
    
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
                },
                {
                    "type": "mold_temp_fluctuation",
                    "parameter": "mold_temp",
                    "deviation": np.random.uniform(-20, 20),  # ±20°C
                    "duration": np.random.randint(4, 12)
                },
                {
                    "type": "forming_pressure_spike",
                    "parameter": "forming_pressure",
                    "deviation": np.random.uniform(10, 20),  # +10 to +20 MPa
                    "duration": np.random.randint(3, 7)
                },
                {
                    "type": "annealing_temp_drop",
                    "parameter": "annealing_temp",
                    "deviation": -np.random.uniform(30, 50),  # -30 to -50°C drop
                    "duration": np.random.randint(5, 10)
                },
                {
                    "type": "cooling_rate_spike",
                    "parameter": "cooling_rate",
                    "deviation": np.random.uniform(2, 4),  # +2 to +4 °C/min
                    "duration": np.random.randint(3, 8)
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
        
        # Add subtle periodic variations to create more diverse conditions
        current_time = datetime.utcnow().timestamp()
        elapsed_time = current_time - self.start_time
        
        # Add parameter-specific variations
        if parameter == "furnace_temperature":
            # Subtle temperature waves
            value += 5 * np.sin(2 * np.pi * elapsed_time / 900)  # 15-minute cycle
        elif parameter == "forming_pressure":
            # Pressure variations that affect cloudiness and stains
            value += 3 * np.sin(2 * np.pi * elapsed_time / 600)  # 10-minute cycle
        elif parameter == "mold_temp":
            # Mold temperature variations that affect chips
            value += 8 * np.sin(2 * np.pi * elapsed_time / 1200)  # 20-minute cycle
        elif parameter == "annealing_temp":
            # Annealing temperature variations
            value += 4 * np.sin(2 * np.pi * elapsed_time / 800)  # 13-minute cycle
        elif parameter == "cooling_rate":
            # Cooling rate variations
            value += 0.5 * np.sin(2 * np.pi * elapsed_time / 400)  # 7-minute cycle
        
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
        
        # Use realistic simulation for temperature and speed
        if state_key == "furnace_temperature":
            value = self.simulate_furnace_temperature()
        elif state_key == "belt_speed":
            value = self.simulate_belt_speed()
        else:
            # Generate value with realistic drift for other parameters
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
        """Generate defect detection event from MIK-1 camera with more diverse defect types"""
        # Base defect probabilities for different defect types
        base_defect_probs = {
            "bubble": 0.03,      # 3% base rate - most common
            "crack": 0.02,       # 2% base rate
            "stain": 0.015,      # 1.5% base rate
            "cloudiness": 0.01,   # 1% base rate
            "chip": 0.008,       # 0.8% base rate
            "deformation": 0.007  # 0.7% base rate
        }
        
        # Increase probability based on parameter deviations to create more diverse conditions
        temp_deviation = abs(self.current_state["furnace_temperature"] - 1520) / 30
        speed_deviation = abs(self.current_state["belt_speed"] - 150) / 10
        pressure_deviation = abs(self.current_state["forming_pressure"] - 45) / 5
        mold_temp_deviation = abs(self.current_state["mold_temp"] - 320) / 20
        
        # Modify defect probabilities based on specific parameter conditions
        defect_probabilities = base_defect_probs.copy()
        
        # High temperature increases crack probability
        if self.current_state["furnace_temperature"] > 1580:
            defect_probabilities["crack"] += temp_deviation * 0.02
        
        # Temperature fluctuations increase bubble probability
        if temp_deviation > 1.5:
            defect_probabilities["bubble"] += temp_deviation * 0.015
        
        # High belt speed increases deformation probability
        if self.current_state["belt_speed"] > 170:
            defect_probabilities["deformation"] += speed_deviation * 0.02
        
        # Pressure variations affect cloudiness and stains
        if pressure_deviation > 1.0:
            defect_probabilities["cloudiness"] += pressure_deviation * 0.01
            defect_probabilities["stain"] += pressure_deviation * 0.008
        
        # Mold temperature affects chip probability
        if mold_temp_deviation > 1.0:
            defect_probabilities["chip"] += mold_temp_deviation * 0.01
        
        # Add periodic variations to create conditions for different defect types
        current_time = datetime.utcnow().timestamp()
        elapsed_time = current_time - self.start_time
        
        # Create periodic cycles that favor different defect types
        cycle_phase = (elapsed_time % 14400) / 14400  # 4-hour cycle
        
        if cycle_phase < 0.2:  # First period - conditions for cracks
            # Create temperature conditions that favor cracks
            defect_probabilities["crack"] += 0.03
            # Reduce other defect probabilities to make cracks more prominent
            for defect_type in defect_probabilities:
                if defect_type != "crack":
                    defect_probabilities[defect_type] *= 0.7
        elif cycle_phase < 0.4:  # Second period - conditions for bubbles
            # Create temperature fluctuation conditions that favor bubbles
            defect_probabilities["bubble"] += 0.04
            # Reduce other defect probabilities
            for defect_type in defect_probabilities:
                if defect_type != "bubble":
                    defect_probabilities[defect_type] *= 0.7
        elif cycle_phase < 0.6:  # Third period - conditions for deformations
            # Create high speed conditions that favor deformations
            defect_probabilities["deformation"] += 0.03
            # Reduce other defect probabilities
            for defect_type in defect_probabilities:
                if defect_type != "deformation":
                    defect_probabilities[defect_type] *= 0.7
        elif cycle_phase < 0.8:  # Fourth period - conditions for chips
            # Create mold temperature conditions that favor chips
            defect_probabilities["chip"] += 0.025
            # Reduce other defect probabilities
            for defect_type in defect_probabilities:
                if defect_type != "chip":
                    defect_probabilities[defect_type] *= 0.7
        else:  # Fifth period - mixed conditions
            # Slightly increase all defect probabilities
            for defect_type in defect_probabilities:
                defect_probabilities[defect_type] += 0.01
        
        # Calculate total probability
        total_defect_probability = sum(defect_probabilities.values())
        total_defect_probability = min(total_defect_probability, 0.4)  # Cap at 40%
        
        if np.random.random() < total_defect_probability:
            # Select defect type based on weighted probabilities
            defect_types = list(defect_probabilities.keys())
            defect_weights = list(defect_probabilities.values())
            # Normalize weights
            total_weight = sum(defect_weights)
            if total_weight > 0:
                defect_weights = [w/total_weight for w in defect_weights]
            
            defect_type = np.random.choice(defect_types, p=defect_weights)
            
            # Defect severity based on parameter state and defect type
            severity_score = (temp_deviation + speed_deviation + pressure_deviation + mold_temp_deviation) / 4
            
            # Different defect types have different severity characteristics
            if defect_type in ["crack", "deformation"]:
                severity_score *= 1.2  # These are typically more severe
            elif defect_type in ["stain", "cloudiness"]:
                severity_score *= 0.8   # These are typically less severe
            
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
