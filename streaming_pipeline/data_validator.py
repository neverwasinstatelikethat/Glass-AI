"""
Data Validator for Industrial Sensor Data
Performs real-time validation, cleaning, and anomaly detection
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
import json
import numpy as np
from scipy import stats
from collections import deque, defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Real-time data validation and anomaly detection for industrial sensors"""
    
    def __init__(
        self,
        window_size: int = 100,
        outlier_threshold: float = 3.0,
        validation_callback: Optional[Callable] = None
    ):
        self.window_size = window_size
        self.outlier_threshold = outlier_threshold
        self.validation_callback = validation_callback
        
        # Data buffers for statistical analysis
        self.data_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.validation_stats: Dict[str, Dict[str, float]] = {}
        self.anomaly_history: deque = deque(maxlen=1000)
        
        # Define valid ranges for different sensor types
        self.sensor_valid_ranges = {
            # Furnace sensors
            "furnace_temperature": {"min": 1000, "max": 1800, "unit": "°C", "critical": True},
            "furnace_pressure": {"min": 0, "max": 50, "unit": "bar", "critical": True},
            "furnace_melt_level": {"min": 1000, "max": 4000, "unit": "mm", "critical": True},
            "furnace_o2_percent": {"min": 0, "max": 21, "unit": "%", "critical": False},
            "furnace_co2_percent": {"min": 0, "max": 20, "unit": "%", "critical": False},
            "furnace_power": {"min": 0, "max": 100, "unit": "%", "critical": False},
            
            # Forming machine sensors
            "forming_belt_speed": {"min": 50, "max": 300, "unit": "m/min", "critical": True},
            "forming_mold_temp": {"min": 200, "max": 500, "unit": "°C", "critical": True},
            "forming_pressure": {"min": 10, "max": 100, "unit": "bar", "critical": False},
            
            # Annealing oven sensors
            "annealing_temp_zone1": {"min": 400, "max": 700, "unit": "°C", "critical": False},
            "annealing_temp_zone2": {"min": 400, "max": 700, "unit": "°C", "critical": False},
            "annealing_temp_zone3": {"min": 400, "max": 700, "unit": "°C", "critical": False},
            
            # Quality metrics
            "quality_score": {"min": 0, "max": 1, "unit": "ratio", "critical": True},
            "defect_count": {"min": 0, "max": 100, "unit": "count", "critical": True},
            "production_rate": {"min": 0, "max": 5000, "unit": "units/hr", "critical": False}
        }
        
        # Statistical models for anomaly detection
        self.statistical_models: Dict[str, Dict[str, float]] = {}
        
        # Validation rules
        self.validation_rules = {
            "missing_data_threshold": 0.1,  # 10% missing data allowed
            "stale_data_threshold": 300,    # 5 minutes stale data threshold
            "consistency_checks": True
        }
    
    def validate_sensor_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sensor data and detect anomalies"""
        try:
            timestamp = datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat()))
            sensors = data.get("sensors", {})
            production_line = data.get("production_line", "unknown")
            
            # Initialize validation result
            validation_result = {
                "timestamp": timestamp.isoformat(),
                "production_line": production_line,
                "validation_status": "valid",
                "validated_sensors": {},
                "anomalies": [],
                "recommendations": [],
                "statistics": {}
            }
            
            # Validate each sensor
            for sensor_name, sensor_data in sensors.items():
                validated_value, anomaly_info = self._validate_single_sensor(
                    sensor_name, sensor_data, timestamp
                )
                
                if validated_value is not None:
                    validation_result["validated_sensors"][sensor_name] = validated_value
                
                if anomaly_info:
                    validation_result["anomalies"].append(anomaly_info)
            
            # Perform cross-sensor consistency checks
            consistency_issues = self._check_sensor_consistency(sensors)
            validation_result["anomalies"].extend(consistency_issues)
            
            # Update validation statistics
            self._update_validation_stats(validation_result)
            
            # Set overall validation status
            if validation_result["anomalies"]:
                critical_anomalies = [a for a in validation_result["anomalies"] 
                                    if a.get("severity") == "CRITICAL"]
                if critical_anomalies:
                    validation_result["validation_status"] = "critical_error"
                    validation_result["recommendations"].append(
                        "STOP PRODUCTION - Critical sensor anomaly detected"
                    )
                else:
                    validation_result["validation_status"] = "warning"
            
            # Add statistics
            validation_result["statistics"] = {
                "total_sensors": len(sensors),
                "validated_sensors": len(validation_result["validated_sensors"]),
                "anomalies_detected": len(validation_result["anomalies"]),
                "validation_timestamp": datetime.utcnow().isoformat()
            }
            
            # Call callback if provided
            if self.validation_callback:
                asyncio.create_task(self.validation_callback(validation_result))
            
            # Log critical anomalies
            critical_anomalies = [a for a in validation_result["anomalies"] 
                                if a.get("severity") == "CRITICAL"]
            if critical_anomalies:
                logger.critical(f"🚨 CRITICAL ANOMALY DETECTED: {critical_anomalies}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Error validating sensor data: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "validation_status": "error",
                "error": str(e)
            }
    
    def _validate_single_sensor(
        self, 
        sensor_name: str, 
        sensor_data: Any, 
        timestamp: datetime
    ) -> tuple:
        """Validate a single sensor reading"""
        try:
            # Extract value (handle different data formats)
            if isinstance(sensor_data, dict):
                value = sensor_data.get("value")
                status = sensor_data.get("status", "OK")
            else:
                value = sensor_data
                status = "OK"
            
            # Skip if no value or sensor error
            if value is None or status != "OK":
                anomaly_info = {
                    "sensor": sensor_name,
                    "type": "missing_data" if value is None else "sensor_error",
                    "severity": "HIGH",
                    "timestamp": timestamp.isoformat(),
                    "message": f"Missing or invalid data for {sensor_name}"
                }
                return None, anomaly_info
            
            # Convert to numeric if needed
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError:
                    anomaly_info = {
                        "sensor": sensor_name,
                        "type": "invalid_format",
                        "severity": "MEDIUM",
                        "timestamp": timestamp.isoformat(),
                        "message": f"Invalid data format for {sensor_name}: {value}"
                    }
                    return None, anomaly_info
            
            # Check valid ranges
            if sensor_name in self.sensor_valid_ranges:
                range_info = self.sensor_valid_ranges[sensor_name]
                if not (range_info["min"] <= value <= range_info["max"]):
                    severity = "CRITICAL" if range_info["critical"] else "HIGH"
                    anomaly_info = {
                        "sensor": sensor_name,
                        "type": "out_of_range",
                        "severity": severity,
                        "timestamp": timestamp.isoformat(),
                        "message": f"{sensor_name}={value} out of range [{range_info['min']}, {range_info['max']}] {range_info['unit']}",
                        "expected_range": f"{range_info['min']}-{range_info['max']} {range_info['unit']}",
                        "actual_value": value
                    }
                    return value, anomaly_info
            
            # Add to buffer for statistical analysis
            self.data_buffers[sensor_name].append((timestamp, value))
            
            # Perform statistical anomaly detection
            statistical_anomaly = self._detect_statistical_anomaly(sensor_name, value, timestamp)
            if statistical_anomaly:
                return value, statistical_anomaly
            
            # Return validated value with no anomaly
            return value, None
            
        except Exception as e:
            logger.error(f"❌ Error validating {sensor_name}: {e}")
            anomaly_info = {
                "sensor": sensor_name,
                "type": "validation_error",
                "severity": "MEDIUM",
                "timestamp": timestamp.isoformat(),
                "message": f"Validation error for {sensor_name}: {str(e)}"
            }
            return None, anomaly_info
    
    def _detect_statistical_anomaly(
        self, 
        sensor_name: str, 
        value: float, 
        timestamp: datetime
    ) -> Optional[Dict[str, Any]]:
        """Detect statistical anomalies using z-score and other methods"""
        try:
            buffer = self.data_buffers[sensor_name]
            if len(buffer) < 10:  # Need sufficient data for statistics
                return None
            
            # Extract values for analysis
            values = [v for t, v in buffer]
            timestamps = [t for t, v in buffer]
            
            # Calculate statistics
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # Z-score anomaly detection
            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                if z_score > self.outlier_threshold:
                    anomaly_info = {
                        "sensor": sensor_name,
                        "type": "statistical_anomaly",
                        "severity": "HIGH",
                        "timestamp": timestamp.isoformat(),
                        "message": f"Statistical anomaly detected for {sensor_name}: z-score={z_score:.2f}",
                        "z_score": z_score,
                        "mean": mean_val,
                        "std": std_val,
                        "value": value
                    }
                    self.anomaly_history.append(anomaly_info)
                    return anomaly_info
            
            # Rate of change anomaly detection
            if len(values) >= 2:
                rate_of_change = abs(values[-1] - values[-2])
                time_diff = (timestamps[-1] - timestamps[-2]).total_seconds()
                if time_diff > 0:
                    rate_per_second = rate_of_change / time_diff
                    
                    # Check if rate is unusually high (compared to historical rates)
                    if len(values) >= 3:
                        historical_rates = []
                        for i in range(1, min(10, len(values))):
                            if i < len(timestamps):
                                dt = (timestamps[-i] - timestamps[-i-1]).total_seconds()
                                if dt > 0:
                                    historical_rates.append(abs(values[-i] - values[-i-1]) / dt)
                        
                        if historical_rates:
                            mean_rate = np.mean(historical_rates)
                            std_rate = np.std(historical_rates)
                            if std_rate > 0 and rate_per_second > (mean_rate + 3 * std_rate):
                                anomaly_info = {
                                    "sensor": sensor_name,
                                    "type": "rapid_change",
                                    "severity": "MEDIUM",
                                    "timestamp": timestamp.isoformat(),
                                    "message": f"Rapid change detected for {sensor_name}: {rate_per_second:.2f} units/sec",
                                    "rate_of_change": rate_per_second,
                                    "historical_mean_rate": mean_rate
                                }
                                self.anomaly_history.append(anomaly_info)
                                return anomaly_info
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error in statistical anomaly detection for {sensor_name}: {e}")
            return None
    
    def _check_sensor_consistency(self, sensors: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform cross-sensor consistency checks"""
        inconsistencies = []
        
        try:
            # Check furnace temperature vs melt level consistency
            furnace_temp = sensors.get("furnace_temperature")
            melt_level = sensors.get("furnace_melt_level")
            
            if furnace_temp is not None and melt_level is not None:
                # Expected relationship: higher temperature should correlate with adequate melt level
                # This is a simplified check - in reality, this would be more complex
                if furnace_temp > 1600 and melt_level < 2000:
                    inconsistencies.append({
                        "type": "inconsistency",
                        "severity": "MEDIUM",
                        "message": "High furnace temperature with low melt level - possible sensor error",
                        "sensors": ["furnace_temperature", "furnace_melt_level"]
                    })
            
            # Check forming speed vs quality consistency
            belt_speed = sensors.get("forming_belt_speed")
            quality_score = sensors.get("quality_score")
            
            if belt_speed is not None and quality_score is not None:
                # Higher speeds might correlate with lower quality
                if belt_speed > 200 and quality_score > 0.95:
                    inconsistencies.append({
                        "type": "inconsistency",
                        "severity": "LOW",
                        "message": "High belt speed with high quality score - verify measurements",
                        "sensors": ["forming_belt_speed", "quality_score"]
                    })
            
        except Exception as e:
            logger.error(f"❌ Error in consistency checks: {e}")
        
        return inconsistencies
    
    def _update_validation_stats(self, validation_result: Dict[str, Any]):
        """Update validation statistics"""
        try:
            timestamp = datetime.fromisoformat(validation_result["timestamp"])
            
            # Update per-sensor statistics
            for sensor_name in validation_result["validated_sensors"]:
                if sensor_name not in self.validation_stats:
                    self.validation_stats[sensor_name] = {
                        "total_validations": 0,
                        "anomalies_detected": 0,
                        "last_validation": timestamp.isoformat()
                    }
                
                self.validation_stats[sensor_name]["total_validations"] += 1
                self.validation_stats[sensor_name]["last_validation"] = timestamp.isoformat()
            
            # Update anomaly counts
            for anomaly in validation_result["anomalies"]:
                sensor_name = anomaly.get("sensor")
                if sensor_name and sensor_name in self.validation_stats:
                    self.validation_stats[sensor_name]["anomalies_detected"] += 1
        
        except Exception as e:
            logger.error(f"❌ Error updating validation stats: {e}")
    
    def clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and interpolate missing or invalid data"""
        try:
            cleaned_data = data.copy()
            sensors = cleaned_data.get("sensors", {})
            
            for sensor_name, sensor_data in sensors.items():
                # Handle missing or invalid data
                if isinstance(sensor_data, dict):
                    value = sensor_data.get("value")
                    status = sensor_data.get("status", "OK")
                else:
                    value = sensor_data
                    status = "OK"
                
                # If data is missing or invalid, try to interpolate
                if value is None or status != "OK":
                    interpolated_value = self._interpolate_missing_value(sensor_name)
                    if interpolated_value is not None:
                        if isinstance(sensor_data, dict):
                            sensors[sensor_name]["value"] = interpolated_value
                            sensors[sensor_name]["status"] = "INTERPOLATED"
                        else:
                            sensors[sensor_name] = interpolated_value
                        logger.info(f"🔄 Interpolated value for {sensor_name}: {interpolated_value}")
            
            cleaned_data["sensors"] = sensors
            return cleaned_data
            
        except Exception as e:
            logger.error(f"❌ Error cleaning data: {e}")
            return data
    
    def _interpolate_missing_value(self, sensor_name: str) -> Optional[float]:
        """Interpolate missing sensor value based on historical data"""
        try:
            buffer = self.data_buffers[sensor_name]
            if len(buffer) < 2:
                return None
            
            # Simple linear interpolation using last two valid values
            values = [v for t, v in buffer if v is not None]
            if len(values) >= 2:
                # Use last two values for interpolation
                return (values[-1] + values[-2]) / 2
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error interpolating {sensor_name}: {e}")
            return None
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "validation_stats": self.validation_stats,
            "recent_anomalies": list(self.anomaly_history),
            "sensor_valid_ranges": self.sensor_valid_ranges,
            "configuration": {
                "window_size": self.window_size,
                "outlier_threshold": self.outlier_threshold
            }
        }
    
    def reset_statistics(self):
        """Reset all validation statistics"""
        self.validation_stats.clear()
        self.anomaly_history.clear()
        self.data_buffers.clear()
        logger.info("📊 Validation statistics reset")


class DataQualityMonitor:
    """Monitor overall data quality metrics"""
    
    def __init__(self, validator: DataValidator):
        self.validator = validator
        self.quality_metrics = {
            "data_completeness": 1.0,
            "data_freshness": 0.0,  # seconds since last update
            "anomaly_rate": 0.0,
            "validation_rate": 1.0
        }
    
    def update_quality_metrics(self, validation_result: Dict[str, Any]) -> Dict[str, float]:
        """Update data quality metrics based on validation results"""
        try:
            total_sensors = validation_result.get("statistics", {}).get("total_sensors", 0)
            validated_sensors = validation_result.get("statistics", {}).get("validated_sensors", 0)
            anomalies = validation_result.get("statistics", {}).get("anomalies_detected", 0)
            
            # Calculate completeness
            if total_sensors > 0:
                self.quality_metrics["data_completeness"] = validated_sensors / total_sensors
            
            # Calculate anomaly rate
            if validated_sensors > 0:
                self.quality_metrics["anomaly_rate"] = anomalies / validated_sensors
            
            # Update freshness (time since last validation)
            validation_time = datetime.fromisoformat(validation_result["timestamp"])
            current_time = datetime.utcnow()
            self.quality_metrics["data_freshness"] = (current_time - validation_time).total_seconds()
            
            return self.quality_metrics.copy()
            
        except Exception as e:
            logger.error(f"❌ Error updating quality metrics: {e}")
            return self.quality_metrics
    
    def get_quality_score(self) -> float:
        """Calculate overall data quality score (0-1)"""
        try:
            # Weighted score based on different metrics
            completeness_score = self.quality_metrics["data_completeness"]
            freshness_score = max(0, 1 - (self.quality_metrics["data_freshness"] / 300))  # 5 min threshold
            anomaly_score = 1 - self.quality_metrics["anomaly_rate"]
            
            # Weighted average
            quality_score = (
                0.4 * completeness_score +
                0.3 * freshness_score +
                0.3 * anomaly_score
            )
            
            return max(0, min(1, quality_score))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"❌ Error calculating quality score: {e}")
            return 0.0


async def main_example():
    """Example usage of Data Validator"""
    
    async def validation_callback(result):
        """Callback for validation results"""
        print(f"\n🔍 Validation Result: {result['validation_status']}")
        print(f"   Timestamp: {result['timestamp']}")
        print(f"   Validated sensors: {len(result['validated_sensors'])}")
        print(f"   Anomalies: {len(result['anomalies'])}")
        
        if result['anomalies']:
            print("   🔴 Anomalies detected:")
            for anomaly in result['anomalies'][:3]:  # Show first 3
                print(f"     - {anomaly.get('message', 'Unknown anomaly')}")
    
    # Create validator
    validator = DataValidator(validation_callback=validation_callback)
    
    # Simulate sensor data validation
    print("🧪 Testing Data Validator with simulated sensor data...")
    
    # Normal data
    normal_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "production_line": "Line_A",
        "sensors": {
            "furnace_temperature": 1500.0,
            "furnace_pressure": 15.0,
            "furnace_melt_level": 2500.0,
            "forming_belt_speed": 150.0,
            "quality_score": 0.95
        }
    }
    
    result = validator.validate_sensor_data(normal_data)
    print(f"✅ Normal data validation: {result['validation_status']}")
    
    # Anomalous data
    anomalous_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "production_line": "Line_A",
        "sensors": {
            "furnace_temperature": 2000.0,  # Too high
            "furnace_pressure": 15.0,
            "furnace_melt_level": 500.0,   # Too low
            "forming_belt_speed": 150.0,
            "quality_score": 0.95
        }
    }
    
    result = validator.validate_sensor_data(anomalous_data)
    print(f"🔴 Anomalous data validation: {result['validation_status']}")
    if result['anomalies']:
        print(f"   Detected {len(result['anomalies'])} anomalies")
    
    # Show validation report
    report = validator.get_validation_report()
    print(f"\n📊 Validation Report:")
    print(f"   Tracked sensors: {len(report['validation_stats'])}")
    print(f"   Recent anomalies: {len(report['recent_anomalies'])}")


if __name__ == "__main__":
    asyncio.run(main_example())