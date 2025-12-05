"""
Real-Time Feature Engineering for Glass Production
Computes streaming features from sensor data for ML models
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeFeatureExtractor:
    """Extracts real-time features from streaming sensor data"""
    
    def __init__(
        self,
        window_size: int = 60,
        feature_callback: Optional[Callable] = None
    ):
        self.window_size = window_size
        self.feature_callback = feature_callback
        self.running = False
        
        # Sliding windows for different sensors
        self.sensor_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        
        # Timestamp window
        self.timestamp_window = deque(maxlen=window_size)
        
        # Feature buffers for derived features
        self.derived_features: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        
        # Feature definitions
        self.feature_definitions = {
            "statistical": ["mean", "std", "min", "max", "median", "skew", "kurtosis"],
            "temporal": ["trend", "rate_of_change", "volatility", "momentum"],
            "frequency": ["dominant_frequency", "spectral_power", "frequency_stability"],
            "cross_sensor": ["correlation", "ratio", "difference"],
            "domain_specific": ["viscosity_index", "thermal_gradient", "process_stability"]
        }
    
    async def update_with_sensor_data(self, sensor_data: Dict[str, Any]):
        """Update feature extractor with new sensor data"""
        try:
            timestamp = datetime.fromisoformat(sensor_data.get("timestamp", datetime.utcnow().isoformat()))
            
            # Update timestamp window
            self.timestamp_window.append(timestamp)
            
            # Flatten nested sensor data
            flat_sensors = self._flatten_sensor_data(sensor_data.get("sensors", {}))
            
            # Update sensor windows
            for sensor_name, value in flat_sensors.items():
                if value is not None and not np.isnan(value):
                    self.sensor_windows[sensor_name].append((timestamp, float(value)))
            
            # Compute derived features
            await self._compute_derived_features(flat_sensors, timestamp)
            
            # Extract features if we have enough data
            if len(self.timestamp_window) >= 10:
                features = self.extract_features()
                
                # Call callback if provided
                if self.feature_callback:
                    await self.feature_callback(features)
                
                return features
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error updating with sensor data: {e}")
            return {}
    
    def _flatten_sensor_data(self, sensors: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested sensor data structure"""
        flat_data = {}
        
        def _flatten_recursive(data, prefix=""):
            for key, value in data.items():
                new_key = f"{prefix}{key}" if prefix else key
                
                if isinstance(value, dict):
                    # Recursively flatten nested dictionaries
                    if "value" in value:
                        # Handle sensor data format { "value": X, "status": "OK" }
                        flat_data[new_key] = value["value"]
                    else:
                        _flatten_recursive(value, f"{new_key}_")
                else:
                    flat_data[new_key] = value
        
        _flatten_recursive(sensors)
        return flat_data
    
    async def _compute_derived_features(self, flat_sensors: Dict[str, float], timestamp: datetime):
        """Compute derived features from raw sensor data"""
        try:
            # Compute viscosity index (domain-specific)
            furnace_temp = flat_sensors.get("furnace_temperature")
            if furnace_temp is not None:
                # Simplified viscosity model: viscosity decreases with temperature
                viscosity_index = 1000 / (furnace_temp + 273.15)  # Kelvin
                self.derived_features["viscosity_index"].append((timestamp, viscosity_index))
            
            # Compute thermal gradient
            furnace_temp = flat_sensors.get("furnace_temperature")
            mold_temp = flat_sensors.get("forming_mold_temperature")
            if furnace_temp is not None and mold_temp is not None:
                thermal_gradient = furnace_temp - mold_temp
                self.derived_features["thermal_gradient"].append((timestamp, thermal_gradient))
            
            # Compute process stability index
            await self._compute_stability_index(flat_sensors, timestamp)
            
        except Exception as e:
            logger.error(f"❌ Error computing derived features: {e}")
    
    async def _compute_stability_index(self, flat_sensors: Dict[str, float], timestamp: datetime):
        """Compute process stability index from sensor variations"""
        try:
            stability_scores = []
            
            # Calculate coefficient of variation for each sensor
            for sensor_name, window in self.sensor_windows.items():
                if len(window) >= 5:
                    values = [v for t, v in window]
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        if mean_val != 0:
                            cv = std_val / abs(mean_val)
                            # Stability score: lower CV means higher stability
                            stability_score = 1 / (1 + cv)
                            stability_scores.append(stability_score)
            
            if stability_scores:
                stability_index = np.mean(stability_scores)
                self.derived_features["process_stability_index"].append((timestamp, stability_index))
            
        except Exception as e:
            logger.error(f"❌ Error computing stability index: {e}")
    
    def extract_features(self) -> Dict[str, Any]:
        """Extract all features from current windows"""
        try:
            features = {
                "timestamp": datetime.utcnow().isoformat(),
                "extraction_time": datetime.utcnow().isoformat(),
                "feature_counts": {}
            }
            
            # Extract statistical features
            stat_features = self._extract_statistical_features()
            features.update(stat_features)
            features["feature_counts"]["statistical"] = len(stat_features) - 3  # Exclude metadata
            
            # Extract temporal features
            temporal_features = self._extract_temporal_features()
            features.update(temporal_features)
            features["feature_counts"]["temporal"] = len(temporal_features) - len(stat_features)
            
            # Extract frequency features
            frequency_features = self._extract_frequency_features()
            features.update(frequency_features)
            features["feature_counts"]["frequency"] = len(frequency_features) - len(temporal_features)
            
            # Extract cross-sensor features
            cross_features = self._extract_cross_sensor_features()
            features.update(cross_features)
            features["feature_counts"]["cross_sensor"] = len(cross_features) - len(frequency_features)
            
            # Extract domain-specific features
            domain_features = self._extract_domain_features()
            features.update(domain_features)
            features["feature_counts"]["domain_specific"] = len(domain_features) - len(cross_features)
            
            # Add metadata
            features["window_size"] = len(self.timestamp_window)
            features["data_age_seconds"] = (
                datetime.utcnow() - min(self.timestamp_window) 
                if self.timestamp_window else 0
            ).total_seconds()
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    def _extract_statistical_features(self) -> Dict[str, float]:
        """Extract statistical features from sensor windows"""
        features = {}
        
        for sensor_name, window in self.sensor_windows.items():
            if len(window) >= 3:
                values = np.array([v for t, v in window])
                
                # Remove NaN values
                values = values[~np.isnan(values)]
                if len(values) == 0:
                    continue
                
                prefix = f"{sensor_name}_"
                
                # Basic statistics
                features[f"{prefix}mean"] = float(np.mean(values))
                features[f"{prefix}std"] = float(np.std(values))
                features[f"{prefix}min"] = float(np.min(values))
                features[f"{prefix}max"] = float(np.max(values))
                features[f"{prefix}median"] = float(np.median(values))
                
                # Higher-order statistics
                if len(values) >= 4:
                    features[f"{prefix}skew"] = float(stats.skew(values))
                    features[f"{prefix}kurtosis"] = float(stats.kurtosis(values))
                
                # Quantiles
                features[f"{prefix}q25"] = float(np.percentile(values, 25))
                features[f"{prefix}q75"] = float(np.percentile(values, 75))
                
                # Range and variation
                features[f"{prefix}range"] = float(np.ptp(values))
                if np.mean(values) != 0:
                    features[f"{prefix}cv"] = float(np.std(values) / abs(np.mean(values)))
        
        return features
    
    def _extract_temporal_features(self) -> Dict[str, float]:
        """Extract temporal features (trends, rates, etc.)"""
        features = {}
        
        for sensor_name, window in self.sensor_windows.items():
            if len(window) >= 5:
                timestamps = [t for t, v in window]
                values = np.array([v for t, v in window])
                
                # Remove NaN values
                valid_mask = ~np.isnan(values)
                if np.sum(valid_mask) < 3:
                    continue
                
                timestamps = [timestamps[i] for i in range(len(timestamps)) if valid_mask[i]]
                values = values[valid_mask]
                
                if len(values) < 3:
                    continue
                
                prefix = f"{sensor_name}_"
                
                # Trend using linear regression
                time_seconds = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
                if len(np.unique(time_seconds)) > 1:  # Avoid singular matrix
                    coeffs = np.polyfit(time_seconds, values, 1)
                    features[f"{prefix}trend_slope"] = float(coeffs[0])
                    features[f"{prefix}trend_intercept"] = float(coeffs[1])
                
                # Rate of change (last point vs first point)
                if len(values) >= 2:
                    time_diff = (timestamps[-1] - timestamps[0]).total_seconds()
                    if time_diff > 0:
                        rate_of_change = (values[-1] - values[0]) / time_diff
                        features[f"{prefix}rate_of_change"] = float(rate_of_change)
                
                # Volatility (standard deviation of differences)
                if len(values) >= 3:
                    differences = np.diff(values)
                    features[f"{prefix}volatility"] = float(np.std(differences))
                
                # Momentum (difference between last two points)
                if len(values) >= 2:
                    momentum = values[-1] - values[-2]
                    features[f"{prefix}momentum"] = float(momentum)
        
        return features
    
    def _extract_frequency_features(self) -> Dict[str, float]:
        """Extract frequency domain features using FFT"""
        features = {}
        
        for sensor_name, window in self.sensor_windows.items():
            if len(window) >= 10:
                values = np.array([v for t, v in window])
                values = values[~np.isnan(values)]
                
                if len(values) < 8:
                    continue
                
                prefix = f"{sensor_name}_"
                
                # Apply FFT
                fft_values = np.fft.fft(values)
                fft_magnitude = np.abs(fft_values)[:len(values)//2]
                frequencies = np.fft.fftfreq(len(values))[:len(values)//2]
                
                # Dominant frequency
                if len(fft_magnitude) > 0:
                    dominant_idx = np.argmax(fft_magnitude)
                    features[f"{prefix}dominant_frequency"] = float(frequencies[dominant_idx])
                    
                    # Spectral power
                    features[f"{prefix}spectral_power"] = float(np.sum(fft_magnitude**2))
                    
                    # Frequency stability (std of frequencies weighted by magnitude)
                    if np.sum(fft_magnitude) > 0:
                        weighted_freq_std = np.sqrt(
                            np.sum(fft_magnitude * (frequencies - np.average(frequencies, weights=fft_magnitude))**2) / 
                            np.sum(fft_magnitude)
                        )
                        features[f"{prefix}frequency_stability"] = float(weighted_freq_std)
        
        return features
    
    def _extract_cross_sensor_features(self) -> Dict[str, float]:
        """Extract features that relate multiple sensors"""
        features = {}
        
        # Get all sensor names with sufficient data
        valid_sensors = [name for name, window in self.sensor_windows.items() if len(window) >= 10]
        
        # Compute pairwise correlations
        for i, sensor1 in enumerate(valid_sensors):
            for j, sensor2 in enumerate(valid_sensors):
                if i < j:  # Avoid duplicate pairs
                    window1 = self.sensor_windows[sensor1]
                    window2 = self.sensor_windows[sensor2]
                    
                    # Align timestamps and get common data points
                    data1 = {t: v for t, v in window1}
                    data2 = {t: v for t, v in window2}
                    
                    common_timestamps = set(data1.keys()) & set(data2.keys())
                    if len(common_timestamps) >= 5:
                        values1 = np.array([data1[t] for t in common_timestamps])
                        values2 = np.array([data2[t] for t in common_timestamps])
                        
                        # Remove NaN values
                        valid_mask = ~(np.isnan(values1) | np.isnan(values2))
                        values1 = values1[valid_mask]
                        values2 = values2[valid_mask]
                        
                        if len(values1) >= 3:
                            # Correlation
                            correlation = np.corrcoef(values1, values2)[0, 1]
                            if not np.isnan(correlation):
                                features[f"{sensor1}_{sensor2}_correlation"] = float(correlation)
                            
                            # Ratio (avoid division by zero)
                            if np.mean(np.abs(values2)) > 1e-10:
                                ratio = np.mean(values1) / np.mean(values2)
                                features[f"{sensor1}_{sensor2}_ratio"] = float(ratio)
                            
                            # Difference
                            diff = np.mean(values1 - values2)
                            features[f"{sensor1}_{sensor2}_difference"] = float(diff)
        
        return features
    
    def _extract_domain_features(self) -> Dict[str, float]:
        """Extract domain-specific features for glass production"""
        features = {}
        
        # Add derived features
        for feature_name, window in self.derived_features.items():
            if len(window) >= 3:
                values = np.array([v for t, v in window])
                values = values[~np.isnan(values)]
                
                if len(values) >= 1:
                    features[f"{feature_name}_latest"] = float(values[-1])
                
                if len(values) >= 3:
                    features[f"{feature_name}_mean"] = float(np.mean(values))
                    features[f"{feature_name}_std"] = float(np.std(values))
                    features[f"{feature_name}_trend"] = float(np.polyfit(
                        range(len(values)), values, 1
                    )[0]) if len(np.unique(values)) > 1 else 0.0
        
        # Specific domain features
        # Furnace efficiency index
        furnace_temp = None
        furnace_power = None
        
        for sensor_name, window in self.sensor_windows.items():
            if "furnace_temperature" in sensor_name and len(window) >= 1:
                furnace_temp = [v for t, v in window][-1]
            elif "furnace_power" in sensor_name and len(window) >= 1:
                furnace_power = [v for t, v in window][-1]
        
        if furnace_temp is not None and furnace_power is not None and furnace_power > 0:
            efficiency_index = furnace_temp / furnace_power
            features["furnace_efficiency_index"] = float(efficiency_index)
        
        # Quality prediction index
        quality_score = None
        defect_count = None
        
        for sensor_name, window in self.sensor_windows.items():
            if "quality_score" in sensor_name and len(window) >= 1:
                quality_score = [v for t, v in window][-1]
            elif "defect_count" in sensor_name and len(window) >= 1:
                defect_count = [v for t, v in window][-1]
        
        if quality_score is not None:
            features["quality_prediction_index"] = float(quality_score)
        
        if defect_count is not None:
            features["defect_rate_prediction"] = float(defect_count / max(1, len(self.timestamp_window)))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names"""
        # This would be dynamically generated in a real implementation
        base_names = list(self.sensor_windows.keys()) + list(self.derived_features.keys())
        feature_names = []
        
        for base_name in base_names:
            # Add statistical features
            for stat in self.feature_definitions["statistical"]:
                feature_names.append(f"{base_name}_{stat}")
            
            # Add temporal features
            for temp in self.feature_definitions["temporal"]:
                feature_names.append(f"{base_name}_{temp}")
            
            # Add frequency features
            for freq in self.feature_definitions["frequency"]:
                feature_names.append(f"{base_name}_{freq}")
        
        # Add cross-sensor features
        valid_sensors = [name for name, window in self.sensor_windows.items() if len(window) >= 10]
        for i, sensor1 in enumerate(valid_sensors):
            for j, sensor2 in enumerate(valid_sensors):
                if i < j:
                    for cross in self.feature_definitions["cross_sensor"]:
                        feature_names.append(f"{sensor1}_{sensor2}_{cross}")
        
        # Add domain features
        for domain in self.feature_definitions["domain_specific"]:
            feature_names.append(domain)
        
        return feature_names
    
    def reset_windows(self):
        """Reset all data windows"""
        self.sensor_windows.clear()
        self.timestamp_window.clear()
        self.derived_features.clear()
        logger.info("🔄 Feature extractor windows reset")


async def main_example():
    """Example usage of Real-Time Feature Extractor"""
    
    async def feature_callback(features):
        """Callback for extracted features"""
        print(f"\n📊 Features extracted at {features.get('timestamp', 'unknown')}")
        print(f"   Total features: {sum(features.get('feature_counts', {}).values())}")
        print(f"   Feature types: {list(features.get('feature_counts', {}).keys())}")
        
        # Show some example features
        example_features = []
        for key, value in features.items():
            if key not in ['timestamp', 'extraction_time', 'feature_counts', 'window_size', 'data_age_seconds']:
                example_features.append(f"{key}: {value:.4f}")
                if len(example_features) >= 5:
                    break
        
        if example_features:
            print(f"   Sample features: {', '.join(example_features)}")
    
    # Create feature extractor
    extractor = RealTimeFeatureExtractor(feature_callback=feature_callback)
    
    # Simulate sensor data stream
    print("🔄 Simulating sensor data stream for feature extraction...")
    
    # Generate sample sensor data
    for i in range(20):
        timestamp = datetime.utcnow()
        
        sensor_data = {
            "timestamp": timestamp.isoformat(),
            "production_line": "Line_A",
            "sensors": {
                "furnace": {
                    "temperature": 1500 + (50 * (0.5 - np.random.random())),
                    "pressure": 15 + (3 * (0.5 - np.random.random())),
                    "melt_level": 2500 + (200 * (0.5 - np.random.random()))
                },
                "forming": {
                    "belt_speed": 150 + (30 * (0.5 - np.random.random())),
                    "mold_temperature": 320 + (40 * (0.5 - np.random.random())),
                    "pressure": 50 + (10 * (0.5 - np.random.random()))
                },
                "quality": {
                    "defect_count": int(5 * np.random.random()),
                    "quality_score": 0.95 + (0.05 * (0.5 - np.random.random()))
                }
            }
        }
        
        # Update extractor
        await extractor.update_with_sensor_data(sensor_data)
        
        # Extract features every 5 updates
        if i % 5 == 4:
            features = extractor.extract_features()
            print(f"📈 Extracted {len(features) - 5} features (excluding metadata)")
        
        await asyncio.sleep(0.1)
    
    # Show final feature names
    feature_names = extractor.get_feature_names()
    print(f"\n📋 Total possible feature names: {len(feature_names)}")
    print(f"   Example features: {feature_names[:10]}")


if __name__ == "__main__":
    import numpy as np
    from scipy import stats
    asyncio.run(main_example())