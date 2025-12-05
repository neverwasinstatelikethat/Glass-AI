"""
Domain-Specific Feature Engineering for Glass Production
Physics-based and process-specific features for glass manufacturing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import numpy as np
from collections import deque, defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlassProductionFeatureExtractor:
    """Domain-specific feature extraction for glass production processes"""
    
    def __init__(
        self,
        feature_callback: Optional[Callable] = None
    ):
        self.feature_callback = feature_callback
        
        # Process parameters and constants
        self.process_constants = {
            "glass_transition_temp": 550,  # °C
            "working_temp_min": 1000,      # °C
            "working_temp_max": 1600,      # °C
            "forming_temp_optimal": 350,   # °C
            "annealing_temp_range": (500, 650),  # °C
            "density_glass": 2500,         # kg/m³
            "specific_heat_glass": 840,    # J/kg·K
            "thermal_conductivity": 1.4,   # W/m·K
        }
        
        # Data buffers for temporal analysis
        self.process_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # State tracking
        self.process_state = {
            "melting_efficiency": 0.0,
            "forming_stability": 0.0,
            "annealing_quality": 0.0,
            "energy_efficiency": 0.0,
            "defect_prediction": 0.0
        }
    
    async def update_with_process_data(self, process_data: Dict[str, Any]):
        """Update domain features with new process data"""
        try:
            timestamp = datetime.fromisoformat(process_data.get("timestamp", datetime.utcnow().isoformat()))
            
            # Flatten nested process data
            flat_data = self._flatten_process_data(process_data)
            
            # Update process windows
            for key, value in flat_data.items():
                if value is not None and not np.isnan(value):
                    self.process_windows[key].append((timestamp, float(value)))
            
            # Compute domain-specific features
            features = await self.compute_domain_features(flat_data, timestamp)
            
            # Update process state
            self._update_process_state(features)
            
            # Call callback if provided
            if self.feature_callback:
                await self.feature_callback(features)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error updating with process data: {e}")
            return {}
    
    def _flatten_process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested process data structure"""
        flat_data = {}
        
        def _flatten_recursive(obj, prefix=""):
            for key, value in obj.items():
                new_key = f"{prefix}{key}" if prefix else key
                
                if isinstance(value, dict):
                    if "value" in value:
                        flat_data[new_key] = value["value"]
                    else:
                        _flatten_recursive(value, f"{new_key}_")
                else:
                    flat_data[new_key] = value
        
        _flatten_recursive(data)
        return flat_data
    
    async def compute_domain_features(self, flat_data: Dict[str, float], timestamp: datetime) -> Dict[str, float]:
        """Compute all domain-specific features"""
        try:
            features = {
                "timestamp": timestamp.isoformat(),
                "computation_time": datetime.utcnow().isoformat()
            }
            
            # Melting process features
            melting_features = self._compute_melting_features(flat_data)
            features.update(melting_features)
            
            # Forming process features
            forming_features = self._compute_forming_features(flat_data)
            features.update(forming_features)
            
            # Annealing process features
            annealing_features = self._compute_annealing_features(flat_data)
            features.update(annealing_features)
            
            # Quality prediction features
            quality_features = self._compute_quality_features(flat_data)
            features.update(quality_features)
            
            # Energy efficiency features
            energy_features = self._compute_energy_features(flat_data)
            features.update(energy_features)
            
            # Defect prediction features
            defect_features = self._compute_defect_features(flat_data)
            features.update(defect_features)
            
            # Process stability features
            stability_features = self._compute_stability_features()
            features.update(stability_features)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error computing domain features: {e}")
            return {
                "timestamp": timestamp.isoformat(),
                "error": str(e)
            }
    
    def _compute_melting_features(self, flat_data: Dict[str, float]) -> Dict[str, float]:
        """Compute melting process features"""
        features = {}
        
        furnace_temp = flat_data.get("furnace_temperature")
        melt_level = flat_data.get("furnace_melt_level")
        furnace_power = flat_data.get("furnace_power")
        batch_flow = flat_data.get("batch_flow")
        
        if furnace_temp is not None:
            # Temperature-based features
            features["melting_temperature_index"] = float(furnace_temp / self.process_constants["working_temp_max"])
            
            # Temperature deviation from optimal
            optimal_melt_temp = (self.process_constants["working_temp_min"] + 
                               self.process_constants["working_temp_max"]) / 2
            features["melting_temp_deviation"] = float(abs(furnace_temp - optimal_melt_temp))
            
            # Overheating risk
            features["overheating_risk"] = float(max(0, furnace_temp - self.process_constants["working_temp_max"]) / 100)
            
            # Underheating risk
            features["underheating_risk"] = float(max(0, self.process_constants["working_temp_min"] - furnace_temp) / 100)
        
        if melt_level is not None:
            # Melt level stability
            features["melt_level_stability"] = float(1 - abs(melt_level - 2500) / 1000)  # Assuming 2500mm is optimal
            
            # Melt level trend
            if len(self.process_windows["furnace_melt_level"]) >= 10:
                levels = np.array([v for t, v in self.process_windows["furnace_melt_level"][-10:]])
                if len(levels) >= 2:
                    trend = np.polyfit(range(len(levels)), levels, 1)[0]
                    features["melt_level_trend"] = float(trend)
        
        # Melting efficiency (requires multiple parameters)
        if all(v is not None for v in [furnace_temp, furnace_power, batch_flow]):
            # Simplified efficiency model
            base_efficiency = 0.7  # Base efficiency
            temp_factor = 1 - abs(furnace_temp - 1400) / 1000  # Optimal at 1400°C
            power_factor = min(1, furnace_power / 80)  # Assume 80% power is optimal
            flow_factor = 1 - abs(batch_flow - 2000) / 2000  # Assume 2000 kg/hr is optimal
            
            efficiency = base_efficiency * temp_factor * power_factor * flow_factor
            features["melting_efficiency"] = float(max(0, min(1, efficiency)))
        
        return features
    
    def _compute_forming_features(self, flat_data: Dict[str, float]) -> Dict[str, float]:
        """Compute forming process features"""
        features = {}
        
        mold_temp = flat_data.get("forming_mold_temperature")
        belt_speed = flat_data.get("forming_belt_speed")
        forming_pressure = flat_data.get("forming_pressure")
        furnace_temp = flat_data.get("furnace_temperature")
        
        if mold_temp is not None:
            # Mold temperature control
            temp_diff = abs(mold_temp - self.process_constants["forming_temp_optimal"])
            features["mold_temp_control"] = float(1 - temp_diff / 100)  # Assume ±100°C acceptable
            
            # Thermal shock risk
            features["thermal_shock_risk"] = float(max(0, abs(mold_temp - 350) - 50) / 50)
        
        if belt_speed is not None:
            # Speed stability
            features["forming_speed_stability"] = float(1 - abs(belt_speed - 150) / 100)  # Optimal 150 m/min
            
            # Speed trend
            if len(self.process_windows["forming_belt_speed"]) >= 10:
                speeds = np.array([v for t, v in self.process_windows["forming_belt_speed"][-10:]])
                if len(speeds) >= 2:
                    trend = np.polyfit(range(len(speeds)), speeds, 1)[0]
                    features["forming_speed_trend"] = float(trend)
        
        # Forming temperature gradient
        if furnace_temp is not None and mold_temp is not None:
            temp_gradient = furnace_temp - mold_temp
            features["forming_temp_gradient"] = float(temp_gradient)
            
            # Optimal gradient range (simplified)
            optimal_gradient = 1150  # 1500°C furnace - 350°C mold
            features["temp_gradient_deviation"] = float(abs(temp_gradient - optimal_gradient) / optimal_gradient)
        
        # Forming quality index (requires multiple parameters)
        if all(v is not None for v in [mold_temp, belt_speed, forming_pressure]):
            # Simplified quality model
            temp_factor = 1 - abs(mold_temp - self.process_constants["forming_temp_optimal"]) / 50
            speed_factor = 1 - abs(belt_speed - 150) / 75
            pressure_factor = 1 - abs(forming_pressure - 50) / 25  # Optimal 50 bar
            
            quality_index = temp_factor * speed_factor * pressure_factor
            features["forming_quality_index"] = float(max(0, min(1, quality_index)))
        
        return features
    
    def _compute_annealing_features(self, flat_data: Dict[str, float]) -> Dict[str, float]:
        """Compute annealing process features"""
        features = {}
        
        anneal_zone1 = flat_data.get("annealing_temp_zone1")
        anneal_zone2 = flat_data.get("annealing_temp_zone2")
        anneal_zone3 = flat_data.get("annealing_temp_zone3")
        mold_temp = flat_data.get("forming_mold_temperature")
        
        # Zone temperature control
        if anneal_zone1 is not None:
            zone1_optimal = 600  # Example optimal temperature
            features["anneal_zone1_control"] = float(1 - abs(anneal_zone1 - zone1_optimal) / 100)
        
        if anneal_zone2 is not None:
            zone2_optimal = 550  # Example optimal temperature
            features["anneal_zone2_control"] = float(1 - abs(anneal_zone2 - zone2_optimal) / 100)
        
        if anneal_zone3 is not None:
            zone3_optimal = 500  # Example optimal temperature
            features["anneal_zone3_control"] = float(1 - abs(anneal_zone3 - zone3_optimal) / 100)
        
        # Temperature profile consistency
        zones = [t for t in [anneal_zone1, anneal_zone2, anneal_zone3] if t is not None]
        if len(zones) >= 2:
            # Check if temperatures are in descending order
            is_descending = all(zones[i] >= zones[i+1] for i in range(len(zones)-1))
            features["annealing_profile_consistency"] = float(is_descending)
            
            # Temperature gradient stability
            gradients = [zones[i] - zones[i+1] for i in range(len(zones)-1)]
            if gradients:
                features["annealing_gradient_stability"] = float(1 / (1 + np.std(gradients)))
        
        # Cooling rate estimation
        if (mold_temp is not None and len(zones) >= 1):
            cooling_rate = mold_temp - zones[0]  # Simplified cooling rate
            features["estimated_cooling_rate"] = float(cooling_rate)
            
            # Optimal cooling rate range
            optimal_cooling = 300  # Example optimal cooling rate
            features["cooling_rate_deviation"] = float(abs(cooling_rate - optimal_cooling) / optimal_cooling)
        
        return features
    
    def _compute_quality_features(self, flat_data: Dict[str, float]) -> Dict[str, float]:
        """Compute quality prediction features"""
        features = {}
        
        # Current quality score
        quality_score = flat_data.get("quality_score")
        if quality_score is not None:
            features["current_quality_score"] = float(quality_score)
        
        # Defect count trend
        defect_count = flat_data.get("defect_count")
        if defect_count is not None:
            features["current_defect_count"] = float(defect_count)
            
            # Defect rate trend
            if len(self.process_windows["defect_count"]) >= 10:
                defects = np.array([v for t, v in self.process_windows["defect_count"][-10:]])
                if len(defects) >= 2:
                    trend = np.polyfit(range(len(defects)), defects, 1)[0]
                    features["defect_trend"] = float(trend)
        
        # Predictive quality index
        temp_stability = self._get_stability_score("furnace_temperature")
        speed_stability = self._get_stability_score("forming_belt_speed")
        pressure_stability = self._get_stability_score("forming_pressure")
        
        quality_prediction = (temp_stability + speed_stability + pressure_stability) / 3
        features["quality_prediction_index"] = float(quality_prediction)
        
        # Quality risk factors
        overheating_risk = features.get("overheating_risk", 0)
        thermal_shock_risk = features.get("thermal_shock_risk", 0)
        quality_risk = (overheating_risk + thermal_shock_risk) / 2
        features["quality_risk_index"] = float(quality_risk)
        
        return features
    
    def _compute_energy_features(self, flat_data: Dict[str, float]) -> Dict[str, float]:
        """Compute energy efficiency features"""
        features = {}
        
        furnace_power = flat_data.get("furnace_power")
        belt_speed = flat_data.get("forming_belt_speed")
        production_rate = flat_data.get("production_rate")
        
        # Specific energy consumption (SEC)
        if all(v is not None for v in [furnace_power, production_rate]) and production_rate > 0:
            sec = furnace_power / production_rate  # kWh per unit
            features["specific_energy_consumption"] = float(sec)
            
            # Energy efficiency index (lower SEC is better)
            # Assume optimal SEC is 0.1 kWh/unit
            efficiency = max(0, 1 - sec / 0.1) if sec <= 0.2 else max(0, (0.3 - sec) / 0.1)
            features["energy_efficiency_index"] = float(efficiency)
        
        # Power utilization
        if furnace_power is not None:
            # Assume maximum power is 100%
            power_util = furnace_power / 100
            features["power_utilization"] = float(power_util)
        
        # Speed-power relationship
        if all(v is not None for v in [belt_speed, furnace_power]) and belt_speed > 0:
            power_per_speed = furnace_power / belt_speed
            features["power_per_speed_ratio"] = float(power_per_speed)
        
        return features
    
    def _compute_defect_features(self, flat_data: Dict[str, float]) -> Dict[str, float]:
        """Compute defect prediction features"""
        features = {}
        
        # Current defect count
        defect_count = flat_data.get("defect_count")
        if defect_count is not None:
            features["current_defect_count"] = float(defect_count)
        
        # Temperature-related defects
        furnace_temp = flat_data.get("furnace_temperature")
        if furnace_temp is not None:
            # High temperature defects (bubbles, inclusions)
            high_temp_defects = max(0, (furnace_temp - 1600) / 100)
            features["high_temp_defect_risk"] = float(high_temp_defects)
            
            # Low temperature defects (cracks, stresses)
            low_temp_defects = max(0, (1400 - furnace_temp) / 100)
            features["low_temp_defect_risk"] = float(low_temp_defects)
        
        # Speed-related defects
        belt_speed = flat_data.get("forming_belt_speed")
        if belt_speed is not None:
            # High speed defects (surface imperfections)
            high_speed_defects = max(0, (belt_speed - 200) / 50)
            features["high_speed_defect_risk"] = float(high_speed_defects)
        
        # Pressure-related defects
        forming_pressure = flat_data.get("forming_pressure")
        if forming_pressure is not None:
            # Pressure variation defects
            pressure_variation = abs(forming_pressure - 50) / 25  # Deviation from 50 bar
            features["pressure_defect_risk"] = float(pressure_variation)
        
        # Combined defect prediction
        risk_factors = [
            features.get("high_temp_defect_risk", 0),
            features.get("low_temp_defect_risk", 0),
            features.get("high_speed_defect_risk", 0),
            features.get("pressure_defect_risk", 0)
        ]
        
        if risk_factors:
            defect_prediction = np.mean(risk_factors)
            features["defect_prediction_index"] = float(defect_prediction)
        
        return features
    
    def _compute_stability_features(self) -> Dict[str, float]:
        """Compute process stability features"""
        features = {}
        
        # Temperature stability
        temp_stability = self._get_stability_score("furnace_temperature")
        features["temperature_stability"] = float(temp_stability)
        
        # Speed stability
        speed_stability = self._get_stability_score("forming_belt_speed")
        features["speed_stability"] = float(speed_stability)
        
        # Pressure stability
        pressure_stability = self._get_stability_score("forming_pressure")
        features["pressure_stability"] = float(pressure_stability)
        
        # Overall process stability
        overall_stability = (temp_stability + speed_stability + pressure_stability) / 3
        features["overall_process_stability"] = float(overall_stability)
        
        return features
    
    def _get_stability_score(self, sensor_name: str, window_size: int = 20) -> float:
        """Calculate stability score for a sensor"""
        try:
            window = self.process_windows[sensor_name]
            if len(window) < 5:
                return 0.5  # Neutral stability score
            
            values = np.array([v for t, v in window][-window_size:])
            values = values[~np.isnan(values)]
            
            if len(values) < 3:
                return 0.5
            
            # Coefficient of variation (lower is more stable)
            mean_val = np.mean(values)
            if mean_val == 0:
                return 0.5
            
            cv = np.std(values) / abs(mean_val)
            
            # Convert to stability score (0-1, where 1 is most stable)
            stability_score = 1 / (1 + cv)
            return float(stability_score)
            
        except Exception as e:
            logger.debug(f"⚠️ Error calculating stability for {sensor_name}: {e}")
            return 0.5
    
    def _update_process_state(self, features: Dict[str, float]):
        """Update internal process state tracking"""
        try:
            # Update melting efficiency
            if "melting_efficiency" in features:
                self.process_state["melting_efficiency"] = features["melting_efficiency"]
            
            # Update forming stability
            if "forming_quality_index" in features:
                self.process_state["forming_stability"] = features["forming_quality_index"]
            
            # Update annealing quality
            if "annealing_profile_consistency" in features:
                self.process_state["annealing_quality"] = features["annealing_profile_consistency"]
            
            # Update energy efficiency
            if "energy_efficiency_index" in features:
                self.process_state["energy_efficiency"] = features["energy_efficiency_index"]
            
            # Update defect prediction
            if "defect_prediction_index" in features:
                self.process_state["defect_prediction"] = features["defect_prediction_index"]
        
        except Exception as e:
            logger.error(f"❌ Error updating process state: {e}")
    
    def get_process_state(self) -> Dict[str, float]:
        """Get current process state"""
        return self.process_state.copy()
    
    def get_process_health_score(self) -> float:
        """Calculate overall process health score (0-1)"""
        try:
            # Weighted average of process state metrics
            weights = {
                "melting_efficiency": 0.2,
                "forming_stability": 0.25,
                "annealing_quality": 0.2,
                "energy_efficiency": 0.15,
                "defect_prediction": 0.2  # Inverted (lower defect risk = higher health)
            }
            
            health_components = []
            weight_sum = 0
            
            for metric, weight in weights.items():
                if metric in self.process_state:
                    value = self.process_state[metric]
                    # Invert defect prediction (lower risk = higher health)
                    if metric == "defect_prediction":
                        value = 1 - value
                    
                    health_components.append(value * weight)
                    weight_sum += weight
            
            if weight_sum > 0:
                health_score = sum(health_components) / weight_sum
                return float(max(0, min(1, health_score)))
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"❌ Error calculating process health score: {e}")
            return 0.5
    
    def reset_process_state(self):
        """Reset process state tracking"""
        self.process_state = {
            "melting_efficiency": 0.0,
            "forming_stability": 0.0,
            "annealing_quality": 0.0,
            "energy_efficiency": 0.0,
            "defect_prediction": 0.0
        }
        self.process_windows.clear()
        logger.info("🔄 Process state and windows reset")


async def main_example():
    """Example usage of Glass Production Feature Extractor"""
    
    async def feature_callback(features):
        """Callback for extracted features"""
        print(f"\n🏭 Glass Production Features at {features.get('timestamp', 'unknown')}")
        print(f"   Features computed: {len(features) - 2}")  # Exclude timestamps
        
        # Show key domain features
        key_features = [
            "melting_efficiency", "forming_quality_index", "energy_efficiency_index",
            "defect_prediction_index", "overall_process_stability"
        ]
        
        for feature in key_features:
            if feature in features:
                print(f"   {feature}: {features[feature]:.4f}")
        
        # Show process state
        health_score = extractor.get_process_health_score()
        print(f"   🏥 Process Health Score: {health_score:.4f}")
    
    # Create feature extractor
    extractor = GlassProductionFeatureExtractor(feature_callback=feature_callback)
    
    # Simulate process data stream
    print("🔄 Simulating glass production process data...")
    
    # Generate realistic process data
    for i in range(50):
        timestamp = datetime.utcnow()
        
        process_data = {
            "timestamp": timestamp.isoformat(),
            "production_line": "Line_A",
            "process_data": {
                "furnace": {
                    "temperature": 1500 + (50 * (0.5 - np.random.random())),  # 1450-1550°C
                    "melt_level": 2500 + (200 * (0.5 - np.random.random())),   # 2300-2700mm
                    "power": 80 + (20 * (0.5 - np.random.random())),           # 70-90%
                },
                "forming": {
                    "mold_temperature": 350 + (30 * (0.5 - np.random.random())), # 320-380°C
                    "belt_speed": 150 + (30 * (0.5 - np.random.random())),      # 120-180 m/min
                    "pressure": 50 + (10 * (0.5 - np.random.random())),         # 40-60 bar
                },
                "annealing": {
                    "zone1": 600 + (30 * (0.5 - np.random.random())),           # 570-630°C
                    "zone2": 550 + (30 * (0.5 - np.random.random())),           # 520-580°C
                    "zone3": 500 + (30 * (0.5 - np.random.random())),           # 470-530°C
                },
                "quality": {
                    "defect_count": int(3 * np.random.random()),                # 0-3 defects
                    "quality_score": 0.92 + (0.08 * (0.5 - np.random.random())), # 0.88-1.00
                    "production_rate": 1200 + (200 * (0.5 - np.random.random())) # 1100-1300 units/hr
                }
            }
        }
        
        # Update extractor
        await extractor.update_with_process_data(process_data)
        
        # Show process state every 10 iterations
        if i % 10 == 9:
            state = extractor.get_process_state()
            health = extractor.get_process_health_score()
            print(f"\n📊 Process State (Iteration {i+1}):")
            for metric, value in state.items():
                print(f"   {metric}: {value:.4f}")
            print(f"   Overall Health: {health:.4f}")
        
        await asyncio.sleep(0.2)
    
    # Final process state
    print(f"\n🏁 Final Process State:")
    final_state = extractor.get_process_state()
    for metric, value in final_state.items():
        print(f"   {metric}: {value:.4f}")


if __name__ == "__main__":
    import numpy as np
    asyncio.run(main_example())