"""
WebSocket Broadcaster for Real-Time Dashboard Updates
ML-Driven approach: Simulates sensors, passes data to ML models for predictions/recommendations

Architecture:
- Sensor Simulation Layer: Generates realistic sensor data
- ML Prediction Layer: LSTM (defect prediction), GNN (sensor anomaly detection)
- RL Optimization Layer: PPO agent generates recommendations
- Alert Generation: Based on ML predictions, not hardcoded rules
"""

import asyncio
import logging
from typing import Dict, Any, List, Set, Optional
from datetime import datetime
import json
import os
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict

# Import Knowledge Graph for real-time enrichment
try:
    from knowledge_graph.causal_graph import EnhancedGlassProductionKnowledgeGraph
    from feature_engineering.real_time_features import RealTimeFeatureExtractor
    from explainability.feature_attribution import EnhancedGlassProductionExplainer
    KG_AVAILABLE = True
    FEATURE_ENGINEERING_AVAILABLE = True
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    KG_AVAILABLE = False
    EnhancedGlassProductionKnowledgeGraph = None
    FEATURE_ENGINEERING_AVAILABLE = False
    EXPLAINABILITY_AVAILABLE = False
    RealTimeFeatureExtractor = None
    EnhancedGlassProductionExplainer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Knowledge Graph instance for enrichment
_knowledge_graph: Optional['EnhancedGlassProductionKnowledgeGraph'] = None

def get_knowledge_graph() -> Optional['EnhancedGlassProductionKnowledgeGraph']:
    """Get or create Knowledge Graph instance for enrichment"""
    global _knowledge_graph
    
    if not KG_AVAILABLE:
        return None
    
    if _knowledge_graph is None:
        try:
            is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
            
            if is_docker:
                neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
                redis_host = os.getenv("REDIS_HOST", "redis")
            else:
                neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                redis_host = os.getenv("REDIS_HOST", "localhost")
            
            _knowledge_graph = EnhancedGlassProductionKnowledgeGraph(
                uri=neo4j_uri,
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "neo4jpassword"),
                redis_host=redis_host,
                redis_port=int(os.getenv("REDIS_PORT", "6379"))
            )
            logger.info("🧠 Knowledge Graph initialized for real-time enrichment")
        except Exception as e:
            logger.warning(f"⚠️ KG initialization failed: {e}")
            _knowledge_graph = None
    
    return _knowledge_graph

# Defect type labels (matches LSTM output order)
DEFECT_TYPES = ["crack", "bubble", "chip", "stain", "cloudiness", "deformation"]


class MLInferencePipeline:
    """
    ML Inference Pipeline for real-time predictions
    Manages ONNX models (LSTM, GNN) and RL agent for recommendations
    
    This class encapsulates all ML logic, keeping sensor simulation separate.
    Sensor data flows IN -> ML models produce predictions/recommendations OUT
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.lstm_session = None
        self.gnn_session = None
        self.rl_agent = None
        self.feature_extractor = None
        self.explainer = None
        self.is_loaded = False
        
        # Sequence buffer for LSTM (maintains temporal context)
        self.sequence_buffer: List[np.ndarray] = []
        self.sequence_length = 30  # 30 timesteps for LSTM
        self.feature_dim = 20  # Number of input features
        
        # Load models on init
        self._load_models()
        self._initialize_feature_extraction()
    
    def _load_models(self):
        """Load all ONNX models and RL agent"""
        try:
            import onnxruntime as ort
            
            # Load LSTM model
            lstm_path = os.path.join(self.models_dir, "lstm_predictor", "lstm_model.onnx")
            if os.path.exists(lstm_path):
                self.lstm_session = ort.InferenceSession(lstm_path)
                logger.info(f"✅ LSTM model loaded: {lstm_path}")
            else:
                logger.warning(f"⚠️ LSTM model not found: {lstm_path}")
            
            # Load GNN model
            gnn_path = os.path.join(self.models_dir, "gnn_sensor_network", "gnn_model.onnx")
            if os.path.exists(gnn_path):
                self.gnn_session = ort.InferenceSession(gnn_path)
                logger.info(f"✅ GNN model loaded: {gnn_path}")
            else:
                logger.warning(f"⚠️ GNN model not found: {gnn_path}")
            
            # Load RL agent
            try:
                from reinforcement_learning.ppo_optimizer import GlassProductionPPO, PPOConfig
                
                # Создаем конфигурацию с правильными параметрами, соответствующими обучению
                config = PPOConfig()
                config.hidden_size = 128  # Соответствует обученной модели
                
                self.rl_agent = GlassProductionPPO(
                    state_dim=5,  # Должно совпадать с обучением (5 параметров из среды)
                    continuous_action_dim=3,
                    discrete_action_dims=[5, 5, 5],
                    config=config
                )
                logger.info("✅ RL Agent loaded")
            except Exception as e:
                logger.warning(f"⚠️ RL Agent not loaded: {e}")
            
            self.is_loaded = self.lstm_session is not None or self.gnn_session is not None
            
        except Exception as e:
            logger.error(f"❌ Failed to load ML models: {e}")
            self.is_loaded = False
    
    def _initialize_feature_extraction(self):
        """Initialize feature extraction and explainability components"""
        try:
            if FEATURE_ENGINEERING_AVAILABLE:
                self.feature_extractor = RealTimeFeatureExtractor(window_size=60)
                logger.info("✅ Feature extractor initialized")
            
            # Note: Explainer requires actual models which may not be available at init time
            # Will be initialized when models are loaded
            
        except Exception as e:
            logger.warning(f"⚠️ Feature extraction initialization failed: {e}")
            self.feature_extractor = None
    
    async def _state_to_features(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Convert production state to feature vector for LSTM
        Uses real-time feature extraction instead of hardcoded mapping
        """
        # Use feature extractor if available
        if self.feature_extractor is not None:
            try:
                # Use timezone-aware UTC time for consistency
                from datetime import timezone
                # Convert state to sensor data format expected by feature extractor
                sensor_data = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sensors": {
                        "furnace": {
                            "temperature": state.get("furnace_temperature", 1520),
                            "pressure": state.get("furnace_pressure", 15),
                            "melt_level": state.get("melt_level", 2500),
                            "o2_percent": state.get("o2_level", 21),
                            "co2_percent": state.get("co2_level", 0.04)
                        },
                        "forming": {
                            "belt_speed": state.get("belt_speed", 150),
                            "mold_temperature": state.get("mold_temp", 320),
                            "pressure": state.get("forming_pressure", 50)
                        },
                        "annealing": {
                            "temperature": state.get("annealing_temp", 580)
                        },
                        "process": {
                            "batch_flow": state.get("raw_material_flow", 100)
                        }
                    }
                }
                
                # Логируем структуру данных перед отправкой
                logger.debug(f"Sensor data for feature extraction:")
                for sensor_group, values in sensor_data['sensors'].items():
                    for key, value in values.items():
                        logger.debug(f"  {sensor_group}.{key}: {value} (type: {type(value)})")
                
                # Update feature extractor with current data
                await self.feature_extractor.update_with_sensor_data(sensor_data)
                
                # Extract features
                features_result = self.feature_extractor.extract_features()
                
                # Логируем полную структуру результата
                logger.debug(f"Full features result structure:")
                logger.debug(f"  Keys in features_result: {list(features_result.keys())}")
                
                # Determine if we have nested structure or flat structure
                if "features" in features_result and "metadata" in features_result:
                    # New structure: features are in "features" key
                    features = features_result.get("features", {})
                    metadata = features_result.get("metadata", {})
                    logger.debug(f"Using nested structure with {len(features)} features")
                else:
                    # Old structure: features are the dict itself
                    features = features_result
                    logger.debug(f"Using flat structure with {len(features)} items")
                
                # Filter out non-numeric features (like timestamps, metadata)
                numeric_features = {}
                skipped_keys = []
                
                for key, value in features.items():
                    # Skip keys that are clearly not features
                    if key in ["timestamp", "extraction_time", "error", "computation_time"]:
                        skipped_keys.append(f"{key} (metadata)")
                        continue
                    
                    # Skip feature_counts and other dicts
                    if isinstance(value, dict):
                        skipped_keys.append(f"{key} (dict)")
                        continue
                    
                    # Try to convert to float
                    try:
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            numeric_value = float(value)
                            if not np.isnan(numeric_value):
                                numeric_features[key] = numeric_value
                            else:
                                skipped_keys.append(f"{key} (NaN)")
                        elif isinstance(value, str):
                            # Skip strings that look like timestamps
                            if (len(value) > 10 and 
                                (value.count('-') >= 2 or value.count(':') >= 2 or 'T' in value or '+' in value or 'Z' in value)):
                                skipped_keys.append(f"{key} (timestamp string)")
                                continue
                            # Try to convert other strings to float
                            numeric_value = float(value)
                            if not np.isnan(numeric_value):
                                numeric_features[key] = numeric_value
                            else:
                                skipped_keys.append(f"{key} (converted to NaN)")
                        else:
                            # For other types, try to convert to float
                            numeric_value = float(value)
                            if not np.isnan(numeric_value):
                                numeric_features[key] = numeric_value
                            else:
                                skipped_keys.append(f"{key} (converted to NaN)")
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Skipping feature {key} with value {value} (type: {type(value)}): {e}")
                        skipped_keys.append(f"{key} (conversion error)")
                        continue
                
                # Log what we found
                logger.debug(f"Total items in features: {len(features)}")
                logger.debug(f"Numeric features extracted: {len(numeric_features)}")
                logger.debug(f"Skipped items: {len(skipped_keys)}")
                if skipped_keys:
                    logger.debug(f"First 10 skipped keys: {skipped_keys[:10]}")
                if numeric_features:
                    logger.debug(f"First 5 numeric features: {list(numeric_features.items())[:5]}")
                
                if not numeric_features:
                    logger.warning("⚠️ No numeric features extracted, using fallback")
                    # Fallback to original approach
                    raise ValueError("No numeric features available")
                
                # Get feature names and sort for consistency
                feature_names = sorted(numeric_features.keys())
                
                # Create feature vector
                feature_vector = np.array([numeric_features[name] for name in feature_names], dtype=np.float32)
                
                # Pad or truncate to expected size (20 features)
                if len(feature_vector) < self.feature_dim:
                    padded = np.zeros(self.feature_dim, dtype=np.float32)
                    padded[:len(feature_vector)] = feature_vector
                    logger.debug(f"Padded feature vector from {len(feature_vector)} to {self.feature_dim}")
                    return padded
                elif len(feature_vector) > self.feature_dim:
                    logger.debug(f"Truncating feature vector from {len(feature_vector)} to {self.feature_dim} elements")
                    return feature_vector[:self.feature_dim]
                else:
                    return feature_vector
                    
            except Exception as e:
                logger.warning(f"⚠️ Feature extraction failed: {e}, using fallback")
        
        # Fallback to original hardcoded approach
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # Normalize features to [-1, 1] range based on expected ranges
        # Furnace parameters
        features[0] = (state.get("furnace_temperature", 1520) - 1520) / 100  # Temp deviation
        features[1] = (state.get("furnace_pressure", 15) - 15) / 10  # Pressure
        features[2] = state.get("o2_level", 21) / 25 - 0.5  # O2 normalized
        features[3] = state.get("co2_level", 0.04) * 10  # CO2 level
        
        # Forming parameters
        features[4] = (state.get("belt_speed", 150) - 150) / 50  # Speed deviation
        features[5] = (state.get("mold_temp", 320) - 320) / 100  # Mold temp deviation
        features[6] = (state.get("forming_pressure", 50) - 50) / 30  # Pressure
        
        # Annealing parameters
        features[7] = (state.get("annealing_temp", 580) - 580) / 100  # Annealing temp
        features[8] = (state.get("cooling_rate", 3.5) - 3.5) / 2  # Cooling rate
        
        # Quality indicators
        features[9] = state.get("quality_score", 0.95) - 0.9  # Quality deviation
        features[10] = state.get("defect_rate", 0.02) * 10  # Defect rate scaled
        
        # Sensor statistics (derived)
        features[11] = state.get("temp_variance", 5) / 20  # Temperature variance
        features[12] = state.get("speed_variance", 2) / 10  # Speed variance
        features[13] = state.get("pressure_variance", 1) / 5  # Pressure variance
        
        # Time-based features
        # Use timezone-aware UTC time for consistency
        from datetime import timezone
        current_hour = datetime.now(timezone.utc).hour
        features[14] = np.sin(2 * np.pi * current_hour / 24)  # Hour sin
        features[15] = np.cos(2 * np.pi * current_hour / 24)  # Hour cos
        
        # Production metrics
        features[16] = state.get("production_rate", 150) / 200 - 0.5  # Production rate
        features[17] = state.get("energy_consumption", 500) / 1000 - 0.5  # Energy
        features[18] = state.get("raw_material_flow", 100) / 200 - 0.5  # Material flow
        features[19] = state.get("humidity", 50) / 100 - 0.5  # Humidity
        
        return features
    
    def _state_to_gnn_features(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Convert production state to GNN node features
        Creates 20 nodes x 10 features representing sensor network
        """
        num_nodes = 20
        feature_dim = 10
        node_features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        
        # Create sensor node features
        base_features = [
            state.get("furnace_temperature", 1520) / 1600,
            state.get("furnace_pressure", 15) / 50,
            state.get("belt_speed", 150) / 200,
            state.get("mold_temp", 320) / 400,
            state.get("forming_pressure", 50) / 100,
            state.get("annealing_temp", 580) / 600,
            state.get("cooling_rate", 3.5) / 5,
            state.get("quality_score", 0.95),
            state.get("production_rate", 150) / 200,
            state.get("humidity", 50) / 100
        ]
        
        # Add noise for each node to simulate different sensors
        for i in range(num_nodes):
            noise = np.random.normal(0, 0.02, feature_dim)
            node_features[i] = np.clip(np.array(base_features) + noise, 0, 1)
        
        return node_features
    
    async def update_sequence_buffer(self, state: Dict[str, Any]):
        """Add new state to sequence buffer for LSTM"""
        features = await self._state_to_features(state)
        logger.debug(f"Adding feature vector of size: {len(features) if features is not None else 'None'} to sequence buffer")
        self.sequence_buffer.append(features)
        
        # Keep only last sequence_length samples
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer = self.sequence_buffer[-self.sequence_length:]
        logger.debug(f"Sequence buffer now contains {len(self.sequence_buffer)} elements")
    
    async def predict_defects_lstm(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Use LSTM model to predict defect probabilities
        
        Returns:
            Dict mapping defect_type -> probability (0-1)
        """
        if self.lstm_session is None:
            logger.debug("⚠️ LSTM not loaded, using fallback")
            return self._fallback_defect_prediction(state)
        
        # Update sequence buffer
        await self.update_sequence_buffer(state)
        
        # Need full sequence for prediction
        if len(self.sequence_buffer) < self.sequence_length:
            logger.debug(f"⏳ Buffering: {len(self.sequence_buffer)}/{self.sequence_length}")
            return self._fallback_defect_prediction(state)
        
        try:
            # Prepare input: [1, seq_len, features]
            sequence = np.array(self.sequence_buffer, dtype=np.float32)
            logger.debug(f"Sequence buffer shape: {sequence.shape}, expected: (1, {self.sequence_length}, {self.feature_dim})")
            
            # Debug: Check individual feature vector sizes
            if len(self.sequence_buffer) > 0:
                logger.debug(f"First feature vector size: {len(self.sequence_buffer[0]) if self.sequence_buffer[0] is not None else 'None'}")
                if len(self.sequence_buffer) > 1:
                    logger.debug(f"Second feature vector size: {len(self.sequence_buffer[1]) if self.sequence_buffer[1] is not None else 'None'}")
            
            input_tensor = sequence.reshape(1, self.sequence_length, self.feature_dim)
            
            # Run inference
            input_name = self.lstm_session.get_inputs()[0].name
            output_name = self.lstm_session.get_outputs()[0].name
            
            predictions = self.lstm_session.run([output_name], {input_name: input_tensor})[0]
            
            # Map predictions to defect types
            # LSTM outputs 6 probabilities matching DEFECT_TYPES
            probabilities = {}
            for i, defect_type in enumerate(DEFECT_TYPES):
                # Ensure valid probability range [0, 1]
                prob = float(np.clip(predictions[0][i] if i < len(predictions[0]) else 0, 0, 1))
                probabilities[defect_type] = prob
            
            logger.debug(f"🤖 LSTM predictions: {probabilities}")
            return probabilities
            
        except Exception as e:
            logger.error(f"❌ LSTM prediction failed: {e}")
            return self._fallback_defect_prediction(state)
    
    async def detect_anomaly_gnn(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use GNN model to detect sensor network anomalies
        
        Returns:
            Dict with anomaly classification and affected sensors
        """
        if self.gnn_session is None:
            return {"has_anomaly": False, "confidence": 0.0, "anomaly_sensors": []}
        
        try:
            # Prepare GNN input
            node_features = self._state_to_gnn_features(state)
            
            # Run inference
            input_name = self.gnn_session.get_inputs()[0].name
            output_name = self.gnn_session.get_outputs()[0].name
            
            output = self.gnn_session.run([output_name], {input_name: node_features})[0]
            
            # GNN outputs binary classification [no_anomaly, has_anomaly]
            # Apply softmax to get probabilities
            exp_output = np.exp(output - np.max(output))
            probs = exp_output / exp_output.sum()
            
            has_anomaly = bool(np.argmax(probs) == 1)
            confidence = float(np.max(probs))
            
            # Identify which sensors might be anomalous (based on feature deviations)
            anomaly_sensors = []
            if has_anomaly:
                for i, node in enumerate(node_features):
                    if np.any(np.abs(node - 0.5) > 0.3):  # Significant deviation
                        anomaly_sensors.append(f"sensor_{i+1}")
            
            return {
                "has_anomaly": has_anomaly,
                "confidence": confidence,
                "anomaly_sensors": anomaly_sensors[:5]  # Top 5 anomalous sensors
            }
            
        except Exception as e:
            logger.error(f"❌ GNN anomaly detection failed: {e}")
            return {"has_anomaly": False, "confidence": 0.0, "anomaly_sensors": []}
    
    async def generate_rl_recommendations(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Use RL agent to generate optimization recommendations with ML-driven explainability
        
        Returns:
            List of recommendation dicts with action, confidence, impact
        """
        recommendations = []
        
        if self.rl_agent is None:
            return self._fallback_recommendations(state)
        
        try:
            # Convert state to RL state vector (5-dimensional to match training)
            # Trained model expects 5-dimensional state matching training environment:
            # [temperature, melt_level, quality_score, defects, energy]
            state_vector = np.array([
                float(state.get("furnace_temperature", 1550.0)),    # furnace_temperature (°C)
                float(state.get("melt_level", 2500.0)),             # melt_level
                float(state.get("quality_score", 0.85)),            # quality_score (normalized)
                float(state.get("defect_rate", 0.05)),              # defects (actual defect rate)
                float(state.get("energy_consumption", 450.0))       # energy (actual consumption)
            ], dtype=np.float32)
            
            # Get action from RL agent
            action_result = self.rl_agent.select_action(state_vector, deterministic=True)
            (continuous_action, discrete_actions), log_probs, values, penalty = action_result
            
            # Calculate dynamic confidence based on value function and log probabilities
            # Higher value = higher confidence in recommendation
            base_confidence = float(1.0 / (1.0 + np.exp(-values[0])))  # Sigmoid of value
            
            # Adjust confidence based on how extreme the action is
            action_extremeness = np.mean(np.abs(continuous_action - 0.5))
            confidence_adjustment = 0.1 * action_extremeness  # Up to 10% adjustment
            
            # Use ML-driven feature attribution for explanations if available
            feature_importance = {}
            if self.feature_extractor is not None and EXPLAINABILITY_AVAILABLE:
                try:
                    # Use timezone-aware UTC time for consistency
                    from datetime import timezone
                    # Extract features for explanation
                    sensor_data = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "sensors": {
                            "furnace": {
                                "temperature": state.get("furnace_temperature", 1520),
                                "pressure": state.get("furnace_pressure", 15),
                                "melt_level": state.get("melt_level", 2500),
                                "o2_percent": state.get("o2_level", 21),
                                "co2_percent": state.get("co2_level", 0.04)
                            },
                            "forming": {
                                "belt_speed": state.get("belt_speed", 150),
                                "mold_temperature": state.get("mold_temp", 320),
                                "pressure": state.get("forming_pressure", 50)
                            },
                            "annealing": {
                                "temperature": state.get("annealing_temp", 580)
                            },
                            "process": {
                                "batch_flow": state.get("raw_material_flow", 100)
                            }
                        }
                    }
                    
                    await self.feature_extractor.update_with_sensor_data(sensor_data)
                    features = self.feature_extractor.extract_features()
                    feature_importance = features
                    
                    # Use explainability module for more detailed feature attribution
                    try:
                        if EnhancedGlassProductionExplainer is not None:
                            # Create input vector for explanation
                            feature_vector = self.feature_extractor.get_feature_vector()
                            if feature_vector is not None:
                                feature_names = self.feature_extractor.get_feature_names()
                                
                                # Create a simple model explainer (using dummy model for now)
                                explainer = EnhancedGlassProductionExplainer(
                                    model=None,  # Using without model for feature importance
                                    feature_names=feature_names
                                )
                                
                                # Generate comprehensive explanation
                                explanation_result = explainer.explain_comprehensive(
                                    input_data=feature_vector,
                                    include_shap=False,  # Skip SHAP for performance in real-time
                                    include_lime=True
                                )
                                
                                # Merge with existing feature importance
                                if explanation_result.feature_importance:
                                    feature_importance.update(explanation_result.feature_importance)
                    except Exception as explain_error:
                        logger.debug(f"ℹ️ Detailed explainability not available: {explain_error}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Feature importance extraction failed: {e}")            
            # Convert actions to recommendations with ML-driven contextual information
            # Continuous actions: [furnace_power, belt_speed_adj, mold_temp_adj]
            
            # Furnace temperature adjustment
            if continuous_action[0] > 0.6:
                current_temp = float(state.get("furnace_temperature", 1520))
                # Scale adjustment appropriately for furnace temperature (reasonable range: ±20°C)
                suggested_temp = float(current_temp + (continuous_action[0] - 0.5) * 40)  # Reduced scale further
                
                # Ensure realistic temperature bounds with tighter control
                suggested_temp = max(1450, min(1650, suggested_temp))  # Tighter bounds for glass production                
                # ML-driven explanation based on feature importance
                explanation = "ML-оптимизация температурного режима"
                impact = "Оптимизация теплового режима"
                
                # Enhance explanation with feature importance if available
                # Process feature importance values with proper scaling to prevent overflow
                processed_feature_importance = {}
                if feature_importance:
                    for k, v in feature_importance.items():
                        try:
                            # Ensure v is a numeric value
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                numeric_value = float(v)
                            else:
                                # If not numeric, try to convert to float
                                numeric_value = float(v)
                            
                            # Apply proper scaling instead of hard clipping
                            if abs(numeric_value) > 1000:
                                # Scale down large values proportionally
                                processed_feature_importance[k] = numeric_value / abs(numeric_value) * 1000
                            else:
                                processed_feature_importance[k] = numeric_value
                        except (ValueError, TypeError, Exception):
                            # Skip any values that cause errors
                            continue
                    
                    # Find most important temperature-related features
                    temp_features = {k: v for k, v in processed_feature_importance.items() if "temperature" in k.lower() or "temp" in k.lower()}
                    if temp_features:
                        # Sort by importance and get top features
                        sorted_features = sorted(temp_features.items(), key=lambda x: abs(x[1]), reverse=True)
                        most_important = sorted_features[0] if sorted_features else None
                        top_features = sorted_features[:3]  # Get top 3 features
                        
                        # Create detailed explanation with multiple features and their actual values
                        if top_features:
                            feature_details = ', '.join([f"'{name}' ({abs(value):.2f})" for name, value in top_features])
                            explanation = f"Оптимизация на основе ключевых факторов: {feature_details}"
                            
                            # Adjust impact based on feature importance with more nuanced descriptions
                            max_importance = abs(most_important[1]) if most_important else 0
                            if max_importance > 50:
                                impact = "Значительная оптимизация теплового режима на основе анализа спектральной мощности и температурных градиентов"
                            elif max_importance > 20:
                                impact = "Умеренная корректировка температуры с учетом текущих условий и спектральных характеристик"
                            elif max_importance > 5:
                                impact = "Небольшая корректировка температуры для стабилизации процесса"
                            else:
                                impact = "Минимальная корректировка температуры для поддержания оптимальных условий"
                        else:
                            explanation = "Оптимизация температурного режима на основе комплексного анализа параметров"
                            impact = "Стандартная корректировка температурного режима"
                    else:
                        explanation = "Оптимизация температурного режима на основе комплексного анализа параметров"
                        impact = "Стандартная корректировка температурного режима"                
                recommendations.append({
                    "action": f"Увеличить температуру печи до {suggested_temp:.1f}°C",
                    "parameter": "furnace_temperature",
                    "priority": "HIGH" if continuous_action[0] > 0.75 else "MEDIUM",
                    "confidence": float(np.clip(base_confidence + confidence_adjustment, 0.7, 0.95)),
                    "expected_impact": impact,
                    "current_value": current_temp,
                    "suggested_value": suggested_temp,
                    "justification": explanation,
                    "model": "PPO_Agent",
                    "feature_importance": dict(list(processed_feature_importance.items())[:5]) if feature_importance else {}
                })
            elif continuous_action[0] < 0.4:
                current_temp = float(state.get("furnace_temperature", 1520))
                suggested_temp = float(current_temp - (0.5 - continuous_action[0]) * 40)  # Scale adjustment
                
                # Ensure realistic temperature bounds with tighter control
                suggested_temp = max(1450, min(1650, suggested_temp))
                
                # ML-driven explanation based on feature importance
                explanation = "ML-оптимизация температурного режима"
                impact = "Оптимизация энергопотребления"
                
                # Enhance explanation with feature importance if available
                # Process feature importance values with proper scaling to prevent overflow
                processed_feature_importance = {}
                if feature_importance:
                    for k, v in feature_importance.items():
                        try:
                            # Ensure v is a numeric value
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                numeric_value = float(v)
                            else:
                                # If not numeric, try to convert to float
                                numeric_value = float(v)
                            
                            # Apply proper scaling instead of hard clipping
                            if abs(numeric_value) > 1000:
                                # Scale down large values proportionally
                                processed_feature_importance[k] = numeric_value / abs(numeric_value) * 1000
                            else:
                                processed_feature_importance[k] = numeric_value
                        except (ValueError, TypeError, Exception):
                            # Skip any values that cause errors
                            continue
                    
                    # Find most important temperature-related features
                    temp_features = {k: v for k, v in processed_feature_importance.items() if "temperature" in k.lower() or "temp" in k.lower()}
                    if temp_features:
                        # Sort by importance and get top features
                        sorted_features = sorted(temp_features.items(), key=lambda x: abs(x[1]), reverse=True)
                        most_important = sorted_features[0] if sorted_features else None
                        top_features = sorted_features[:3]  # Get top 3 features
                        
                        # Create detailed explanation with multiple features and their actual values
                        if top_features:
                            feature_details = ', '.join([f"'{name}' ({abs(value):.2f})" for name, value in top_features])
                            explanation = f"Оптимизация на основе ключевых факторов: {feature_details}"
                            
                            # Adjust impact based on feature importance with more nuanced descriptions
                            max_importance = abs(most_important[1]) if most_important else 0
                            if max_importance > 50:
                                impact = "Значительная оптимизация энергопотребления за счет коррекции температурного режима и спектральных характеристик"
                            elif max_importance > 20:
                                impact = "Умеренная корректировка температуры для снижения энергозатрат и оптимизации теплового баланса"
                            elif max_importance > 5:
                                impact = "Небольшая корректировка температуры для экономии энергии"
                            else:
                                impact = "Минимальная корректировка температуры для оптимизации энергопотребления"
                        else:
                            explanation = "Оптимизация температурного режима для снижения энергозатрат"
                            impact = "Стандартная корректировка температуры для экономии энергии"
                    else:
                        explanation = "Оптимизация температурного режима для снижения энергозатрат"
                        impact = "Стандартная корректировка температуры для экономии энергии"                
                recommendations.append({
                    "action": f"Снизить температуру печи до {suggested_temp:.1f}°C",
                    "parameter": "furnace_temperature",
                    "priority": "MEDIUM",
                    "confidence": float(np.clip(base_confidence - confidence_adjustment, 0.6, 0.9)),
                    "expected_impact": impact,
                    "current_value": current_temp,
                    "suggested_value": suggested_temp,
                    "justification": explanation,
                    "model": "PPO_Agent",
                    "feature_importance": dict(list(processed_feature_importance.items())[:5]) if feature_importance else {}
                })
            
            # Belt speed adjustment
            if continuous_action[1] > 0.6:
                current_speed = float(state.get("belt_speed", 150))
                suggested_speed = float(current_speed + (continuous_action[1] - 0.5) * 20)  # Reduced scale
                
                # Ensure realistic speed bounds
                suggested_speed = max(120, min(180, suggested_speed))  # Realistic bounds for glass production
                
                # ML-driven explanation based on feature importance
                explanation = "ML-оптимизация скорости формования"
                impact = "Оптимизация производительности"
                
                # Enhance explanation with feature importance if available
                # Process feature importance values with proper scaling to prevent overflow
                processed_feature_importance = {}
                if feature_importance:
                    for k, v in feature_importance.items():
                        try:
                            # Ensure v is a numeric value
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                numeric_value = float(v)
                            else:
                                # If not numeric, try to convert to float
                                numeric_value = float(v)
                            
                            # Apply proper scaling instead of hard clipping
                            if abs(numeric_value) > 1000:
                                # Scale down large values proportionally
                                processed_feature_importance[k] = numeric_value / abs(numeric_value) * 1000
                            else:
                                processed_feature_importance[k] = numeric_value
                        except (ValueError, TypeError, Exception):
                            # Skip any values that cause errors
                            continue
                    
                    # Find most important speed-related features
                    speed_features = {k: v for k, v in processed_feature_importance.items() if "speed" in k.lower() or "belt" in k.lower() or "forming" in k.lower()}
                    if speed_features:
                        # Sort by importance and get top features
                        sorted_features = sorted(speed_features.items(), key=lambda x: abs(x[1]), reverse=True)
                        most_important = sorted_features[0] if sorted_features else None
                        top_features = sorted_features[:3]  # Get top 3 features
                        
                        # Create detailed explanation with multiple features and their actual values
                        if top_features:
                            feature_details = ', '.join([f"'{name}' ({abs(value):.2f})" for name, value in top_features])
                            explanation = f"Оптимизация на основе ключевых факторов: {feature_details}"
                            
                            # Adjust impact based on feature importance with more nuanced descriptions
                            max_importance = abs(most_important[1]) if most_important else 0
                            if max_importance > 50:
                                impact = "Значительная оптимизация производительности с учетом динамики формования и спектральных характеристик"
                            elif max_importance > 20:
                                impact = "Умеренная корректировка скорости для баланса качества и производительности"
                            elif max_importance > 5:
                                impact = "Небольшая корректировка скорости для стабилизации процесса формования"
                            else:
                                impact = "Минимальная корректировка скорости для оптимизации производительности"
                        else:
                            explanation = "Оптимизация скорости формования на основе комплексного анализа параметров"
                            impact = "Стандартная корректировка скорости для баланса качества и производительности"
                    else:
                        explanation = "Оптимизация скорости формования на основе комплексного анализа параметров"
                        impact = "Стандартная корректировка скорости для баланса качества и производительности"                
                recommendations.append({
                    "action": f"Увеличить скорость конвейера до {suggested_speed:.1f} м/мин",
                    "parameter": "belt_speed",
                    "priority": "MEDIUM",
                    "confidence": float(np.clip(base_confidence, 0.65, 0.92)),
                    "expected_impact": impact,
                    "current_value": current_speed,
                    "suggested_value": suggested_speed,
                    "justification": explanation,
                    "model": "PPO_Agent",
                    "feature_importance": dict(list(processed_feature_importance.items())[:5]) if feature_importance else {}
                })
            elif continuous_action[1] < 0.4:
                current_speed = float(state.get("belt_speed", 150))
                suggested_speed = float(current_speed - (0.5 - continuous_action[1]) * 20)  # Reduced scale
                
                # Ensure realistic speed bounds
                suggested_speed = max(120, min(180, suggested_speed))  # Realistic bounds for glass production
                
                # ML-driven explanation based on feature importance
                explanation = "ML-оптимизация скорости формования"
                impact = "Оптимизация качества"
                
                # Enhance explanation with feature importance if available
                # Process feature importance values with proper scaling to prevent overflow
                processed_feature_importance = {}
                if feature_importance:
                    for k, v in feature_importance.items():
                        try:
                            # Ensure v is a numeric value
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                numeric_value = float(v)
                            else:
                                # If not numeric, try to convert to float
                                numeric_value = float(v)
                            
                            # Apply proper scaling instead of hard clipping
                            if abs(numeric_value) > 1000:
                                # Scale down large values proportionally
                                processed_feature_importance[k] = numeric_value / abs(numeric_value) * 1000
                            else:
                                processed_feature_importance[k] = numeric_value
                        except (ValueError, TypeError, Exception):
                            # Skip any values that cause errors
                            continue
                    
                    # Find most important speed-related features
                    speed_features = {k: v for k, v in processed_feature_importance.items() if "speed" in k.lower() or "belt" in k.lower() or "forming" in k.lower()}
                    if speed_features:
                        # Sort by importance and get top features
                        sorted_features = sorted(speed_features.items(), key=lambda x: abs(x[1]), reverse=True)
                        most_important = sorted_features[0] if sorted_features else None
                        top_features = sorted_features[:3]  # Get top 3 features
                        
                        # Create detailed explanation with multiple features and their actual values
                        if top_features:
                            feature_details = ', '.join([f"'{name}' ({abs(value):.2f})" for name, value in top_features])
                            explanation = f"Оптимизация на основе ключевых факторов: {feature_details}"
                            
                            # Adjust impact based on feature importance with more nuanced descriptions
                            max_importance = abs(most_important[1]) if most_important else 0
                            if max_importance > 50:
                                impact = "Значительная оптимизация качества за счет коррекции скорости формования и спектрального анализа"
                            elif max_importance > 20:
                                impact = "Умеренная корректировка скорости для улучшения качества поверхности"
                            elif max_importance > 5:
                                impact = "Небольшая корректировка скорости для предотвращения дефектов"
                            else:
                                impact = "Минимальная корректировка скорости для оптимизации качества"
                        else:
                            explanation = "Оптимизация скорости формования для улучшения качества"
                            impact = "Стандартная корректировка скорости для предотвращения дефектов"
                    else:
                        explanation = "Оптимизация скорости формования для улучшения качества"
                        impact = "Стандартная корректировка скорости для предотвращения дефектов"                
                recommendations.append({
                    "action": f"Снизить скорость конвейера до {suggested_speed:.1f} м/мин",
                    "parameter": "belt_speed",
                    "priority": "HIGH" if state.get("defect_rate", 0.05) > 0.1 else "MEDIUM",
                    "confidence": float(np.clip(base_confidence + confidence_adjustment * 0.5, 0.7, 0.93)),
                    "expected_impact": impact,
                    "current_value": current_speed,
                    "suggested_value": suggested_speed,
                    "justification": explanation,
                    "model": "PPO_Agent",
                    "feature_importance": dict(list(processed_feature_importance.items())[:5]) if feature_importance else {}
                })
            
            # Mold temperature adjustment
            if continuous_action[2] > 0.6:
                current_mold_temp = float(state.get("mold_temp", 320))
                suggested_mold_temp = float(current_mold_temp + (continuous_action[2] - 0.5) * 30)  # Reduced scale
                
                # Ensure realistic mold temperature bounds
                suggested_mold_temp = max(280, min(380, suggested_mold_temp))  # Realistic bounds for glass production
                
                # ML-driven explanation based on feature importance
                explanation = "ML-оптимизация температуры формы"
                impact = "Оптимизация температурного режима"
                
                # Enhance explanation with feature importance if available
                # Process feature importance values with proper scaling to prevent overflow
                processed_feature_importance = {}
                if feature_importance:
                    for k, v in feature_importance.items():
                        try:
                            # Ensure v is a numeric value
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                numeric_value = float(v)
                            else:
                                # If not numeric, try to convert to float
                                numeric_value = float(v)
                            
                            # Apply proper scaling instead of hard clipping
                            if abs(numeric_value) > 1000:
                                # Scale down large values proportionally
                                processed_feature_importance[k] = numeric_value / abs(numeric_value) * 1000
                            else:
                                processed_feature_importance[k] = numeric_value
                        except (ValueError, TypeError, Exception):
                            # Skip any values that cause errors
                            continue
                    
                    # Find most important temperature-related features
                    temp_features = {k: v for k, v in processed_feature_importance.items() if "mold" in k.lower() or "temperature" in k.lower() or "temp" in k.lower()}
                    if temp_features:
                        # Sort by importance and get top features
                        sorted_features = sorted(temp_features.items(), key=lambda x: abs(x[1]), reverse=True)
                        most_important = sorted_features[0] if sorted_features else None
                        top_features = sorted_features[:3]  # Get top 3 features
                        
                        # Create detailed explanation with multiple features and their actual values
                        if top_features:
                            feature_details = ', '.join([f"'{name}' ({abs(value):.2f})" for name, value in top_features])
                            explanation = f"Оптимизация на основе ключевых факторов: {feature_details}"
                            
                            # Adjust impact based on feature importance with more nuanced descriptions
                            max_importance = abs(most_important[1]) if most_important else 0
                            if max_importance > 50:
                                impact = "Значительная оптимизация температурного режима формы для предотвращения напряжений и деформаций"
                            elif max_importance > 20:
                                impact = "Умеренная корректировка температуры формы для улучшения качества поверхности"
                            elif max_importance > 5:
                                impact = "Небольшая корректировка температуры формы для стабилизации процесса"
                            else:
                                impact = "Минимальная корректировка температуры формы для оптимизации качества"
                        else:
                            explanation = "Оптимизация температуры формы для улучшения качества"
                            impact = "Стандартная корректировка температуры формы для предотвращения дефектов"
                    else:
                        explanation = "Оптимизация температуры формы для улучшения качества"
                        impact = "Стандартная корректировка температуры формы для предотвращения дефектов"                
                recommendations.append({
                    "action": f"Повысить температуру формы до {suggested_mold_temp:.1f}°C",
                    "parameter": "mold_temp",
                    "priority": "MEDIUM",
                    "confidence": float(np.clip(base_confidence - confidence_adjustment * 0.3, 0.6, 0.88)),
                    "expected_impact": impact,
                    "current_value": current_mold_temp,
                    "suggested_value": suggested_mold_temp,
                    "justification": explanation,
                    "model": "PPO_Agent",
                    "feature_importance": dict(list(processed_feature_importance.items())[:5]) if feature_importance else {}
                })
            elif continuous_action[2] < 0.4:
                current_mold_temp = float(state.get("mold_temp", 320))
                suggested_mold_temp = float(current_mold_temp - (0.5 - continuous_action[2]) * 30)  # Reduced scale
                
                # Ensure realistic mold temperature bounds
                suggested_mold_temp = max(280, min(380, suggested_mold_temp))  # Realistic bounds for glass production
                
                # ML-driven explanation based on feature importance
                explanation = "ML-оптимизация температуры формы"
                impact = "Оптимизация энергопотребления"
                
                # Enhance explanation with feature importance if available
                # Cap feature importance values to prevent overflow
                processed_feature_importance = {}
                if feature_importance:
                    for k, v in feature_importance.items():
                        try:
                            # Ensure v is a numeric value
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                numeric_value = float(v)
                            else:
                                # If not numeric, try to convert to float
                                numeric_value = float(v)
                            
                            # Apply proper scaling instead of hard clipping
                            if abs(numeric_value) > 1000:
                                # Scale down large values proportionally
                                processed_feature_importance[k] = numeric_value / abs(numeric_value) * 1000
                            else:
                                processed_feature_importance[k] = numeric_value
                        except (ValueError, TypeError, Exception):
                            # Skip any values that cause errors
                            continue
                    
                    # Find most important temperature-related features
                    temp_features = {k: v for k, v in processed_feature_importance.items() if "mold" in k.lower() or "temperature" in k.lower()}
                    if temp_features:
                        # Sort by importance and get top features
                        sorted_features = sorted(temp_features.items(), key=lambda x: abs(x[1]), reverse=True)
                        most_important = sorted_features[0]
                        top_features = sorted_features[:3]  # Get top 3 features
                        
                        # Create detailed explanation with multiple features
                        feature_details = ', '.join([f"'{name}' ({abs(value):.3f})" for name, value in top_features])
                        explanation = f"Оптимизация на основе ключевых факторов: {feature_details}"
                        
                        # Adjust impact based on feature importance
                        max_importance = abs(most_important[1])
                        if max_importance > 0.3:
                            impact = "Значительная оптимизация энергопотребления за счет коррекции температуры формы"
                        elif max_importance > 0.1:
                            impact = "Умеренная корректировка температуры формы для экономии энергии"
                        else:
                            impact = "Незначительная корректировка температуры формы для снижения затрат"
                
                recommendations.append({
                    "action": f"Снизить температуру формы до {suggested_mold_temp:.1f}°C",
                    "parameter": "mold_temp",
                    "priority": "MEDIUM",
                    "confidence": float(np.clip(base_confidence + confidence_adjustment * 0.2, 0.65, 0.9)),
                    "expected_impact": impact,
                    "current_value": current_mold_temp,
                    "suggested_value": suggested_mold_temp,
                    "justification": explanation,
                    "model": "PPO_Agent",
                    "feature_importance": dict(list(processed_feature_importance.items())[:5]) if feature_importance else {}
                })
            
            # If no specific recommendations, add maintenance suggestion with contextual info
            if not recommendations:
                quality_score = float(state.get("quality_score", 0.85))
                energy_consumption = float(state.get("energy_consumption", 500))
                
                explanation = "ML-анализ показывает стабильные параметры"
                impact = "Стабильная работа оборудования"
                
                # Enhance explanation with feature importance if available
                # Cap feature importance values to prevent overflow
                processed_feature_importance = {}
                if feature_importance:
                    for k, v in feature_importance.items():
                        try:
                            # Ensure v is a numeric value before clipping
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                processed_feature_importance[k] = float(np.clip(float(v), -1000, 1000))
                            else:
                                # If not numeric, try to convert to float, otherwise skip
                                try:
                                    numeric_v = float(v)
                                    processed_feature_importance[k] = float(np.clip(numeric_v, -1000, 1000))
                                except (ValueError, TypeError):
                                    # Skip non-numeric values
                                    continue
                        except Exception:
                            # Skip any values that cause errors
                            continue
                    
                    # Overall system stability assessment with detailed feature analysis
                    if processed_feature_importance:
                        # Sort by absolute importance and get top features
                        sorted_features = sorted(processed_feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                        top_features = sorted_features[:3]  # Get top 3 features
                        
                        # Create detailed explanation with feature importance values
                        feature_details = ', '.join([f"'{name}' ({abs(value):.3f})" for name, value in top_features])
                        explanation = f"Стабильные параметры на основе анализа: {feature_details}"
                    else:
                        explanation = "Стабильные параметры: все показатели в норме"
                
                recommendations.append({
                    "action": "Параметры в оптимальном диапазоне. Продолжать мониторинг.",
                    "parameter": "all",
                    "priority": "LOW",
                    "confidence": float(np.clip(base_confidence * 0.8, 0.5, 0.8)),
                    "expected_impact": impact,
                    "current_values": {
                        "furnace_temperature": float(state.get("furnace_temperature", 1520)),
                        "belt_speed": float(state.get("belt_speed", 150)),
                        "mold_temp": float(state.get("mold_temp", 320)),
                        "quality_score": quality_score
                    },
                    "justification": explanation,
                    "model": "PPO_Agent",
                    "feature_importance": dict(list(processed_feature_importance.items())[:5]) if feature_importance else {}
                })
            
            logger.debug(f"🤖 RL generated {len(recommendations)} recommendations with ML-driven explanations")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ RL recommendation failed: {e}")
            return self._fallback_recommendations(state)
    
    def _fallback_defect_prediction(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Simple physics-based fallback when ML models unavailable"""
        temp = state.get("furnace_temperature", 1520)
        speed = state.get("belt_speed", 150)
        
        # Simple deviation-based probability
        temp_dev = abs(temp - 1520) / 100
        speed_dev = abs(speed - 150) / 50
        
        base_prob = 0.02 + temp_dev * 0.1 + speed_dev * 0.08
        
        return {
            "crack": float(np.clip(base_prob * 1.2, 0, 0.5)),
            "bubble": float(np.clip(base_prob * 0.8, 0, 0.5)),
            "chip": float(np.clip(base_prob * 0.6, 0, 0.5)),
            "stain": float(np.clip(base_prob * 0.4, 0, 0.5)),
            "cloudiness": float(np.clip(base_prob * 0.5, 0, 0.5)),
            "deformation": float(np.clip(base_prob * 1.0, 0, 0.5))
        }
    
    def _fallback_recommendations(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple rule-based fallback when RL agent unavailable"""
        recommendations = []
        
        temp = state.get("furnace_temperature", 1520)
        speed = state.get("belt_speed", 150)
        
        if temp > 1560:
            recommendations.append({
                "action": "Снизить температуру печи для предотвращения дефектов",
                "parameter": "furnace_temperature",
                "priority": "HIGH",
                "confidence": 0.75,
                "expected_impact": "Снижение риска трещин",
                "model": "Rule-Based Fallback"
            })
        elif temp < 1480:
            recommendations.append({
                "action": "Повысить температуру печи для улучшения текучести",
                "parameter": "furnace_temperature",
                "priority": "MEDIUM",
                "confidence": 0.70,
                "expected_impact": "Улучшение качества поверхности",
                "model": "Rule-Based Fallback"
            })
        
        if speed > 170:
            recommendations.append({
                "action": "Снизить скорость формирования",
                "parameter": "belt_speed",
                "priority": "HIGH",
                "confidence": 0.80,
                "expected_impact": "Снижение деформаций",
                "model": "Rule-Based Fallback"
            })
        
        return recommendations


# Global ML pipeline instance (lazy loading)
_ml_pipeline: Optional[MLInferencePipeline] = None

def get_ml_pipeline() -> MLInferencePipeline:
    """Get or create the global ML inference pipeline"""
    global _ml_pipeline
    if _ml_pipeline is None:
        _ml_pipeline = MLInferencePipeline()
    return _ml_pipeline


class WebSocketBroadcaster:
    """Manages WebSocket connections and broadcasts data to connected clients"""
    
    def __init__(self):
        # Store active connections by channel
        self.active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.all_connections: Set[WebSocket] = set()
        
        # Channels
        self.channels = [
            "sensors",      # Real-time sensor data
            "predictions",  # ML model predictions
            "alerts",       # Alert notifications
            "recommendations",  # RL agent recommendations
            "quality",      # Quality metrics and KPIs
            "all"           # All channels combined
        ]
        
        # Message counters for stats
        self.message_stats = defaultdict(int)
        
    async def connect(self, websocket: WebSocket, channels: List[str] = None):
        """
        Accept a new WebSocket connection and subscribe to channels
        
        Args:
            websocket: FastAPI WebSocket instance
            channels: List of channel names to subscribe to. If None, subscribe to 'all'
        """
        await websocket.accept()
        
        if channels is None:
            channels = ["all"]
        
        # Add to channel subscriptions
        for channel in channels:
            if channel in self.channels:
                self.active_connections[channel].add(websocket)
        
        # Add to all connections
        self.all_connections.add(websocket)
        
        logger.info(f"📡 New WebSocket connection. Channels: {channels}. Total connections: {len(self.all_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        # Remove from all channels
        for channel in self.channels:
            self.active_connections[channel].discard(websocket)
        
        # Remove from all connections
        self.all_connections.discard(websocket)
        
        logger.info(f"📴 WebSocket disconnected. Total connections: {len(self.all_connections)}")
    
    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """
        Broadcast a message to all clients subscribed to a channel
        
        Args:
            channel: Channel name
            message: Message data to broadcast
        """
        if channel not in self.channels:
            logger.warning(f"⚠️ Unknown channel: {channel}")
            return
        
        # Add metadata
        message_with_metadata = {
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **message
        }
        
        # Serialize to JSON with proper type conversion
        try:
            # Convert numpy types to native Python types
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, (np.datetime64, np.timedelta64)):
                    return str(obj)
                else:
                    return obj
                        
            # Convert all values in the message
            converted_message = convert_numpy_types(message_with_metadata)
            message_json = json.dumps(converted_message, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"❌ Error serializing message: {e}")
            # Try a more aggressive conversion as fallback
            try:
                def aggressive_convert(obj):
                    try:
                        return convert_numpy_types(obj)
                    except:
                        return str(obj)
                converted_message = json.loads(json.dumps(message_with_metadata, default=aggressive_convert))
                message_json = json.dumps(converted_message, ensure_ascii=False)
            except Exception as e2:
                logger.error(f"❌ Error in fallback serialization: {e2}")
                return
        # Get connections for this channel and 'all' channel
        target_connections = self.active_connections[channel].union(
            self.active_connections["all"]
        )
        
        # Broadcast to all connections
        disconnected = []
        for connection in target_connections:
            try:
                await connection.send_text(message_json)
                self.message_stats[channel] += 1
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"❌ Error sending message to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
        
        if target_connections:
            logger.debug(f"📤 Broadcasted to {len(target_connections)} clients on channel '{channel}'")
    
    async def broadcast_sensor_update(self, sensor_data: Dict[str, Any]):
        """Broadcast sensor data update"""
        await self.broadcast_to_channel("sensors", {
            "type": "sensor_update",
            "data": sensor_data
        })
    
    async def broadcast_prediction_update(self, predictions: Dict[str, Any]):
        """Broadcast ML model predictions"""
        await self.broadcast_to_channel("predictions", {
            "type": "prediction_update",
            "data": predictions
        })
    
    async def broadcast_alert(self, alert: Dict[str, Any]):
        """Broadcast alert notification"""
        await self.broadcast_to_channel("alerts", {
            "type": "alert",
            "data": alert
        })
    
    async def broadcast_recommendation(self, recommendation: Dict[str, Any]):
        """Broadcast RL agent recommendation"""
        await self.broadcast_to_channel("recommendations", {
            "type": "recommendation",
            "data": recommendation
        })
    
    async def broadcast_quality_metrics(self, metrics: Dict[str, Any]):
        """Broadcast quality metrics and KPIs"""
        await self.broadcast_to_channel("quality", {
            "type": "quality_metrics",
            "data": metrics
        })
    
    async def broadcast_defect_alert(self, defect_data: Dict[str, Any]):
        """Broadcast defect alert with enriched information"""
        await self.broadcast_to_channel("alerts", {
            "type": "defect_alert",
            "data": defect_data
        })
    
    async def broadcast_parameter_update(self, parameter_data: Dict[str, Any]):
        """Broadcast parameter update for real-time monitoring"""
        await self.broadcast_to_channel("sensors", {
            "type": "parameter_update",
            "data": parameter_data
        })
    
    async def broadcast_ml_prediction(self, prediction_data: Dict[str, Any]):
        """Broadcast ML model prediction"""
        await self.broadcast_to_channel("predictions", {
            "type": "ml_prediction",
            "data": prediction_data
        })
    
    async def broadcast_system_health(self, health_data: Dict[str, Any]):
        """Broadcast system health metrics"""
        await self.broadcast_to_channel("all", {
            "type": "system_health",
            "data": health_data
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics"""
        return {
            "total_connections": len(self.all_connections),
            "connections_by_channel": {
                channel: len(connections)
                for channel, connections in self.active_connections.items()
            },
            "messages_sent_by_channel": dict(self.message_stats),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


# Global broadcaster instance
broadcaster = WebSocketBroadcaster()


async def sensor_data_stream_task(broadcaster: WebSocketBroadcaster, data_generator, notification_manager=None, ml_pipeline: MLInferencePipeline = None):
    """
    ML-Driven sensor data streaming with ML-based notifications
    
    Notifications are generated by ML models (GNN anomaly detection),
    not hardcoded status checks.
    
    Args:
        broadcaster: WebSocketBroadcaster instance
        data_generator: GlassProductionDataGenerator instance
        notification_manager: NotificationManager instance (optional)
        ml_pipeline: MLInferencePipeline for anomaly detection (optional)
    """
    logger.info("🤖 Starting ML-driven sensor data stream task...")
    
    if ml_pipeline is None:
        ml_pipeline = get_ml_pipeline()
    
    notification_cooldown = 300  # 5 minutes cooldown
    last_anomaly_notification = 0
    
    try:
        while True:
            # Generate sensor readings (simulation)
            readings = []
            for sensor_key in data_generator.sensors.keys():
                reading = data_generator.generate_sensor_reading(sensor_key)
                readings.append(reading)
            
            # Get current state
            current_state = data_generator.current_state
            state_summary = data_generator.get_current_state_summary()
            
            # ========== ML-BASED ANOMALY DETECTION ==========
            # Use GNN to detect sensor anomalies instead of hardcoded rules
            anomaly_result = await ml_pipeline.detect_anomaly_gnn(current_state)
            
            # Generate ML-driven notifications for anomalies
            if notification_manager and anomaly_result["has_anomaly"]:
                current_time = datetime.utcnow().timestamp()
                
                # Throttle notifications
                if current_time - last_anomaly_notification > notification_cooldown:
                    # Get which sensors are anomalous from GNN
                    anomaly_sensors = anomaly_result.get("anomaly_sensors", [])
                    confidence = anomaly_result.get("confidence", 0.0)
                    
                    # Determine priority based on ML confidence
                    if confidence > 0.9:
                        priority = "CRITICAL"
                    elif confidence > 0.7:
                        priority = "HIGH"
                    else:
                        priority = "MEDIUM"
                    
                    # Get RL recommendations for this anomaly
                    rl_recommendations = await ml_pipeline.generate_rl_recommendations(current_state)
                    
                    notification_manager.create_notification(
                        category="ML_SENSOR_ANOMALY",
                        priority=priority,
                        title=f"GNN: Аномалия датчиков ({len(anomaly_sensors)} датчиков)",
                        message=f"Модель GNN обнаружила аномалию с доверием {confidence:.1%}. Датчики: {', '.join(anomaly_sensors[:5])}",
                        source="GNN_Model",
                        actions=[
                            {"label": "Проверить датчики", "action": "inspect_sensors"},
                            {"label": "Применить RL рекомендацию", "action": "apply_rl_recommendation"},
                            {"label": "Отклонить", "action": "dismiss"}
                        ],
                        metadata={
                            "anomaly_sensors": anomaly_sensors[:5],
                            "confidence": confidence,
                            "current_state": {
                                "furnace_temperature": float(current_state.get("furnace_temperature", 1520)),
                                "furnace_pressure": float(current_state.get("furnace_pressure", 15.2)),
                                "belt_speed": float(current_state.get("belt_speed", 150)),
                                "mold_temp": float(current_state.get("mold_temp", 320)),
                                "forming_pressure": float(current_state.get("forming_pressure", 45)),
                                "annealing_temp": float(current_state.get("annealing_temp", 580)),
                                "cooling_rate": float(current_state.get("cooling_rate", 3.5)),
                                "quality_score": float(current_state.get("quality_score", 0.95)),
                                "defect_rate": float(current_state.get("defect_rate", 0.05)),
                                "energy_consumption": float(current_state.get("energy_consumption", 500))
                            },
                            "recommendations": rl_recommendations
                        }
                    )
                    last_anomaly_notification = current_time
                    logger.info(f"🤖 ML Anomaly notification: {len(anomaly_sensors)} sensors, confidence={confidence:.1%}")            
            # Aggregate sensor data with ML annotations
            aggregated = {
                "production_line": "Line_A",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sensors": readings,
                "state_summary": state_summary,
                "ml_anomaly": anomaly_result  # Include ML anomaly detection
            }
            
            # Broadcast sensor update
            await broadcaster.broadcast_sensor_update(aggregated)
            
            # Also broadcast parameter update for real-time charts
            parameter_update = {
                "furnace": {
                    "temperature": current_state.get("furnace_temperature", 1520),
                    "pressure": current_state.get("furnace_pressure", 15.2)
                },
                "forming": {
                    "speed": current_state.get("belt_speed", 150),
                    "mold_temp": current_state.get("mold_temp", 320),
                    "pressure": current_state.get("forming_pressure", 45)
                },
                "annealing": {
                    "temperature": current_state.get("annealing_temp", 580),
                    "cooling_rate": current_state.get("cooling_rate", 3.5)
                },
                "anomaly_detected": anomaly_result["has_anomaly"],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            await broadcaster.broadcast_parameter_update(parameter_update)
            
            logger.debug(f"📊 Sensor update: {len(readings)} sensors, anomaly={anomaly_result['has_anomaly']}")
            
            # Wait before next update (5 seconds for dashboard)
            await asyncio.sleep(5)
            
    except asyncio.CancelledError:
        logger.info("⏹️ ML sensor data stream task cancelled")
    except Exception as e:
        logger.error(f"❌ Error in ML sensor data stream: {e}")


async def defect_detection_stream_task(broadcaster: WebSocketBroadcaster, data_generator, notification_manager=None, analytics_storage=None, ml_pipeline: MLInferencePipeline = None):
    """
    ML-Driven defect detection using LSTM model
    
    Sensor data is passed to ML models which generate predictions.
    Defects are detected based on LSTM output, not hardcoded physics.
    
    Args:
        broadcaster: WebSocketBroadcaster instance
        data_generator: GlassProductionDataGenerator instance (sensor simulation)
        notification_manager: NotificationManager instance (optional)
        analytics_storage: Storage for analytics data (optional)
        ml_pipeline: MLInferencePipeline for model inference (optional)
    """
    logger.info("🤖 Starting ML-driven defect detection stream task...")
    
    # Use provided or global ML pipeline
    if ml_pipeline is None:
        ml_pipeline = get_ml_pipeline()
    
    defect_counter = 0
    defects_history = []
    
    try:
        while True:
            # Get current production state (simulated sensor data)
            current_state = data_generator.current_state
            
            # ========== ML PREDICTION ==========
            # Pass sensor data to LSTM model for defect prediction
            defect_probabilities = await ml_pipeline.predict_defects_lstm(current_state)
            
            # Also run GNN for sensor anomaly detection
            anomaly_result = await ml_pipeline.detect_anomaly_gnn(current_state)
            
            # Broadcast ML predictions to frontend
            await broadcaster.broadcast_ml_prediction({
                "defect_probabilities": defect_probabilities,
                "anomaly_detection": anomaly_result,
                "model_source": "LSTM+GNN",
                "inference_timestamp": datetime.utcnow().isoformat() + "Z"
            })
            
            # ========== DEFECT GENERATION BASED ON ML ==========
            # Find the defect with the highest probability that exceeds threshold
            max_defect_type = None
            max_probability = 0.0
            
            # First, find the defect with maximum probability
            for defect_type, probability in defect_probabilities.items():
                if probability > max_probability:
                    max_probability = probability
                    max_defect_type = defect_type
            
            # Generate defect alert if the highest probability exceeds threshold
            if max_defect_type and max_probability > 0.15 and np.random.random() < max_probability * 0.1:
                defect_counter += 1
                
                # Determine severity based on ML probability
                if max_probability > 0.6:
                    severity = "CRITICAL"
                elif max_probability > 0.4:
                    severity = "HIGH"
                elif max_probability > 0.25:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"
                
                defect = {
                    "defect_type": max_defect_type,
                    "severity": severity,
                    "probability": float(max_probability),
                    "location": {
                        "line": "Line_A",
                        "position_x": float(np.random.uniform(0, 100)),
                        "position_y": float(np.random.uniform(0, 100))
                    },
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "confidence": float(max_probability),
                    "model_source": "LSTM",
                    "anomaly_detected": anomaly_result["has_anomaly"],
                    "parameters_snapshot": {
                        "furnace_temperature": float(current_state.get("furnace_temperature", 1520)),
                        "furnace_pressure": float(current_state.get("furnace_pressure", 15.2)),
                        "belt_speed": float(current_state.get("belt_speed", 150)),
                        "mold_temp": float(current_state.get("mold_temp", 320)),
                        "forming_pressure": float(current_state.get("forming_pressure", 45)),
                        "annealing_temp": float(current_state.get("annealing_temp", 580)),
                        "cooling_rate": float(current_state.get("cooling_rate", 3.5)),
                        "quality_score": float(current_state.get("quality_score", 0.95)),
                        "defect_rate": float(current_state.get("defect_rate", 0.05)),
                        "energy_consumption": float(current_state.get("energy_consumption", 500))
                    }
                }
                
                defects_history.append(defect)
                if len(defects_history) > 1000:
                    defects_history = defects_history[-1000:]
                
                # Broadcast ML-detected defect
                await broadcaster.broadcast_defect_alert(defect)
                logger.info(f"🤖 ML Defect #{defect_counter}: {max_defect_type} (prob: {max_probability:.2%}, severity: {severity})")
                
                # Get RL recommendations for this defect
                rl_recommendations = await ml_pipeline.generate_rl_recommendations(current_state)
                
                # Create notification for ALL ML-driven defects (any confidence)
                if notification_manager:
                    notification_manager.create_notification(
                        category="ML_DEFECT_PREDICTION",
                        priority=severity,
                        title=f"LSTM: {max_defect_type} ({max_probability:.1%})",
                        message=f"Модель LSTM прогнозирует дефект '{max_defect_type}' с вероятностью {max_probability:.1%}. Уровень: {severity}",
                        source="LSTM_Model",
                        actions=[
                            {"label": "Применить рекомендацию", "action": "apply_recommendation"},
                            {"label": "Отрегулировать параметры", "action": "adjust_parameters"},
                            {"label": "Игнорировать", "action": "dismiss"}
                        ],
                        metadata={
                            "defect_type": max_defect_type,
                            "probability": max_probability,
                            "severity": severity,
                            "model_source": "LSTM",
                            "defect_probabilities": defect_probabilities,
                            "parameters_snapshot": defect["parameters_snapshot"],
                            "recommendations": rl_recommendations
                        }
                    )
                    logger.info(f"📢 ML Notification created: {max_defect_type} ({severity}, {max_probability:.1%})")
                
                # ========== KNOWLEDGE GRAPH ENRICHMENT ==========
                # Enrich KG with ML prediction for real-time learning
                try:
                    kg = get_knowledge_graph()
                    if kg:
                        kg.enrich_from_ml_prediction(
                            defect_type=max_defect_type,
                            probability=max_probability,
                            severity=severity,
                            sensor_snapshot=defect['parameters_snapshot'],
                            model_source="LSTM",
                            timestamp=datetime.utcnow()
                        )
                        logger.debug(f"🧠 KG enriched with {max_defect_type} prediction")
                except Exception as kg_error:
                    logger.debug(f"⚠️ KG enrichment skipped: {kg_error}")
                
                # Store to analytics
                if analytics_storage:
                    try:
                        analytics_storage['defects'].append({
                            "timestamp": defect['timestamp'],
                            "type": max_defect_type,
                            "severity": severity,
                            "probability": max_probability,
                            "model_source": "LSTM",
                            "location": defect['location'],
                            "parameters": defect['parameters_snapshot']
                        })
                    except Exception as e:
                        logger.error(f"❌ Error storing defect: {e}")
            
            # MIK-1 inspection rate: 3-5 seconds
            await asyncio.sleep(4)
            
    except asyncio.CancelledError:
        logger.info("⏹️ ML defect detection task cancelled")
    except Exception as e:
        logger.error(f"❌ Error in ML defect detection: {e}")


async def quality_metrics_stream_task(broadcaster: WebSocketBroadcaster, data_generator, analytics_storage=None, ml_pipeline: MLInferencePipeline = None, notification_manager=None):
    """
    ML-Driven quality metrics using LSTM predictions + sensor-based quality level
    
    Quality metrics are derived from:
    1. ML model predictions (defect probabilities)
    2. Sensor-based quality level (good vs bad sensor readings ratio)
    3. Defect distribution from confirmed defects in notifications
    
    Args:
        broadcaster: WebSocketBroadcaster instance
        data_generator: GlassProductionDataGenerator instance
        analytics_storage: Storage for analytics data (optional)
        ml_pipeline: MLInferencePipeline for model inference (optional)
        notification_manager: NotificationManager for defect distribution (optional)
    """
    logger.info("🤖 Starting ML-driven quality metrics stream task...")
    
    if ml_pipeline is None:
        ml_pipeline = get_ml_pipeline()
    
    # Track sensor quality history
    good_sensor_count = 0
    bad_sensor_count = 0
    
    try:
        while True:
            state = data_generator.current_state
            
            # ========== ML-BASED QUALITY CALCULATION ==========
            # Get defect probabilities from LSTM
            defect_probs = await ml_pipeline.predict_defects_lstm(state)
            
            # Calculate quality score based on ML predictions
            # Lower defect probabilities = higher quality
            avg_defect_prob = sum(defect_probs.values()) / len(defect_probs)
            ml_quality_score = (1.0 - avg_defect_prob) * 100
            
            # Calculate defect rate from ML predictions
            ml_defect_rate = avg_defect_prob * 100  # Convert to percentage
            
            # Get anomaly info from GNN
            anomaly_result = await ml_pipeline.detect_anomaly_gnn(state)
            
            # ========== SENSOR-BASED QUALITY LEVEL ==========
            # User requirement: "брать хорошие сенсорные данные и брать плохие сенсорные данные,
            # считать насколько меньше плохих"
            # Check sensor status from current readings
            current_good = 0
            current_bad = 0
            
            # Evaluate each sensor parameter
            sensor_checks = [
                ("furnace_temperature", 1400, 1600, state.get("furnace_temperature", 1520)),
                ("furnace_pressure", 10, 20, state.get("furnace_pressure", 15.2)),
                ("belt_speed", 120, 180, state.get("belt_speed", 150)),
                ("mold_temp", 250, 400, state.get("mold_temp", 320)),
                ("forming_pressure", 30, 70, state.get("forming_pressure", 45)),
                ("annealing_temp", 500, 700, state.get("annealing_temp", 580)),
                ("cooling_rate", 2, 5, state.get("cooling_rate", 3.5)),
            ]
            
            for param_name, min_val, max_val, current_val in sensor_checks:
                if min_val <= current_val <= max_val:
                    current_good += 1
                else:
                    current_bad += 1
            
            # Update cumulative counters
            good_sensor_count += current_good
            bad_sensor_count += current_bad
            
            # Calculate quality level as ratio: насколько меньше плохих
            total_sensors = good_sensor_count + bad_sensor_count
            if total_sensors > 0:
                quality_level = (good_sensor_count / total_sensors) * 100
            else:
                quality_level = 100.0
            
            # ========== DEFECT DISTRIBUTION FROM NOTIFICATIONS ==========
            # User requirement: "распределение дефектов нужно получать из подтвержденных дефектов из уведомлений"
            defect_distribution = {}
            if notification_manager:
                # Get confirmed defect notifications
                confirmed_defects = [
                    n for n in notification_manager.notifications 
                    if n.get("category") in ["CRITICAL_DEFECT", "ML_DEFECT_PREDICTION"] 
                    and n.get("acknowledged", False)
                ]
                
                # Aggregate by defect type
                for notification in confirmed_defects:
                    # Extract defect type from notification message or metadata
                    title = notification.get("title", "")
                    for defect_type in ["crack", "bubble", "chip", "stain", "cloudiness", "deformation"]:
                        if defect_type in title.lower():
                            defect_distribution[defect_type] = defect_distribution.get(defect_type, 0) + 1
            
            # Production rate (simulated, could come from sensors)
            production_rate = round(150 + np.random.normal(0, 5), 1)
            
            # Adjust quality score if anomaly detected
            if anomaly_result["has_anomaly"]:
                ml_quality_score *= 0.95  # 5% penalty for anomalies
            
            metrics = {
                "production_line": "Line_A",
                "defect_rate": round(ml_defect_rate, 2),
                "quality_score": round(ml_quality_score, 2),
                "quality_level": round(quality_level, 2),  # NEW: Sensor-based quality level
                "good_sensor_count": good_sensor_count,
                "bad_sensor_count": bad_sensor_count,
                "production_rate": production_rate,
                "total_defects": int(ml_defect_rate * 10),
                "critical_alerts": len(anomaly_result.get("anomaly_sensors", [])),
                "anomaly_detected": anomaly_result["has_anomaly"],
                "model_source": "LSTM+GNN",
                "defect_probabilities": {k: round(v * 100, 2) for k, v in defect_probs.items()},
                "defect_distribution": defect_distribution,  # NEW: From confirmed notifications
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            await broadcaster.broadcast_quality_metrics(metrics)
            
            if analytics_storage:
                try:
                    analytics_storage['quality_metrics'].append({
                        "timestamp": metrics['timestamp'],
                        "defect_rate": ml_defect_rate,
                        "quality_score": ml_quality_score,
                        "quality_level": quality_level,
                        "production_rate": production_rate,
                        "model_source": "LSTM",
                        "furnace_temp": state.get("furnace_temperature", 1520),
                        "belt_speed": state.get("belt_speed", 150)
                    })
                    if len(analytics_storage['quality_metrics']) > 500:
                        analytics_storage['quality_metrics'] = analytics_storage['quality_metrics'][-500:]
                except Exception as e:
                    logger.error(f"❌ Error storing quality metrics: {e}")
            
            logger.debug(f"🤖 ML Quality: score={ml_quality_score:.1f}%, quality_level={quality_level:.1f}%, defect_rate={ml_defect_rate:.2f}%")
            
            await asyncio.sleep(30)
            
    except asyncio.CancelledError:
        logger.info("⏹️ ML quality metrics task cancelled")
    except Exception as e:
        logger.error(f"❌ Error in ML quality metrics: {e}")


async def recommendations_stream_task(broadcaster: WebSocketBroadcaster, data_generator, analytics_storage=None, ml_pipeline: MLInferencePipeline = None, notification_manager=None):
    """
    RL-Driven recommendations using PPO agent
    
    Recommendations are generated by the RL agent, not hardcoded templates.
    The agent analyzes current state and suggests optimal actions.
    
    Args:
        broadcaster: WebSocketBroadcaster instance
        data_generator: GlassProductionDataGenerator instance
        analytics_storage: Storage dict for analytics data (optional)
        ml_pipeline: MLInferencePipeline with RL agent (optional)
        notification_manager: NotificationManager instance (optional)
    """
    logger.info("🤖 Starting RL-driven recommendations stream task...")
    
    if ml_pipeline is None:
        ml_pipeline = get_ml_pipeline()
    
    try:
        while True:
            # Get current state (simulated sensor data)
            state = data_generator.current_state
            
            # ========== RL AGENT RECOMMENDATIONS ==========
            # Generate recommendations using PPO agent
            recommendations = await ml_pipeline.generate_rl_recommendations(state)
            
            # Also get current defect predictions for context
            defect_probs = await ml_pipeline.predict_defects_lstm(state)
            
            # Find highest risk defect
            if defect_probs:
                max_defect = max(defect_probs.items(), key=lambda x: x[1])
                highest_risk_defect = max_defect[0]
                highest_risk_prob = max_defect[1]
            else:
                highest_risk_defect = None
                highest_risk_prob = 0
            
            # Broadcast each recommendation
            for rec in recommendations:
                rec["timestamp"] = datetime.utcnow().isoformat() + "Z"
                rec["highest_risk_defect"] = highest_risk_defect
                rec["highest_risk_probability"] = round(highest_risk_prob * 100, 1)
                
                await broadcaster.broadcast_recommendation(rec)
                logger.debug(f"🤖 RL Recommendation: {rec['action'][:60]}...")
                
                # ========== KNOWLEDGE GRAPH ENRICHMENT ==========
                # Enrich KG with RL recommendation for defect-recommendation linking
                if highest_risk_defect:
                    try:
                        kg = get_knowledge_graph()
                        if kg:
                            kg.enrich_from_rl_recommendation(
                                defect_type=highest_risk_defect,
                                recommendation=rec,
                                expected_impact=rec.get('expected_impact', 0.5),
                                timestamp=datetime.utcnow()
                            )
                            logger.debug(f"🧠 KG enriched with RL recommendation for {highest_risk_defect}")
                    except Exception as kg_error:
                        logger.debug(f"⚠️ KG enrichment skipped: {kg_error}")
            
            if recommendations:
                logger.info(f"🤖 RL Agent generated {len(recommendations)} recommendations")
                
                # Create notification for RL recommendations if notification manager is available
                if notification_manager and len(recommendations) > 0:
                    # Get the first (most confident) recommendation as the primary one
                    primary_rec = recommendations[0]
                    
                    # Determine priority based on confidence
                    confidence = primary_rec.get("confidence", 0.7)
                    if confidence > 0.85:
                        priority = "HIGH"
                    elif confidence > 0.7:
                        priority = "MEDIUM"
                    else:
                        priority = "LOW"
                    
                    notification_manager.create_notification(
                        category="RL_RECOMMENDATION",
                        priority=priority,
                        title=f"RL Агент: {primary_rec.get('action', 'Рекомендация по оптимизации')[:50]}...",
                        message=f"RL агент предлагает действие с уверенностью {confidence:.1%}. {primary_rec.get('expected_impact', '')}",
                        source="PPO_Agent",
                        actions=[
                            {"label": "Применить рекомендацию", "action": "apply_recommendation"},
                            {"label": "Подробнее", "action": "view_details"},
                            {"label": "Игнорировать", "action": "dismiss"}
                        ],
                        metadata={
                            "recommendations": recommendations,
                            "highest_risk_defect": highest_risk_defect,
                            "highest_risk_probability": highest_risk_prob,
                            "current_state": {
                                "furnace_temperature": float(state.get("furnace_temperature", 1520)),
                                "furnace_pressure": float(state.get("furnace_pressure", 15.2)),
                                "belt_speed": float(state.get("belt_speed", 150)),
                                "mold_temp": float(state.get("mold_temp", 320)),
                                "forming_pressure": float(state.get("forming_pressure", 45)),
                                "annealing_temp": float(state.get("annealing_temp", 580)),
                                "cooling_rate": float(state.get("cooling_rate", 3.5)),
                                "quality_score": float(state.get("quality_score", 0.95)),
                                "defect_rate": float(state.get("defect_rate", 0.05)),
                                "energy_consumption": float(state.get("energy_consumption", 500))
                            }
                        }
                    )
            
            # Store to analytics if available
            if analytics_storage and recommendations:
                try:
                    for rec in recommendations:
                        analytics_storage.setdefault('recommendations', []).append({
                            "timestamp": rec['timestamp'],
                            "action": rec['action'],
                            "priority": rec.get('priority', 'MEDIUM'),
                            "confidence": rec.get('confidence', 0.7),
                            "model": rec.get('model', 'PPO_Agent'),
                            "highest_risk_defect": highest_risk_defect
                        })
                    # Keep only last 200 recommendations
                    if len(analytics_storage.get('recommendations', [])) > 200:
                        analytics_storage['recommendations'] = analytics_storage['recommendations'][-200:]
                except Exception as e:
                    logger.error(f"❌ Error storing recommendations: {e}")
            
            # Generate recommendations every 10 minutes (600 seconds)
            await asyncio.sleep(600)
            
    except asyncio.CancelledError:
        logger.info("⏹️ RL recommendations task cancelled")
    except Exception as e:
        logger.error(f"❌ Error in RL recommendations: {e}")


async def start_background_tasks(
    broadcaster: WebSocketBroadcaster, 
    data_generator, 
    notification_manager=None, 
    analytics_storage=None,
    ml_pipeline: MLInferencePipeline = None
):
    """
    Start all ML-driven background streaming tasks
    
    Architecture:
    - Sensor simulation: data_generator provides simulated sensor readings
    - ML Inference: ml_pipeline runs LSTM/GNN/RL models on sensor data
    - Broadcasting: Results are sent to frontend via WebSocket
    
    Args:
        broadcaster: WebSocketBroadcaster instance
        data_generator: GlassProductionDataGenerator instance (sensor simulation)
        notification_manager: NotificationManager instance (optional)
        analytics_storage: Storage dict for analytics data (optional)
        ml_pipeline: MLInferencePipeline for model inference (optional)
    """
    # Initialize ML pipeline if not provided
    if ml_pipeline is None:
        ml_pipeline = get_ml_pipeline()
    
    logger.info(f"🚀 Starting ML-driven background tasks:")
    logger.info(f"   - notification_manager: {'enabled' if notification_manager else 'disabled'}")
    logger.info(f"   - analytics_storage: {'enabled' if analytics_storage else 'disabled'}")
    logger.info(f"   - ML pipeline: {'loaded' if ml_pipeline.is_loaded else 'fallback mode'}")
    
    tasks = [
        # ML-driven sensor streaming (GNN anomaly detection)
        asyncio.create_task(sensor_data_stream_task(
            broadcaster, data_generator, notification_manager, ml_pipeline
        )),
        
        # ML-driven defect detection (LSTM + GNN)
        asyncio.create_task(defect_detection_stream_task(
            broadcaster, data_generator, notification_manager, analytics_storage, ml_pipeline
        )),
        
        # ML-driven quality metrics (LSTM predictions + sensor quality)
        asyncio.create_task(quality_metrics_stream_task(
            broadcaster, data_generator, analytics_storage, ml_pipeline, notification_manager
        )),
        
        # RL-driven recommendations (PPO agent)
        asyncio.create_task(recommendations_stream_task(
            broadcaster, data_generator, analytics_storage, ml_pipeline, notification_manager
        )),
    ]
    
    logger.info(f"✅ Created {len(tasks)} ML-driven streaming tasks:")
    logger.info(f"   1. sensors (GNN anomaly detection)")
    logger.info(f"   2. defects (LSTM + GNN + notifications)")
    logger.info(f"   3. quality (LSTM + sensor level)")
    logger.info(f"   4. recommendations (PPO Agent)")
    
    return tasks


if __name__ == "__main__":
    # Test the broadcaster
    from data_ingestion.synthetic_data_generator import GlassProductionDataGenerator
    
    async def test_broadcaster():
        broadcaster = WebSocketBroadcaster()
        generator = GlassProductionDataGenerator()
        
        logger.info("🧪 Testing WebSocket broadcaster...")
        
        # Start background tasks
        tasks = await start_background_tasks(broadcaster, generator)
        
        # Run for 30 seconds
        await asyncio.sleep(30)
        
        # Cancel tasks
        for task in tasks:
            task.cancel()
        
        # Show stats
        stats = broadcaster.get_stats()
        logger.info(f"📊 Broadcaster stats: {stats}")
    
    asyncio.run(test_broadcaster())
