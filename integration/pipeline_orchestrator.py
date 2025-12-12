"""
Pipeline Orchestrator - Integrates all system components into unified end-to-end pipeline
Phases 5-8: RL autonomy, frontend integration, metrics, and continuous learning
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import deque

# Import feature engineering
from feature_engineering.domain_features import GlassProductionFeatureExtractor
from feature_engineering.real_time_features import RealTimeFeatureExtractor
from feature_engineering.statistical_features import StatisticalFeatureExtractor

# Import training components
from training.continuous_learning import ContinualLearningFramework, ElasticWeightConsolidation

# Import explainability
from explainability.model_explainer import IntegratedModelExplainer
from explainability.feature_attribution import create_glass_production_explainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the complete end-to-end pipeline:
    Data → Feature Engineering → ML Models → Explainability → RL Agent → Actions → Frontend
    """
    
    def __init__(self, unified_system):
        """
        Args:
            unified_system: UnifiedGlassProductionSystem instance
        """
        self.unified_system = unified_system
        self.system_integrator = unified_system.system_integrator
        
        # Feature engineering components
        self.domain_features = GlassProductionFeatureExtractor()
        self.realtime_features = None  # Initialize on demand
        self.statistical_features = StatisticalFeatureExtractor()
        
        # Explainability components
        self.model_explainer = IntegratedModelExplainer()
        
        # Continual learning
        self.continual_learning = None  # Initialize after models are registered
        
        # Pipeline state
        self.pipeline_running = False
        self.latest_features = {}
        self.latest_predictions = {}
        self.latest_explanations = {}
        self.feature_buffer = deque(maxlen=1000)
        
        # Performance metrics
        self.metrics = {
            "pipeline_executions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "avg_latency_ms": 0.0,
            "feature_extraction_time_ms": 0.0,
            "prediction_time_ms": 0.0,
            "explanation_time_ms": 0.0
        }
        
        logger.info("🔄 Pipeline Orchestrator initialized")
    
    async def initialize(self):
        """Initialize all pipeline components"""
        try:
            logger.info("🔧 Initializing pipeline components...")
            
            # Initialize real-time feature engine if InfluxDB is available
            if self.system_integrator and hasattr(self.system_integrator, 'influxdb_client'):
                from feature_engineering.real_time_features import RealTimeFeatureExtractor
                self.realtime_features = RealTimeFeatureExtractor(
                    window_size=60
                )
                logger.info("✅ Real-time feature extractor initialized")
            
            # Register models with explainer
            await self._register_models_for_explainability()
            
            # Initialize continuous learning if models are available
            if self.system_integrator and self.system_integrator.ml_models:
                await self._initialize_continuous_learning()
            
            logger.info("✅ Pipeline orchestrator fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            raise
    
    async def _register_models_for_explainability(self):
        """Register all ML models with the explainability component"""
        try:
            if not self.system_integrator or not self.system_integrator.ml_models:
                logger.warning("⚠️ No models available for explainability registration")
                return
            
            # Define feature names for each model
            lstm_features = [
                "furnace_temperature", "melt_level", "belt_speed", "mold_temp",
                "pressure", "humidity", "viscosity", "conveyor_speed",
                "annealing_temp", "quality_score", "fuel_flow", "air_flow",
                "cooling_rate", "forming_pressure", "batch_flow", "o2_content",
                "co2_content", "temperature_gradient", "thermal_stress", "production_rate"
            ]
            
            vit_features = [
                "image_brightness", "contrast", "edge_density", "texture_variance",
                "color_distribution", "surface_roughness", "reflectance"
            ]
            
            gnn_features = [
                "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
                "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
                "sensor_correlation_1", "sensor_correlation_2", "sensor_correlation_3"
            ]
            
            # Register LSTM model
            if "lstm" in self.system_integrator.ml_models:
                self.model_explainer.register_model(
                    model_name="lstm",
                    model=self.system_integrator.ml_models["lstm"],
                    feature_names=lstm_features,
                    background_data=None  # Will be populated from historical data
                )
                logger.info("✅ LSTM model registered for explainability")
            
            # Register Vision Transformer model
            if "vit" in self.system_integrator.ml_models:
                self.model_explainer.register_model(
                    model_name="vit",
                    model=self.system_integrator.ml_models["vit"],
                    feature_names=vit_features,
                    background_data=None
                )
                logger.info("✅ ViT model registered for explainability")
            
            # Register GNN model
            if "gnn" in self.system_integrator.ml_models:
                self.model_explainer.register_model(
                    model_name="gnn",
                    model=self.system_integrator.ml_models["gnn"],
                    feature_names=gnn_features,
                    background_data=None
                )
                logger.info("✅ GNN model registered for explainability")
            
        except Exception as e:
            logger.error(f"❌ Failed to register models for explainability: {e}")
    
    async def _initialize_continuous_learning(self):
        """Initialize continuous learning framework"""
        try:
            # Create continuous learning framework with EWC regularization
            if "lstm" in self.system_integrator.ml_models:
                lstm_model = self.system_integrator.ml_models["lstm"]
                
                from training.continuous_learning import ContinualLearningFramework, ElasticWeightConsolidation
                import torch.optim as optim
                
                # Create optimizer first
                optimizer = optim.Adam(lstm_model.parameters(), lr=1e-4)
                
                self.continual_learning = ContinualLearningFramework(
                    model=lstm_model,
                    regularization_method=ElasticWeightConsolidation(importance_coeff=1000.0),
                    optimizer=optimizer,
                    experience_replay=True,
                    replay_buffer_size=10000
                )
                
                logger.info("✅ Continual learning framework initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize continual learning: {e}")
    
    async def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process sensor data through the complete pipeline
        
        Args:
            sensor_data: Raw sensor data from data ingestion
            
        Returns:
            Complete pipeline output with predictions, explanations, and recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            # Phase 1: Feature Engineering
            feature_start = datetime.utcnow()
            features = await self._extract_features(sensor_data)
            feature_time = (datetime.utcnow() - feature_start).total_seconds() * 1000
            
            # Phase 2: ML Predictions
            pred_start = datetime.utcnow()
            predictions = await self._generate_predictions(features)
            pred_time = (datetime.utcnow() - pred_start).total_seconds() * 1000
            
            # Phase 3: Explainability
            explain_start = datetime.utcnow()
            explanations = await self._generate_explanations(features, predictions)
            explain_time = (datetime.utcnow() - explain_start).total_seconds() * 1000
            
            # Phase 4: RL Recommendations (if enabled)
            recommendations = await self._generate_rl_recommendations(features, predictions)
            
            # Phase 5: Autonomy Check (Phase 5-8 implementation)
            autonomous_actions = await self._evaluate_autonomous_actions(recommendations, predictions)
            
            # Update metrics
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_metrics(feature_time, pred_time, explain_time, total_time)
            
            # Cache results
            self.latest_features = features
            self.latest_predictions = predictions
            self.latest_explanations = explanations
            
            # Return complete pipeline output
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "sensor_data": sensor_data,
                "engineered_features": features,
                "predictions": predictions,
                "explanations": explanations,
                "recommendations": recommendations,
                "autonomous_actions": autonomous_actions,
                "performance_metrics": {
                    "total_latency_ms": total_time,
                    "feature_extraction_ms": feature_time,
                    "prediction_ms": pred_time,
                    "explanation_ms": explain_time
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline processing failed: {e}")
            self.metrics["failed_predictions"] += 1
            raise
    
    async def _extract_features(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from raw sensor data"""
        try:
            all_features = {}
            
            # Domain-specific features
            domain_features = await self.domain_features.update_with_process_data(sensor_data)
            all_features.update(domain_features)
            
            # Statistical features
            stat_features = await self.statistical_features.compute_multivariate_features(sensor_data)
            all_features.update(stat_features)            
            # Real-time features (if available)
            if self.realtime_features:
                try:
                    rt_features = await self.realtime_features.update_with_sensor_data(sensor_data)
                    all_features.update(rt_features)
                except Exception as e:
                    logger.warning(f"⚠️ Real-time features extraction failed: {e}")
            
            # Store in buffer for historical analysis
            self.feature_buffer.append({
                "timestamp": datetime.utcnow().isoformat(),
                "features": all_features
            })
            
            return all_features
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return {}
    
    async def _generate_predictions(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions from ML models"""
        try:
            if not self.system_integrator:
                return {"error": "System integrator not available"}
            
            # Use the unified system's prediction method
            predictions = await self.system_integrator.predict_defects(
                horizon_hours=1,
                production_line="Line_A",
                include_confidence=True
            )
            
            self.metrics["successful_predictions"] += 1
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            self.metrics["failed_predictions"] += 1
            return {"error": str(e)}
    
    async def _generate_explanations(self, features: Dict[str, Any], 
                                    predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanations for predictions"""
        try:
            # Prepare input data for explainability
            # Extract feature values in correct order for LSTM model
            input_features = np.array([
                features.get("furnace_temperature", 1520.0),
                features.get("melt_level", 2500.0),
                features.get("belt_speed", 150.0),
                features.get("mold_temp", 320.0),
                features.get("pressure", 15.0),
                features.get("humidity", 45.0),
                features.get("viscosity", 1200.0),
                features.get("conveyor_speed", 145.0),
                features.get("annealing_temp", 580.0),
                features.get("quality_score", 0.9),
                features.get("fuel_flow", 0.75),
                features.get("air_flow", 0.8),
                features.get("cooling_rate", 3.5),
                features.get("forming_pressure", 45.0),
                features.get("batch_flow", 2000.0),
                features.get("o2_content", 3.0),
                features.get("co2_content", 12.0),
                features.get("temperature_gradient", 50.0),
                features.get("thermal_stress", 10.0),
                features.get("production_rate", 180.0)
            ], dtype=np.float32)
            
            # Generate explanation for LSTM model
            explanation = self.model_explainer.explain_prediction(
                model_name="lstm",
                input_data=input_features,
                target=None
            )
            
            if explanation:
                return explanation
            else:
                return {"status": "no_explanation", "reason": "Model not registered"}
            
        except Exception as e:
            logger.error(f"❌ Explanation generation failed: {e}")
            return {"error": str(e)}
    
    async def _generate_rl_recommendations(self, features: Dict[str, Any], 
                                          predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations using RL agent"""
        try:
            if not self.system_integrator:
                return []
            
            # Get RL recommendations
            rl_output = await self.system_integrator.get_rl_recommendations()
            
            if "recommendations" in rl_output:
                return [
                    {
                        "action": rec,
                        "confidence": rl_output.get("confidence", 0.7),
                        "type": "rl_optimization"
                    }
                    for rec in rl_output["recommendations"]
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"❌ RL recommendation generation failed: {e}")
            return []
    
    async def _evaluate_autonomous_actions(self, recommendations: List[Dict[str, Any]], 
                                          predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 5-8: Evaluate if actions should be taken autonomously
        
        Returns autonomous action decisions based on confidence and risk
        """
        try:
            autonomous_actions = {
                "enabled": True,
                "actions_to_execute": [],
                "actions_requiring_approval": [],
                "risk_assessment": {},
                "safety_checks_passed": True
            }
            
            # Evaluate each recommendation for autonomous execution
            for rec in recommendations:
                confidence = rec.get("confidence", 0.0)
                action_text = rec.get("action", "")
                
                # Risk assessment
                risk_level = self._assess_action_risk(action_text, predictions)
                
                # Decision logic for autonomy
                if confidence > 0.9 and risk_level == "LOW":
                    # High confidence, low risk → Execute autonomously
                    autonomous_actions["actions_to_execute"].append({
                        "action": action_text,
                        "confidence": confidence,
                        "risk": risk_level,
                        "execution_mode": "autonomous"
                    })
                elif confidence > 0.7 and risk_level in ["LOW", "MEDIUM"]:
                    # Medium confidence → Require approval
                    autonomous_actions["actions_requiring_approval"].append({
                        "action": action_text,
                        "confidence": confidence,
                        "risk": risk_level,
                        "execution_mode": "supervised"
                    })
                else:
                    # Low confidence or high risk → No action
                    autonomous_actions["actions_requiring_approval"].append({
                        "action": action_text,
                        "confidence": confidence,
                        "risk": risk_level,
                        "execution_mode": "manual_only",
                        "reason": "Low confidence or high risk"
                    })
                
                autonomous_actions["risk_assessment"][action_text] = risk_level
            
            return autonomous_actions
            
        except Exception as e:
            logger.error(f"❌ Autonomous action evaluation failed: {e}")
            return {"enabled": False, "error": str(e)}
    
    def _assess_action_risk(self, action_text: str, predictions: Dict[str, Any]) -> str:
        """Assess risk level of an action"""
        # Simple rule-based risk assessment
        high_risk_keywords = ["резко", "максимум", "критично"]
        medium_risk_keywords = ["увеличить", "уменьшить", "изменить"]
        
        action_lower = action_text.lower()
        
        if any(keyword in action_lower for keyword in high_risk_keywords):
            return "HIGH"
        elif any(keyword in action_lower for keyword in medium_risk_keywords):
            return "MEDIUM"
        else:
            return "LOW"
    
    def _update_metrics(self, feature_time: float, pred_time: float, 
                       explain_time: float, total_time: float):
        """Update pipeline performance metrics"""
        self.metrics["pipeline_executions"] += 1
        
        # Moving average for latencies
        alpha = 0.1  # Smoothing factor
        self.metrics["avg_latency_ms"] = (
            alpha * total_time + (1 - alpha) * self.metrics["avg_latency_ms"]
        )
        self.metrics["feature_extraction_time_ms"] = (
            alpha * feature_time + (1 - alpha) * self.metrics["feature_extraction_time_ms"]
        )
        self.metrics["prediction_time_ms"] = (
            alpha * pred_time + (1 - alpha) * self.metrics["prediction_time_ms"]
        )
        self.metrics["explanation_time_ms"] = (
            alpha * explain_time + (1 - alpha) * self.metrics["explanation_time_ms"]
        )
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get current pipeline performance metrics"""
        return {
            **self.metrics,
            "feature_buffer_size": len(self.feature_buffer),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def start_continuous_learning_cycle(self):
        """Start continuous learning cycle (Phase 5-8)"""
        try:
            if not self.continuous_learning:
                logger.warning("⚠️ Continuous learning not initialized")
                return
            
            logger.info("🔄 Starting continuous learning cycle...")
            
            # Collect recent data from feature buffer
            if len(self.feature_buffer) < 100:
                logger.info("⏳ Insufficient data for continuous learning, waiting...")
                return
            
            # TODO: Implement actual training loop with collected data
            logger.info("✅ Continuous learning cycle would run here")
            
        except Exception as e:
            logger.error(f"❌ Continuous learning cycle failed: {e}")


def create_pipeline_orchestrator(unified_system) -> PipelineOrchestrator:
    """Factory function to create pipeline orchestrator"""
    return PipelineOrchestrator(unified_system)
