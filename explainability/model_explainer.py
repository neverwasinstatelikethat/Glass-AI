"""
Model Explainability Integration for Glass Production AI System
Connects explainability components with main prediction models
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from .feature_attribution import EnhancedGlassProductionExplainer
from models.lstm_predictor.attention_lstm import EnhancedAttentionLSTM
from models.vision_transformer.defect_detector import MultiTaskViT
from models.gnn_sensor_network.gnn_model import EnhancedGATSensorGNN, EnhancedSensorGraphAnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedModelExplainer:
    """Integrated explainer for all main prediction models"""
    
    def __init__(self):
        self.explainers = {}
        self.feature_names = {}
        self.model_configs = {}
        
        logger.info("✅ Integrated Model Explainer initialized")
    
    def register_model(self, model_name: str, model: torch.nn.Module, 
                      feature_names: List[str], background_data: Optional[np.ndarray] = None):
        """
        Register a model for explainability
        
        Args:
            model_name: Name of the model
            model: PyTorch model instance
            feature_names: List of feature names
            background_data: Background data for SHAP (optional)
        """
        try:
            # Create explainer for the model
            explainer = EnhancedGlassProductionExplainer(
                model=model,
                feature_names=feature_names,
                device='cpu'
            )
            
            # Store explainer and metadata
            self.explainers[model_name] = explainer
            self.feature_names[model_name] = feature_names
            self.model_configs[model_name] = {
                "model": model,
                "background_data": background_data,
                "feature_names": feature_names
            }
            
            logger.info(f"✅ Model '{model_name}' registered for explainability")
            
        except Exception as e:
            logger.error(f"❌ Failed to register model '{model_name}': {e}")
    
    def _get_feature_ranges(self, feature_names: List[str]) -> Dict[str, Tuple[float, float]]:
        """Get reasonable ranges for features based on domain knowledge"""
        ranges = {}
        
        for name in feature_names:
            if "temperature" in name.lower():
                ranges[name] = (1400.0, 1700.0)  # Furnace temperature range
            elif "speed" in name.lower():
                ranges[name] = (100.0, 200.0)    # Belt speed range
            elif "pressure" in name.lower():
                ranges[name] = (40.0, 60.0)      # Pressure range
            elif "level" in name.lower():
                ranges[name] = (2000.0, 3000.0)  # Melt level range
            elif "o2" in name.lower() or "oxygen" in name.lower():
                ranges[name] = (1.0, 5.0)        # Oxygen content range
            else:
                ranges[name] = (0.0, 100.0)      # Default range
        
        return ranges
    
    def explain_prediction(self, model_name: str, input_data: np.ndarray, 
                          target: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Generate comprehensive explanation for a model prediction
        
        Args:
            model_name: Name of the model
            input_data: Input data for prediction
            target: Target class for classification (optional)
            
        Returns:
            Explanation dictionary or None if failed
        """
        if model_name not in self.explainers:
            logger.warning(f"⚠️ Model '{model_name}' not registered for explainability")
            return None
        
        try:
            explainer = self.explainers[model_name]
            background_data = self.model_configs[model_name].get("background_data")
            
            # Generate comprehensive explanation
            explanation = explainer.explain_comprehensive(
                input_data=input_data,
                background_data=background_data,
                target=target,
                include_counterfactuals=True,
                include_interactions=True
            )
            
            # Add metadata
            explanation_dict = {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "explanation": explanation,
                "quality_score": explanation.explanation_quality
            }
            
            logger.info(f"✅ Generated explanation for model '{model_name}'")
            return explanation_dict
            
        except Exception as e:
            logger.error(f"❌ Failed to generate explanation for model '{model_name}': {e}")
            return None
    
    def explain_ensemble_prediction(self, model_inputs: Dict[str, np.ndarray], 
                                   ensemble_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Generate explanations for ensemble prediction
        
        Args:
            model_inputs: Dictionary of inputs for each model
            ensemble_weights: Weights for each model in ensemble (optional)
            
        Returns:
            Ensemble explanation dictionary
        """
        individual_explanations = {}
        weighted_importance = {}
        
        # Generate explanations for each model
        for model_name, input_data in model_inputs.items():
            if model_name in self.explainers:
                explanation = self.explain_prediction(model_name, input_data)
                if explanation:
                    individual_explanations[model_name] = explanation
                    
                    # Aggregate feature importance with weights
                    if ensemble_weights and model_name in ensemble_weights:
                        weight = ensemble_weights[model_name]
                    else:
                        weight = 1.0 / len(model_inputs)  # Equal weights if not specified
                    
                    feature_importance = explanation["explanation"].feature_importance
                    for feature, importance in feature_importance.items():
                        if feature not in weighted_importance:
                            weighted_importance[feature] = 0.0
                        weighted_importance[feature] += importance * weight
        
        # Sort features by importance
        sorted_features = sorted(
            weighted_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        ensemble_explanation = {
            "timestamp": datetime.now().isoformat(),
            "individual_explanations": individual_explanations,
            "ensemble_feature_importance": dict(sorted_features[:20]),  # Top 20 features
            "ensemble_weights": ensemble_weights or {name: 1.0/len(model_inputs) for name in model_inputs},
            "models_explained": list(individual_explanations.keys())
        }
        
        logger.info(f"✅ Generated ensemble explanation for {len(individual_explanations)} models")
        return ensemble_explanation
    
    def get_model_insights(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get insights about a registered model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model insights dictionary or None if model not found
        """
        if model_name not in self.model_configs:
            logger.warning(f"⚠️ Model '{model_name}' not found")
            return None
        
        model_config = self.model_configs[model_name]
        model = model_config["model"]
        
        insights = {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "feature_names": model_config["feature_names"],
            "feature_count": len(model_config["feature_names"]),
            "has_background_data": model_config["background_data"] is not None,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        return insights
    
    def generate_recommendations(self, explanation: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on explanation
        
        Args:
            explanation: Explanation dictionary
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Extract feature importance
            if "explanation" in explanation:
                expl = explanation["explanation"]
                feature_importance = expl.feature_importance
                
                # Sort by importance
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                
                # Generate recommendations for top features
                for feature, importance in sorted_features[:5]:  # Top 5 features
                    if abs(importance) > 0.1:  # Significant importance threshold
                        if "temperature" in feature.lower():
                            if importance > 0:
                                recommendations.append(f"Monitor {feature} closely - high positive impact on prediction")
                            else:
                                recommendations.append(f"Optimize {feature} - high negative impact on prediction")
                        elif "speed" in feature.lower():
                            recommendations.append(f"Adjust {feature} for optimal performance")
                        elif "pressure" in feature.lower():
                            recommendations.append(f"Maintain stable {feature} levels")
                        else:
                            recommendations.append(f"Feature '{feature}' has significant impact (importance: {importance:.3f})")
            
            # Add quality-based recommendation
            if "quality_score" in explanation:
                quality = explanation["quality_score"]
                if quality < 0.5:
                    recommendations.append("Explanation quality is low - consider collecting more background data")
                elif quality < 0.8:
                    recommendations.append("Explanation quality is moderate - results are reasonably reliable")
                else:
                    recommendations.append("Explanation quality is high - results are highly reliable")
                    
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to error")
        
        return recommendations


def create_integrated_explainer() -> IntegratedModelExplainer:
    """Factory function to create integrated model explainer"""
    explainer = IntegratedModelExplainer()
    logger.info("✅ Integrated Model Explainer created")
    return explainer


# Example usage and integration with main models
def integrate_with_main_models():
    """Example of integrating explainability with main models"""
    print("🧪 Integrating Explainability with Main Models...")
    
    # Create integrated explainer
    explainer = create_integrated_explainer()
    
    # Example: Register LSTM model (this would be the actual trained model)
    try:
        lstm_model = EnhancedAttentionLSTM(
            input_size=20,
            hidden_size=64,
            num_layers=2,
            output_size=5
        )
        
        feature_names = [f"sensor_{i}" for i in range(20)]
        background_data = np.random.randn(100, 20).astype(np.float32)
        
        explainer.register_model(
            "lstm_predictor",
            lstm_model,
            feature_names,
            background_data
        )
        
        print("✅ LSTM model registered for explainability")
        
    except Exception as e:
        print(f"❌ Failed to register LSTM model: {e}")
    
    # Example: Register Vision Transformer model
    try:
        vit_model = VisionTransformer(
            img_size=32,
            patch_size=8,
            num_classes=5
        )
        
        # For image models, we might use different feature names
        feature_names = [f"pixel_{i}" for i in range(32*32*3)]
        
        explainer.register_model(
            "vision_transformer",
            vit_model,
            feature_names
        )
        
        print("✅ Vision Transformer model registered for explainability")
        
    except Exception as e:
        print(f"❌ Failed to register Vision Transformer model: {e}")
    
    # Example: Generate explanation
    if "lstm_predictor" in explainer.explainers:
        test_input = np.random.randn(20).astype(np.float32)
        explanation = explainer.explain_prediction("lstm_predictor", test_input)
        
        if explanation:
            print(f"\n🔍 Explanation generated for LSTM model")
            print(f"   Quality score: {explanation['quality_score']:.2%}")
            
            # Get recommendations
            recommendations = explainer.generate_recommendations(explanation)
            print(f"\n💡 Recommendations:")
            for rec in recommendations[:3]:
                print(f"   • {rec}")
    
    # Get model insights
    insights = explainer.get_model_insights("lstm_predictor")
    if insights:
        print(f"\n📊 Model Insights:")
        print(f"   Model type: {insights['model_type']}")
        print(f"   Features: {insights['feature_count']}")
        print(f"   Parameters: {insights['parameter_count']:,}")
    
    print("\n✅ Integration test completed!")


if __name__ == "__main__":
    integrate_with_main_models()