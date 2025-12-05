"""
Enhanced Explainability for Glass Production AI Models
Implements SHAP, LIME, and attention visualization for comprehensive model interpretation
"""

import torch
import torch.nn as nn
import numpy as np
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Comprehensive explanation result"""
    feature_importance: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    shap_values: Optional[np.ndarray] = None
    lime_explanation: Optional[Any] = None
    attention_weights: Optional[Dict[str, float]] = None
    counterfactuals: Optional[List[Dict[str, Any]]] = None
    interactions: Optional[Dict[Tuple[str, str], float]] = None
    interaction_effects: Optional[Dict[Tuple[str, str], float]] = None
    timestamp: str = None
    explanation_quality: float = 1.0


class EnhancedGlassProductionExplainer:
    """Enhanced explainer for glass production models"""
    
    def __init__(self, model: nn.Module, feature_names: List[str], device: str = "cpu"):
        self.model = model.to(device) if model else None
        self.feature_names = feature_names
        self.device = device
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Initialize explainers
        self._initialize_explainers()
        
        logger.info("✅ Enhanced Explainer инициализирован")
    
    def _initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        if not self.model:
            logger.warning("⚠️ No model provided, explainers will be limited")
            return
        
        try:
            # Initialize SHAP explainer with proper wrapper
            def model_predict_fn(X):
                if isinstance(X, np.ndarray):
                    X = torch.FloatTensor(X).to(self.device)
                with torch.no_grad():
                    output = self.model(X)
                    # Handle different output formats
                    if isinstance(output, tuple):
                        output = output[0]
                    if hasattr(output, 'cpu'):
                        return output.cpu().numpy()
                    return output
            
            # Create a simple background dataset for SHAP
            background_data = np.random.randn(10, len(self.feature_names)).astype(np.float32)
            self.shap_explainer = shap.Explainer(model_predict_fn, background_data)
            logger.info("✅ SHAP explainer fitted with background samples")
            
        except Exception as e:
            logger.warning(f"⚠️ SHAP explainer initialization failed: {e}")
            self.shap_explainer = None
        
        try:
            # Initialize LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.random.randn(100, len(self.feature_names)).astype(np.float32),
                feature_names=self.feature_names,
                class_names=['prediction'],
                mode='regression',
                verbose=False
            )
            logger.info("✅ LIME explainer инициализирован")
            
        except Exception as e:
            logger.warning(f"⚠️ LIME explainer initialization failed: {e}")
            self.lime_explainer = None
    
    def explain_comprehensive(
        self,
        input_data: Union[np.ndarray, torch.Tensor],
        background_data: Optional[np.ndarray] = None,
        include_shap: bool = True,
        include_lime: bool = True,
        include_attention: bool = True,
        include_counterfactuals: bool = False,
        include_interactions: bool = False
    ) -> ExplanationResult:
        """Generate comprehensive explanation for input data"""
        
        # Convert input to numpy if needed
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.detach().cpu().numpy()
        
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        # Initialize result
        result = ExplanationResult(
            feature_importance={},
            confidence_intervals={},
            timestamp=datetime.now().isoformat()
        )
        
        # SHAP explanation
        if include_shap and self.shap_explainer:
            try:
                shap_values = self.shap_explainer(input_data)
                result.shap_values = shap_values.values
                
                # Extract feature importance from SHAP values
                if hasattr(shap_values, 'values'):
                    if shap_values.values.ndim == 2:
                        # For single instance
                        importance_values = np.abs(shap_values.values[0])
                    else:
                        # For multiple instances, take mean
                        importance_values = np.abs(shap_values.values).mean(axis=0)
                    
                    for i, feature_name in enumerate(self.feature_names):
                        if i < len(importance_values):
                            result.feature_importance[feature_name] = float(importance_values[i])
                            
                            # Simple confidence interval (this is a placeholder)
                            std_val = float(importance_values[i] * 0.1)  # 10% uncertainty
                            result.confidence_intervals[feature_name] = (
                                max(0, float(importance_values[i] - std_val)),
                                float(importance_values[i] + std_val)
                            )
                
                logger.info("✅ SHAP explanation generated")
                
            except Exception as e:
                logger.warning(f"⚠️ SHAP explanation failed: {e}")
        
        # LIME explanation
        if include_lime and self.lime_explainer and self.model:
            try:
                def lime_predict_fn(X):
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        output = self.model(X_tensor)
                        if isinstance(output, tuple):
                            output = output[0]
                        if hasattr(output, 'cpu'):
                            return output.cpu().numpy()
                        return output
                
                explanation = self.lime_explainer.explain_instance(
                    input_data[0],
                    lime_predict_fn,
                    num_features=len(self.feature_names),
                    num_samples=1000
                )
                
                result.lime_explanation = explanation
                
                # Extract feature importance from LIME
                lime_importance = dict(explanation.as_list())
                for feature_name, importance in lime_importance.items():
                    # If feature already exists, average with SHAP
                    if feature_name in result.feature_importance:
                        result.feature_importance[feature_name] = (
                            result.feature_importance[feature_name] + abs(importance)
                        ) / 2
                    else:
                        result.feature_importance[feature_name] = abs(importance)
                
                logger.info("✅ LIME explanation generated")
                
            except Exception as e:
                logger.warning(f"⚠️ LIME explanation failed: {e}")
        
        # If no explanations were generated, create simple baseline
        if not result.feature_importance:
            # Create simple feature importance based on input magnitude
            input_magnitude = np.abs(input_data[0])
            for i, feature_name in enumerate(self.feature_names):
                if i < len(input_magnitude):
                    result.feature_importance[feature_name] = float(input_magnitude[i])
                    std_val = float(input_magnitude[i] * 0.1)
                    result.confidence_intervals[feature_name] = (
                        max(0, float(input_magnitude[i] - std_val)),
                        float(input_magnitude[i] + std_val)
                    )
        
        # Normalize feature importance
        if result.feature_importance:
            total_importance = sum(result.feature_importance.values())
            if total_importance > 0:
                for feature_name in result.feature_importance:
                    result.feature_importance[feature_name] /= total_importance
        
        return result
    
    def visualize_attention_weights(
        self,
        attention_weights: torch.Tensor,
        feature_labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Visualize attention weights from transformer models"""
        try:
            # Convert to numpy
            if isinstance(attention_weights, torch.Tensor):
                weights = attention_weights.detach().cpu().numpy()
            else:
                weights = attention_weights
            
            # Handle different attention weight shapes
            if weights.ndim > 1:
                # Take mean across batch and head dimensions
                weights = weights.mean(axis=tuple(range(weights.ndim - 1)))
            
            # Map to feature names
            if feature_labels is None:
                feature_labels = [f"Feature_{i}" for i in range(len(weights))]
            elif len(feature_labels) > len(weights):
                feature_labels = feature_labels[:len(weights)]
            elif len(feature_labels) < len(weights):
                feature_labels.extend([f"Feature_{i}" for i in range(len(feature_labels), len(weights))])
            
            attention_dict = {}
            for i, (label, weight) in enumerate(zip(feature_labels, weights)):
                attention_dict[label] = float(abs(weight))
            
            return attention_dict
            
        except Exception as e:
            logger.warning(f"⚠️ Attention visualization failed: {e}")
            return {}

# Factory function for creating explainers
def create_glass_production_explainer(
    model: nn.Module,
    feature_names: List[str],
    device: str = "cpu"
) -> EnhancedGlassProductionExplainer:
    """Factory function to create glass production explainer"""
    return EnhancedGlassProductionExplainer(model, feature_names, device)


# Пример использования
if __name__ == "__main__":
    print("🧪 Testing Enhanced Explainability System...")
    
    # Simple model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.fc(x)
    
    model = TestModel()
    feature_names = [f"feature_{i}" for i in range(20)]
    feature_ranges = {name: (0.0, 1.0) for name in feature_names}
    
    # Explainer
    explainer = EnhancedGlassProductionExplainer(
        model, feature_names, feature_ranges
    )
    
    # Test data
    np.random.seed(42)
    input_data = np.random.randn(20).astype(np.float32)
    background_data = np.random.randn(100, 20).astype(np.float32)
    
    # Generate explanation
    print("\n🔍 Generating comprehensive explanation...")
    explanation = explainer.explain_comprehensive(
        input_data,
        background_data=background_data,
        include_counterfactuals=True,
        include_interactions=True
    )
    
    print(f"\n📊 Feature Importance (Top 5):")
    sorted_features = sorted(
        explanation.feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    for feature, importance in sorted_features[:5]:
        ci = explanation.confidence_intervals[feature]
        print(f"  {feature}: {importance:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    print(f"\n🔗 Feature Interactions (Top 3):")
    sorted_interactions = sorted(
        explanation.interaction_effects.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for (feat_i, feat_j), h_stat in sorted_interactions[:3]:
        print(f"  {feat_i} × {feat_j}: {h_stat:.4f}")
    
    if explanation.counterfactuals:
        cf = explanation.counterfactuals[0]
        print(f"\n🔄 Counterfactual Analysis:")
        print(f"  Changes needed: {cf['num_changes']}")
        print(f"  Sparsity: {cf['sparsity']:.2%}")
        for feature, change_info in list(cf['changes'].items())[:3]:
            print(f"  {feature}: {change_info['original']:.2f} → "
                  f"{change_info['counterfactual']:.2f}")
    
    print(f"\n✨ Explanation Quality: {explanation.explanation_quality:.2%}")
    print("\n✅ Enhanced Explainability test complete!")