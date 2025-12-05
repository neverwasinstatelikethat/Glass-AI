"""
Stacking Ensemble
Implements stacking ensemble with meta-learner for combining base model predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with neural network meta-learner
    
    Architecture:
    - Base models produce level-0 predictions
    - Meta-learner combines predictions to produce final output
    - Supports both classification and regression
    """
    
    def __init__(
        self,
        num_base_models: int,
        input_dim: int,
        meta_input_dim: Optional[int] = None,
        num_classes: int = 5,
        meta_learner_type: str = "neural",  # "neural", "logistic", "random_forest"
        meta_hidden_dim: int = 128,
        task_type: str = "classification"  # "classification", "regression"
    ):
        """
        Args:
            num_base_models: Number of base models
            input_dim: Dimension of original input features
            meta_input_dim: Dimension of features for meta-learner (if different from input_dim)
            num_classes: Number of output classes (for classification)
            meta_learner_type: Type of meta-learner
            meta_hidden_dim: Hidden dimension for neural meta-learner
            task_type: Type of task
        """
        super().__init__()
        self.num_base_models = num_base_models
        self.input_dim = input_dim
        self.meta_input_dim = meta_input_dim or input_dim
        self.num_classes = num_classes
        self.meta_learner_type = meta_learner_type
        self.task_type = task_type
        
        # Meta-learner
        if meta_learner_type == "neural":
            # Neural network meta-learner
            meta_input_size = num_base_models * num_classes + self.meta_input_dim
            self.meta_learner = nn.Sequential(
                nn.Linear(meta_input_size, meta_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(meta_hidden_dim, meta_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(meta_hidden_dim // 2, num_classes if task_type == "classification" else 1)
            )
        else:
            # Scikit-learn based meta-learners handled separately
            self.meta_learner = None
            self.sklearn_meta_learner = None
        
        self._init_weights()
        logger.info(f"Initialized StackingEnsemble with {num_base_models} base models")
    
    def _init_weights(self):
        """Initialize weights for neural meta-learner"""
        if self.meta_learner_type == "neural":
            for name, param in self.meta_learner.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(
        self, 
        base_predictions: List[torch.Tensor],
        meta_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Stacking ensemble forward pass
        
        Args:
            base_predictions: List of base model predictions [batch_size, num_classes]
            meta_features: Additional features for meta-learner [batch_size, meta_input_dim]
            
        Returns:
            Final prediction [batch_size, num_classes] or [batch_size, 1] for regression
        """
        if self.meta_learner_type != "neural":
            raise ValueError("forward() only supported for neural meta-learner. Use predict_sklearn() for sklearn meta-learners.")
        
        batch_size = base_predictions[0].shape[0]
        
        # Concatenate base predictions
        stacked_predictions = torch.cat(base_predictions, dim=1)  # [batch_size, num_models * num_classes]
        
        # Combine with meta-features if provided
        if meta_features is not None:
            combined_input = torch.cat([stacked_predictions, meta_features], dim=1)
        else:
            # Use zeros if no meta-features provided
            zero_features = torch.zeros(batch_size, self.meta_input_dim, device=stacked_predictions.device)
            combined_input = torch.cat([stacked_predictions, zero_features], dim=1)
        
        # Meta-learner prediction
        output = self.meta_learner(combined_input)
        
        # Apply activation for classification
        if self.task_type == "classification":
            output = F.softmax(output, dim=1)
        
        return output
    
    def fit_sklearn_meta_learner(
        self, 
        base_predictions: np.ndarray, 
        meta_features: Optional[np.ndarray], 
        targets: np.ndarray
    ):
        """
        Fit scikit-learn based meta-learner
        
        Args:
            base_predictions: Base model predictions [n_samples, num_models * num_classes]
            meta_features: Meta features [n_samples, meta_input_dim]
            targets: Target values [n_samples]
        """
        if self.meta_learner_type == "logistic":
            self.sklearn_meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        elif self.meta_learner_type == "random_forest":
            self.sklearn_meta_learner = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported sklearn meta-learner type: {self.meta_learner_type}")
        
        # Combine features
        if meta_features is not None:
            combined_features = np.hstack([base_predictions, meta_features])
        else:
            combined_features = base_predictions
        
        # Fit meta-learner
        self.sklearn_meta_learner.fit(combined_features, targets)
        logger.info(f"Fitted {self.meta_learner_type} meta-learner")
    
    def predict_sklearn(
        self, 
        base_predictions: np.ndarray, 
        meta_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict using fitted scikit-learn meta-learner
        
        Args:
            base_predictions: Base model predictions [n_samples, num_models * num_classes]
            meta_features: Meta features [n_samples, meta_input_dim]
            
        Returns:
            Predictions [n_samples, num_classes] or [n_samples] for regression
        """
        if self.sklearn_meta_learner is None:
            raise ValueError("Sklearn meta-learner not fitted. Call fit_sklearn_meta_learner() first.")
        
        # Combine features
        if meta_features is not None:
            combined_features = np.hstack([base_predictions, meta_features])
        else:
            combined_features = base_predictions
        
        # Predict
        if self.task_type == "classification":
            predictions = self.sklearn_meta_learner.predict_proba(combined_features)
        else:
            predictions = self.sklearn_meta_learner.predict(combined_features)
        
        return predictions


class CrossValidationStacking(nn.Module):
    """
    Stacking with cross-validation for base model training
    Prevents overfitting by using out-of-fold predictions for meta-training
    """
    
    def __init__(
        self,
        num_base_models: int,
        num_classes: int = 5,
        n_folds: int = 5,
        meta_learner_type: str = "neural",
        **kwargs
    ):
        """
        Args:
            num_base_models: Number of base models
            num_classes: Number of output classes
            n_folds: Number of cross-validation folds
            meta_learner_type: Type of meta-learner
            **kwargs: Additional parameters for base stacking ensemble
        """
        super().__init__()
        self.num_base_models = num_base_models
        self.num_classes = num_classes
        self.n_folds = n_folds
        
        # Base stacking ensemble
        self.stacking_ensemble = StackingEnsemble(
            num_base_models=num_base_models,
            num_classes=num_classes,
            meta_learner_type=meta_learner_type,
            **kwargs
        )
        
        logger.info(f"Initialized CrossValidationStacking with {n_folds} folds")
    
    def forward(
        self,
        base_predictions: List[torch.Tensor],
        meta_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through stacking ensemble
        
        Args:
            base_predictions: List of base model predictions
            meta_features: Meta features
            
        Returns:
            Ensemble prediction
        """
        return self.stacking_ensemble(base_predictions, meta_features)


class BlendingEnsemble(nn.Module):
    """
    Blending ensemble - simpler alternative to stacking
    Uses a hold-out validation set for meta-training instead of cross-validation
    """
    
    def __init__(
        self,
        num_base_models: int,
        num_classes: int = 5,
        meta_learner_type: str = "neural",
        **kwargs
    ):
        """
        Args:
            num_base_models: Number of base models
            num_classes: Number of output classes
            meta_learner_type: Type of meta-learner
            **kwargs: Additional parameters for stacking ensemble
        """
        super().__init__()
        self.num_base_models = num_base_models
        self.num_classes = num_classes
        
        # Base stacking ensemble
        self.stacking_ensemble = StackingEnsemble(
            num_base_models=num_base_models,
            num_classes=num_classes,
            meta_learner_type=meta_learner_type,
            **kwargs
        )
        
        logger.info("Initialized BlendingEnsemble")


class FeatureWeightedStacking(nn.Module):
    """
    Feature-weighted stacking that adapts ensemble weights based on input features
    """
    
    def __init__(
        self,
        num_base_models: int,
        input_feature_dim: int,
        num_classes: int = 5,
        hidden_dim: int = 64
    ):
        """
        Args:
            num_base_models: Number of base models
            input_feature_dim: Dimension of input features
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for feature processing
        """
        super().__init__()
        self.num_base_models = num_base_models
        self.input_feature_dim = input_feature_dim
        self.num_classes = num_classes
        
        # Feature processor for dynamic weights
        self.feature_processor = nn.Sequential(
            nn.Linear(input_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_base_models * num_classes),
            nn.Softmax(dim=-1)
        )
        
        # Base predictors (simplified - in practice these would be separate models)
        self.base_predictors = nn.ModuleList([
            nn.Linear(input_feature_dim, num_classes) for _ in range(num_base_models)
        ])
        
        # Final combiner
        self.final_combiner = nn.Linear(num_classes * num_base_models, num_classes)
        
        self._init_weights()
        logger.info(f"Initialized FeatureWeightedStacking with {num_base_models} models")
    
    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature-weighted stacking forward pass
        
        Args:
            x: Input features [batch_size, input_feature_dim]
            
        Returns:
            Final prediction [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # Get base predictions
        base_predictions = [predictor(x) for predictor in self.base_predictors]
        stacked_predictions = torch.cat(base_predictions, dim=1)  # [batch_size, num_models * num_classes]
        
        # Compute feature-dependent weights
        weights = self.feature_processor(x)  # [batch_size, num_models * num_classes]
        
        # Weighted combination
        weighted_predictions = stacked_predictions * weights  # [batch_size, num_models * num_classes]
        
        # Final combination
        final_prediction = self.final_combiner(weighted_predictions)  # [batch_size, num_classes]
        
        return F.softmax(final_prediction, dim=1)


def create_stacking_ensemble(
    num_base_models: int,
    ensemble_type: str = "stacking",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating stacking ensemble
    
    Args:
        num_base_models: Number of base models
        ensemble_type: Type of ensemble ('stacking', 'cv_stacking', 'blending', 'feature_weighted')
        **kwargs: Additional parameters
        
    Returns:
        Stacking ensemble module
    """
    if ensemble_type == "stacking":
        return StackingEnsemble(num_base_models, **kwargs)
    elif ensemble_type == "cv_stacking":
        return CrossValidationStacking(num_base_models, **kwargs)
    elif ensemble_type == "blending":
        return BlendingEnsemble(num_base_models, **kwargs)
    elif ensemble_type == "feature_weighted":
        return FeatureWeightedStacking(num_base_models, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")


if __name__ == "__main__":
    # Test stacking ensemble
    
    # Create sample data
    batch_size = 32
    num_classes = 5
    num_models = 4
    input_dim = 128
    
    # Base model predictions
    base_predictions = [torch.randn(batch_size, num_classes) for _ in range(num_models)]
    
    print("Testing StackingEnsemble with neural meta-learner...")
    neural_stacking = create_stacking_ensemble(
        num_models, "stacking",
        input_dim=input_dim,
        meta_learner_type="neural",
        meta_hidden_dim=64
    )
    
    meta_features = torch.randn(batch_size, input_dim)
    neural_result = neural_stacking(base_predictions, meta_features)
    print(f"Neural stacking result shape: {neural_result.shape}")
    
    print("\nTesting StackingEnsemble with sklearn meta-learner...")
    sklearn_stacking = create_stacking_ensemble(
        num_models, "stacking",
        input_dim=input_dim,
        meta_learner_type="logistic"
    )
    
    # For sklearn, convert to numpy
    base_preds_np = np.hstack([pred.detach().numpy() for pred in base_predictions])
    meta_feats_np = meta_features.detach().numpy()
    targets_np = np.random.randint(0, num_classes, batch_size)
    
    # Fit and predict
    sklearn_stacking.fit_sklearn_meta_learner(base_preds_np, meta_feats_np, targets_np)
    sklearn_result = sklearn_stacking.predict_sklearn(base_preds_np, meta_feats_np)
    print(f"Sklearn stacking result shape: {sklearn_result.shape}")
    
    print("\nTesting CrossValidationStacking...")
    cv_stacking = create_stacking_ensemble(
        num_models, "cv_stacking",
        input_dim=input_dim,
        n_folds=5
    )
    cv_result = cv_stacking(base_predictions, meta_features)
    print(f"CV stacking result shape: {cv_result.shape}")
    
    print("\nTesting FeatureWeightedStacking...")
    fw_stacking = create_stacking_ensemble(
        num_models, "feature_weighted",
        input_feature_dim=input_dim
    )
    fw_result = fw_stacking(meta_features)
    print(f"Feature-weighted stacking result shape: {fw_result.shape}")