"""
Weighted Voting Ensemble
Combines predictions from multiple models using learned or static weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeightedVotingEnsemble(nn.Module):
    """
    Ensemble that combines model predictions using weighted voting
    
    Supports:
    - Static weights
    - Learnable weights
    - Confidence-based weighting
    - Rank-based voting
    """
    
    def __init__(
        self,
        num_models: int,
        num_classes: int = 5,
        weight_type: str = "learnable",  # "static", "learnable", "confidence", "rank"
        static_weights: Optional[List[float]] = None,
        temperature: float = 1.0
    ):
        """
        Args:
            num_models: Number of models in ensemble
            num_classes: Number of output classes
            weight_type: How to compute weights
            static_weights: Predefined static weights (for static type)
            temperature: Temperature for softmax weighting
        """
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.weight_type = weight_type
        self.temperature = temperature
        
        if weight_type == "static":
            if static_weights is None:
                static_weights = [1.0 / num_models] * num_models
            self.register_buffer('static_weights', torch.tensor(static_weights))
        elif weight_type == "learnable":
            # Learnable weights with softmax normalization
            self.learnable_weights = nn.Parameter(torch.ones(num_models))
        elif weight_type == "confidence":
            # Confidence-based weighting computed dynamically
            pass
        elif weight_type == "rank":
            # Rank-based weighting computed dynamically
            pass
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")
        
        logger.info(f"Initialized WeightedVotingEnsemble with {num_models} models, type: {weight_type}")
    
    def forward(
        self, 
        predictions: List[torch.Tensor],
        confidences: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Combine predictions using weighted voting
        
        Args:
            predictions: List of model predictions [batch_size, num_classes]
            confidences: Optional list of model confidences [batch_size]
            
        Returns:
            Ensemble prediction [batch_size, num_classes]
        """
        batch_size = predictions[0].shape[0]
        
        # Compute weights based on type
        if self.weight_type == "static":
            weights = self.static_weights.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_models]
        elif self.weight_type == "learnable":
            weights = F.softmax(self.learnable_weights / self.temperature, dim=0)
            weights = weights.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_models]
        elif self.weight_type == "confidence":
            if confidences is None:
                raise ValueError("Confidences required for confidence-based weighting")
            weights = self._compute_confidence_weights(confidences)  # [batch_size, num_models]
        elif self.weight_type == "rank":
            weights = self._compute_rank_weights(predictions)  # [batch_size, num_models]
        else:
            raise ValueError(f"Unknown weight_type: {self.weight_type}")
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=1)  # [batch_size, num_models, num_classes]
        
        # Weighted average
        weights_expanded = weights.unsqueeze(-1)  # [batch_size, num_models, 1]
        ensemble_pred = (stacked_preds * weights_expanded).sum(dim=1)  # [batch_size, num_classes]
        
        return ensemble_pred
    
    def _compute_confidence_weights(self, confidences: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute weights based on model confidences
        
        Args:
            confidences: List of confidence scores [batch_size]
            
        Returns:
            Weights [batch_size, num_models]
        """
        # Stack confidences
        stacked_confs = torch.stack(confidences, dim=1)  # [batch_size, num_models]
        
        # Softmax normalize
        weights = F.softmax(stacked_confs / self.temperature, dim=1)
        
        return weights
    
    def _compute_rank_weights(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute weights based on model prediction ranks
        
        Args:
            predictions: List of predictions [batch_size, num_classes]
            
        Returns:
            Weights [batch_size, num_models]
        """
        batch_size = predictions[0].shape[0]
        
        # Convert to probabilities if logits
        probs = [F.softmax(pred, dim=1) for pred in predictions]
        
        # Get max probabilities (confidence proxy)
        max_probs = [prob.max(dim=1)[0] for prob in probs]  # List of [batch_size]
        stacked_probs = torch.stack(max_probs, dim=1)  # [batch_size, num_models]
        
        # Rank models by confidence (higher confidence = higher rank)
        # Use negative values for ascending sort (lowest first)
        sorted_indices = torch.argsort(stacked_probs, dim=1, descending=False)
        
        # Assign rank weights (higher rank = higher weight)
        ranks = torch.zeros_like(stacked_probs)
        for i in range(batch_size):
            ranks[i, sorted_indices[i]] = torch.arange(self.num_models, dtype=torch.float32, device=stacked_probs.device)
        
        # Normalize ranks to weights
        weights = F.softmax(ranks / self.temperature, dim=1)
        
        return weights


class DynamicWeightedEnsemble(nn.Module):
    """
    Dynamic ensemble with adaptive weight adjustment based on performance
    """
    
    def __init__(
        self,
        num_models: int,
        num_classes: int = 5,
        adaptation_rate: float = 0.01,
        performance_window: int = 100
    ):
        """
        Args:
            num_models: Number of models in ensemble
            num_classes: Number of output classes
            adaptation_rate: Learning rate for weight updates
            performance_window: Window size for performance tracking
        """
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        
        # Initialize uniform weights
        self.register_buffer('weights', torch.ones(num_models) / num_models)
        
        # Performance tracking
        self.register_buffer('performance_history', torch.zeros(num_models, performance_window))
        self.register_buffer('history_pointer', torch.tensor(0))
        self.register_buffer('sample_count', torch.tensor(0))
        
        logger.info(f"Initialized DynamicWeightedEnsemble with {num_models} models")
    
    def forward(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Dynamic ensemble prediction
        
        Args:
            predictions: List of model predictions [batch_size, num_classes]
            
        Returns:
            Ensemble prediction [batch_size, num_classes]
        """
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=1)  # [batch_size, num_models, num_classes]
        
        # Apply current weights
        weights_expanded = self.weights.unsqueeze(0).unsqueeze(-1)  # [1, num_models, 1]
        ensemble_pred = (stacked_preds * weights_expanded).sum(dim=1)  # [batch_size, num_classes]
        
        return ensemble_pred
    
    def update_weights(self, predictions: List[torch.Tensor], targets: torch.Tensor):
        """
        Update model weights based on recent performance
        
        Args:
            predictions: List of model predictions
            targets: Ground truth labels
        """
        # Compute accuracies for each model
        accuracies = []
        for pred in predictions:
            preds_class = pred.argmax(dim=1)
            accuracy = (preds_class == targets).float().mean()
            accuracies.append(accuracy)
        
        # Update performance history
        self._update_performance_history(torch.tensor(accuracies))
        
        # Update weights based on recent performance
        self._adapt_weights()
    
    def _update_performance_history(self, accuracies: torch.Tensor):
        """
        Update performance history buffer
        
        Args:
            accuracies: Recent accuracies for each model
        """
        self.performance_history[:, self.history_pointer] = accuracies
        self.history_pointer = (self.history_pointer + 1) % self.performance_window
        self.sample_count = min(self.sample_count + 1, self.performance_window)
    
    def _adapt_weights(self):
        """
        Adapt model weights based on performance history
        """
        if self.sample_count < 10:  # Need sufficient history
            return
        
        # Compute average performance
        avg_performance = self.performance_history[:, :self.sample_count].mean(dim=1)
        
        # Update weights using exponential moving average
        performance_gain = avg_performance - self.weights
        self.weights += self.adaptation_rate * performance_gain
        
        # Ensure weights remain positive and normalized
        self.weights = F.relu(self.weights)
        self.weights = self.weights / (self.weights.sum() + 1e-8)
        
        # Log weight updates
        logger.debug(f"Updated ensemble weights: {self.weights.tolist()}")


class SelectiveEnsemble(nn.Module):
    """
    Selective ensemble that chooses subset of models based on input characteristics
    """
    
    def __init__(
        self,
        num_models: int,
        input_feature_dim: int,
        num_classes: int = 5,
        selector_hidden_dim: int = 64
    ):
        """
        Args:
            num_models: Number of models in ensemble
            input_feature_dim: Dimension of input features for selector
            num_classes: Number of output classes
            selector_hidden_dim: Hidden dimension for model selector
        """
        super().__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        
        # Model selector network
        self.selector = nn.Sequential(
            nn.Linear(input_feature_dim, selector_hidden_dim),
            nn.ReLU(),
            nn.Linear(selector_hidden_dim, selector_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(selector_hidden_dim // 2, num_models),
            nn.Softmax(dim=-1)
        )
        
        logger.info(f"Initialized SelectiveEnsemble with {num_models} models")
    
    def forward(
        self, 
        predictions: List[torch.Tensor], 
        input_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective ensemble prediction
        
        Args:
            predictions: List of model predictions [batch_size, num_classes]
            input_features: Features for model selection [batch_size, input_feature_dim]
            
        Returns:
            Tuple of (ensemble prediction, selection weights)
        """
        batch_size = predictions[0].shape[0]
        
        # Compute selection weights
        selection_weights = self.selector(input_features)  # [batch_size, num_models]
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=1)  # [batch_size, num_models, num_classes]
        
        # Weighted average
        weights_expanded = selection_weights.unsqueeze(-1)  # [batch_size, num_models, 1]
        ensemble_pred = (stacked_preds * weights_expanded).sum(dim=1)  # [batch_size, num_classes]
        
        return ensemble_pred, selection_weights


def create_weighted_ensemble(
    num_models: int,
    ensemble_type: str = "weighted_voting",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating weighted ensemble
    
    Args:
        num_models: Number of models in ensemble
        ensemble_type: Type of ensemble ('weighted_voting', 'dynamic', 'selective')
        **kwargs: Additional parameters
        
    Returns:
        Ensemble module
    """
    if ensemble_type == "weighted_voting":
        return WeightedVotingEnsemble(num_models, **kwargs)
    elif ensemble_type == "dynamic":
        return DynamicWeightedEnsemble(num_models, **kwargs)
    elif ensemble_type == "selective":
        return SelectiveEnsemble(num_models, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")


if __name__ == "__main__":
    # Test weighted ensemble
    
    # Create sample data
    batch_size = 32
    num_classes = 5
    num_models = 4
    
    # Model predictions
    predictions = [torch.randn(batch_size, num_classes) for _ in range(num_models)]
    
    print("Testing WeightedVotingEnsemble...")
    # Static weights
    static_ensemble = create_weighted_ensemble(
        num_models, "weighted_voting", 
        weight_type="static", 
        static_weights=[0.1, 0.2, 0.3, 0.4]
    )
    static_result = static_ensemble(predictions)
    print(f"Static ensemble result shape: {static_result.shape}")
    
    # Learnable weights
    learnable_ensemble = create_weighted_ensemble(
        num_models, "weighted_voting",
        weight_type="learnable"
    )
    learnable_result = learnable_ensemble(predictions)
    print(f"Learnable ensemble result shape: {learnable_result.shape}")
    print(f"Learnable weights: {F.softmax(learnable_ensemble.learnable_weights, dim=0)}")
    
    # Confidence-based weights
    confidences = [torch.randn(batch_size) for _ in range(num_models)]
    confidence_ensemble = create_weighted_ensemble(
        num_models, "weighted_voting",
        weight_type="confidence"
    )
    confidence_result = confidence_ensemble(predictions, confidences)
    print(f"Confidence ensemble result shape: {confidence_result.shape}")
    
    print("\nTesting DynamicWeightedEnsemble...")
    dynamic_ensemble = create_weighted_ensemble(num_models, "dynamic")
    dynamic_result = dynamic_ensemble(predictions)
    print(f"Dynamic ensemble result shape: {dynamic_result.shape}")
    
    # Update weights with dummy targets
    dummy_targets = torch.randint(0, num_classes, (batch_size,))
    dynamic_ensemble.update_weights(predictions, dummy_targets)
    print(f"Dynamic weights after update: {dynamic_ensemble.weights}")
    
    print("\nTesting SelectiveEnsemble...")
    input_feature_dim = 128
    selective_ensemble = create_weighted_ensemble(
        num_models, "selective",
        input_feature_dim=input_feature_dim
    )
    input_features = torch.randn(batch_size, input_feature_dim)
    selective_result, selection_weights = selective_ensemble(predictions, input_features)
    print(f"Selective ensemble result shape: {selective_result.shape}")
    print(f"Selection weights shape: {selection_weights.shape}")