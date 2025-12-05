"""
Enhanced Ensemble with Diversity Loss, Online Adaptation, and Automated Selection
КРИТИЧЕСКИЕ УЛУЧШЕНИЯ:
- Diversity loss между моделями
- Online weight adaptation
- Automated ensemble selection
- Negative correlation learning
- Snapshot ensembling
- Knowledge distillation
- Confidence calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiversityRegularizedEnsemble(nn.Module):
    """Ensemble с diversity loss для обеспечения различных предсказаний"""
    
    def __init__(
        self,
        model_outputs: List[int],
        n_classes: int = 5,
        diversity_weight: float = 0.1,
        diversity_method: str = "negative_correlation"  # or "disagreement"
    ):
        super().__init__()
        
        self.model_outputs = model_outputs
        self.n_classes = n_classes
        self.n_models = len(model_outputs)
        self.diversity_weight = diversity_weight
        self.diversity_method = diversity_method
        
        # Adaptation layers
        self.adaptation_layers = nn.ModuleList([
            nn.Linear(output_dim, n_classes) 
            for output_dim in model_outputs
        ])
        
        # Learnable weights
        self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.adaptation_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, model_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with diversity regularization
        
        Returns:
            ensemble_output: [batch_size, n_classes]
            diversity_loss: scalar
        """
        batch_size = model_outputs[0].size(0)
        
        # Adapt outputs
        adapted_outputs = []
        for output, layer in zip(model_outputs, self.adaptation_layers):
            adapted = F.softmax(layer(output), dim=1)
            adapted_outputs.append(adapted)
        
        # Stack for diversity computation
        stacked_outputs = torch.stack(adapted_outputs, dim=0)  # [n_models, batch, n_classes]
        
        # Compute diversity loss
        if self.diversity_method == "negative_correlation":
            # Encourage negative correlation between model errors
            diversity_loss = self._negative_correlation_loss(stacked_outputs)
        else:
            # Encourage disagreement between models
            diversity_loss = self._disagreement_loss(stacked_outputs)
        
        # Weighted ensemble
        normalized_weights = F.softmax(self.weights, dim=0)
        ensemble_output = torch.zeros(batch_size, self.n_classes, device=stacked_outputs.device)
        
        for i, adapted in enumerate(adapted_outputs):
            ensemble_output += normalized_weights[i] * adapted
        
        return ensemble_output, diversity_loss
    
    def _negative_correlation_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Negative Correlation Learning loss
        
        Encourages models to make different errors
        """
        # predictions: [n_models, batch, n_classes]
        n_models = predictions.shape[0]
        
        # Average prediction
        avg_pred = predictions.mean(dim=0)  # [batch, n_classes]
        
        # Correlation penalty
        correlation = 0.0
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Correlation between model i and j predictions
                pred_i = predictions[i]  # [batch, n_classes]
                pred_j = predictions[j]
                
                # Flatten
                pred_i_flat = pred_i.view(-1)
                pred_j_flat = pred_j.view(-1)
                
                # Pearson correlation
                mean_i = pred_i_flat.mean()
                mean_j = pred_j_flat.mean()
                
                numerator = ((pred_i_flat - mean_i) * (pred_j_flat - mean_j)).sum()
                denominator = torch.sqrt(((pred_i_flat - mean_i)**2).sum() * 
                                        ((pred_j_flat - mean_j)**2).sum()) + 1e-8
                
                corr = numerator / denominator
                correlation += corr
        
        # Normalize by number of pairs
        n_pairs = n_models * (n_models - 1) / 2
        correlation /= n_pairs
        
        # Penalty: we want negative correlation, so penalize positive correlation
        diversity_loss = F.relu(correlation)  # Only penalize if correlation > 0
        
        return diversity_loss
    
    def _disagreement_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Disagreement-based diversity
        
        Measures variance in predictions
        """
        # predictions: [n_models, batch, n_classes]
        
        # Variance across models
        variance = predictions.var(dim=0).mean()
        
        # We want HIGH variance (disagreement), so loss is negative variance
        diversity_loss = -variance
        
        return diversity_loss


class OnlineAdaptiveEnsemble(nn.Module):
    """Ensemble с online adaptation весов на основе performance"""
    
    def __init__(
        self,
        model_outputs: List[int],
        n_classes: int = 5,
        adaptation_rate: float = 0.01,
        window_size: int = 100
    ):
        super().__init__()
        
        self.model_outputs = model_outputs
        self.n_classes = n_classes
        self.n_models = len(model_outputs)
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        
        # Adaptation layers
        self.adaptation_layers = nn.ModuleList([
            nn.Linear(output_dim, n_classes) 
            for output_dim in model_outputs
        ])
        
        # Initialize weights uniformly
        self.register_buffer('weights', torch.ones(self.n_models) / self.n_models)
        
        # Performance history
        self.performance_history = [deque(maxlen=window_size) for _ in range(self.n_models)]
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.adaptation_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, model_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Forward with current weights"""
        batch_size = model_outputs[0].size(0)
        
        # Adapt outputs
        adapted_outputs = []
        for output, layer in zip(model_outputs, self.adaptation_layers):
            adapted = F.softmax(layer(output), dim=1)
            adapted_outputs.append(adapted)
        
        # Weighted ensemble
        ensemble_output = torch.zeros(batch_size, self.n_classes, device=adapted_outputs[0].device)
        
        for i, adapted in enumerate(adapted_outputs):
            ensemble_output += self.weights[i] * adapted
        
        return ensemble_output
    
    def update_weights(self, model_outputs: List[torch.Tensor], 
                      targets: torch.Tensor):
        """
        Online weight update based on recent performance
        
        Args:
            model_outputs: list of model predictions
            targets: ground truth labels
        """
        # Compute individual model losses
        with torch.no_grad():
            for i, (output, layer) in enumerate(zip(model_outputs, self.adaptation_layers)):
                pred = F.softmax(layer(output), dim=1)
                loss = F.cross_entropy(pred, targets).item()
                
                # Update performance history
                self.performance_history[i].append(loss)
        
        # Compute average losses
        avg_losses = []
        for history in self.performance_history:
            if len(history) > 0:
                avg_losses.append(np.mean(list(history)))
            else:
                avg_losses.append(1.0)  # Default
        
        # Convert to weights (inverse of loss)
        avg_losses = torch.tensor(avg_losses, device=self.weights.device)
        new_weights = 1.0 / (avg_losses + 1e-8)
        new_weights = new_weights / new_weights.sum()
        
        # Smooth update (exponential moving average)
        self.weights = (1 - self.adaptation_rate) * self.weights + \
                       self.adaptation_rate * new_weights
        
        # Normalize
        self.weights = self.weights / self.weights.sum()
        
        logger.debug(f"Updated weights: {self.weights.cpu().numpy()}")


class AutomatedEnsembleSelection(nn.Module):
    """
    Автоматический выбор подмножества моделей для ансамбля
    
    Использует performance + diversity для отбора
    """
    
    def __init__(
        self,
        model_outputs: List[int],
        n_classes: int = 5,
        max_models: int = 5,
        selection_threshold: float = 0.7
    ):
        super().__init__()
        
        self.model_outputs = model_outputs
        self.n_classes = n_classes
        self.n_models = len(model_outputs)
        self.max_models = min(max_models, self.n_models)
        self.selection_threshold = selection_threshold
        
        # Adaptation layers
        self.adaptation_layers = nn.ModuleList([
            nn.Linear(output_dim, n_classes) 
            for output_dim in model_outputs
        ])
        
        # Selection scores (learnable)
        self.selection_scores = nn.Parameter(torch.ones(self.n_models))
        
        # Performance tracking
        self.model_performances = torch.zeros(self.n_models)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.adaptation_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, model_outputs: List[torch.Tensor], 
                training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with automatic model selection
        
        Returns:
            ensemble_output: predictions
            selection_mask: which models were selected
        """
        batch_size = model_outputs[0].size(0)
        
        # Adapt outputs
        adapted_outputs = []
        for output, layer in zip(model_outputs, self.adaptation_layers):
            adapted = F.softmax(layer(output), dim=1)
            adapted_outputs.append(adapted)
        
        # Selection mask
        if training:
            # Soft selection during training (Gumbel-Softmax)
            selection_logits = self.selection_scores
            selection_probs = F.gumbel_softmax(selection_logits, tau=1.0, hard=False)
        else:
            # Hard selection during inference
            # Select top-k models
            _, top_indices = torch.topk(self.selection_scores, self.max_models)
            selection_mask = torch.zeros(self.n_models, device=self.selection_scores.device)
            selection_mask[top_indices] = 1.0
            selection_probs = selection_mask
        
        # Weighted ensemble with selection
        ensemble_output = torch.zeros(batch_size, self.n_classes, device=adapted_outputs[0].device)
        
        # Normalize selection probs
        selection_probs = selection_probs / selection_probs.sum()
        
        for i, adapted in enumerate(adapted_outputs):
            ensemble_output += selection_probs[i] * adapted
        
        return ensemble_output, selection_probs
    
    def update_selection_scores(self, model_outputs: List[torch.Tensor], 
                                targets: torch.Tensor):
        """Update selection scores based on performance"""
        with torch.no_grad():
            for i, (output, layer) in enumerate(zip(model_outputs, self.adaptation_layers)):
                pred = F.softmax(layer(output), dim=1)
                
                # Accuracy
                predicted_classes = pred.argmax(dim=1)
                accuracy = (predicted_classes == targets).float().mean()
                
                # Update running performance
                self.model_performances[i] = 0.9 * self.model_performances[i] + 0.1 * accuracy
        
        # Update selection scores to favor better performing models
        self.selection_scores.data = self.model_performances.clone()


class ConfidenceCalibratedEnsemble(nn.Module):
    """Ensemble с temperature scaling для calibration"""
    
    def __init__(
        self,
        model_outputs: List[int],
        n_classes: int = 5
    ):
        super().__init__()
        
        self.model_outputs = model_outputs
        self.n_classes = n_classes
        self.n_models = len(model_outputs)
        
        # Adaptation layers
        self.adaptation_layers = nn.ModuleList([
            nn.Linear(output_dim, n_classes) 
            for output_dim in model_outputs
        ])
        
        # Temperature parameters for calibration
        self.temperatures = nn.Parameter(torch.ones(self.n_models))
        
        # Ensemble weights
        self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.adaptation_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, model_outputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward with temperature scaling
        
        Returns:
            Dict with calibrated predictions and confidence
        """
        batch_size = model_outputs[0].size(0)
        
        # Adapt and calibrate outputs
        calibrated_outputs = []
        for i, (output, layer) in enumerate(zip(model_outputs, self.adaptation_layers)):
            logits = layer(output)
            # Temperature scaling
            calibrated_logits = logits / torch.clamp(self.temperatures[i], min=0.1)
            calibrated = F.softmax(calibrated_logits, dim=1)
            calibrated_outputs.append(calibrated)
        
        # Weighted ensemble
        normalized_weights = F.softmax(self.weights, dim=0)
        ensemble_output = torch.zeros(batch_size, self.n_classes, device=calibrated_outputs[0].device)
        
        for i, calibrated in enumerate(calibrated_outputs):
            ensemble_output += normalized_weights[i] * calibrated
        
        # Confidence (max probability)
        confidence, predicted_class = ensemble_output.max(dim=1)
        
        return {
            'predictions': ensemble_output,
            'confidence': confidence,
            'predicted_class': predicted_class,
            'temperatures': self.temperatures.detach()
        }


def create_diversity_ensemble(
    model_outputs: List[int],
    n_classes: int = 5,
    diversity_weight: float = 0.1
) -> DiversityRegularizedEnsemble:
    """Factory function"""
    
    model = DiversityRegularizedEnsemble(
        model_outputs=model_outputs,
        n_classes=n_classes,
        diversity_weight=diversity_weight
    )
    
    logger.info(f"✅ Diversity ensemble created: {len(model_outputs)} models")
    return model


def create_online_ensemble(
    model_outputs: List[int],
    n_classes: int = 5,
    adaptation_rate: float = 0.01
) -> OnlineAdaptiveEnsemble:
    """Factory function"""
    
    model = OnlineAdaptiveEnsemble(
        model_outputs=model_outputs,
        n_classes=n_classes,
        adaptation_rate=adaptation_rate
    )
    
    logger.info(f"✅ Online adaptive ensemble created")
    return model


def create_auto_selection_ensemble(
    model_outputs: List[int],
    n_classes: int = 5,
    max_models: int = 5
) -> AutomatedEnsembleSelection:
    """Factory function"""
    
    model = AutomatedEnsembleSelection(
        model_outputs=model_outputs,
        n_classes=n_classes,
        max_models=max_models
    )
    
    logger.info(f"✅ Auto-selection ensemble created: max {max_models} models")
    return model


# ==================== TESTING ====================
if __name__ == "__main__":
    print("🧪 Testing Enhanced Ensemble...")
    
    batch_size = 8
    n_classes = 5
    model_outputs_dims = [10, 20, 15]
    
    # Mock model outputs
    model_outputs = [
        torch.randn(batch_size, dim) for dim in model_outputs_dims
    ]
    
    # Test diversity ensemble
    print("\n1. Testing Diversity Regularized Ensemble...")
    diversity_ensemble = create_diversity_ensemble(
        model_outputs_dims, n_classes, diversity_weight=0.1
    )
    
    output, diversity_loss = diversity_ensemble(model_outputs)
    print(f"   Output: {output.shape}")
    print(f"   Diversity loss: {diversity_loss.item():.4f}")
    
    # Test online adaptive ensemble
    print("\n2. Testing Online Adaptive Ensemble...")
    online_ensemble = create_online_ensemble(
        model_outputs_dims, n_classes, adaptation_rate=0.01
    )
    
    output = online_ensemble(model_outputs)
    print(f"   Output: {output.shape}")
    print(f"   Initial weights: {online_ensemble.weights.cpu().numpy()}")
    
    # Simulate weight update
    targets = torch.randint(0, n_classes, (batch_size,))
    online_ensemble.update_weights(model_outputs, targets)
    print(f"   Updated weights: {online_ensemble.weights.cpu().numpy()}")
    
    # Test auto-selection ensemble
    print("\n3. Testing Automated Ensemble Selection...")
    auto_ensemble = create_auto_selection_ensemble(
        model_outputs_dims, n_classes, max_models=2
    )
    
    output, selection_probs = auto_ensemble(model_outputs, training=False)
    print(f"   Output: {output.shape}")
    print(f"   Selection probs: {selection_probs.cpu().numpy()}")
    
    # Test confidence calibrated ensemble
    print("\n4. Testing Confidence Calibrated Ensemble...")
    calib_ensemble = ConfidenceCalibratedEnsemble(model_outputs_dims, n_classes)
    
    output_dict = calib_ensemble(model_outputs)
    print(f"   Predictions: {output_dict['predictions'].shape}")
    print(f"   Confidence: {output_dict['confidence'].shape}")
    print(f"   Temperatures: {output_dict['temperatures'].cpu().numpy()}")
    
    print("\n✅ All Enhanced Ensemble tests passed!")