"""
ML Metrics Tracker with MLflow, TensorBoard, and W&B Integration
Tracks comprehensive metrics for model training including F1 score, precision, recall
"""

import logging
import os
import json
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_auc_score, mean_squared_error,
    mean_absolute_error, r2_score
)
import torch

logger = logging.getLogger(__name__)


class MLMetricsTracker:
    """
    Comprehensive ML metrics tracker with support for multiple backends:
    - MLflow
    - TensorBoard
    - Weights & Biases (W&B)
    - JSON export for custom dashboards
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        use_mlflow: bool = True,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        output_dir: str = "ml_metrics"
    ):
        """
        Initialize metrics tracker
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI (default: local file storage)
            use_mlflow: Enable MLflow tracking
            use_tensorboard: Enable TensorBoard tracking
            use_wandb: Enable Weights & Biases tracking
            output_dir: Directory for metrics output
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.use_mlflow = use_mlflow
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics_history = {
            'train': [],
            'validation': [],
            'test': []
        }
        
        # Initialize MLflow
        if self.use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                else:
                    # Use local path for Windows compatibility (not file:// URI)
                    mlruns_dir = os.path.join(os.path.abspath(output_dir), 'mlruns')
                    os.makedirs(mlruns_dir, exist_ok=True)
                    mlflow.set_tracking_uri(mlruns_dir)
                
                mlflow.set_experiment(experiment_name)
                self.mlflow_run = mlflow.start_run()
                logger.info(f"✅ MLflow tracking enabled: {mlflow.get_tracking_uri()}")
            except ImportError:
                logger.warning("❌ MLflow not installed. Install with: pip install mlflow")
                self.use_mlflow = False
            except Exception as e:
                logger.warning(f"❌ MLflow initialization failed: {e}. Disabling MLflow.")
                self.use_mlflow = False
        
        # Initialize TensorBoard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(output_dir, 'tensorboard', experiment_name)
                self.tensorboard_writer = SummaryWriter(log_dir=tb_dir)
                logger.info(f"✅ TensorBoard tracking enabled: {tb_dir}")
            except ImportError:
                logger.warning("❌ TensorBoard not installed. Install with: pip install tensorboard")
                self.use_tensorboard = False
        
        # Initialize Weights & Biases
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project="glass-production-ml", name=experiment_name)
                logger.info("✅ Weights & Biases tracking enabled")
            except ImportError:
                logger.warning("❌ W&B not installed. Install with: pip install wandb")
                self.use_wandb = False
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        if self.use_mlflow:
            self.mlflow.log_params(params)
        
        if self.use_wandb:
            self.wandb.config.update(params)
        
        # Save to JSON
        params_file = os.path.join(self.output_dir, f"{self.experiment_name}_params.json")
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"📊 Logged parameters: {params}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = 'train'
    ):
        """
        Log metrics for current step
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Current training step/epoch
            phase: Phase of training ('train', 'validation', 'test')
        """
        metrics['step'] = step
        metrics['phase'] = phase
        metrics['timestamp'] = datetime.utcnow().isoformat()
        
        # Add to history
        self.metrics_history[phase].append(metrics.copy())
        
        # MLflow
        if self.use_mlflow:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(f"{phase}_{name}", value, step=step)
        
        # TensorBoard
        if self.use_tensorboard:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f"{phase}/{name}", value, step)
        
        # Weights & Biases
        if self.use_wandb:
            wandb_metrics = {f"{phase}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
            self.wandb.log(wandb_metrics, step=step)
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for AUC)
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        if len(np.unique(y_true)) > 2:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
                metrics[f'precision_class_{i}'] = float(p)
                metrics[f'recall_class_{i}'] = float(r)
                metrics[f'f1_score_class_{i}'] = float(f)
        
        # AUC if probabilities provided
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
        
        return metrics
    
    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        mask = y_true != 0
        if mask.sum() > 0:
            metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        return metrics
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        step: int = 0
    ):
        """Log confusion matrix to TensorBoard"""
        if self.use_tensorboard:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=class_names, yticklabels=class_names)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            
            self.tensorboard_writer.add_figure('confusion_matrix', fig, step)
            plt.close()
    
    def save_metrics_summary(self):
        """Save comprehensive metrics summary to JSON"""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.utcnow().isoformat(),
            'metrics_history': self.metrics_history
        }
        
        # Calculate best metrics
        if self.metrics_history['validation']:
            val_metrics = self.metrics_history['validation']
            
            # Find best epoch based on different criteria
            if 'f1_score' in val_metrics[0]:
                best_f1_idx = max(range(len(val_metrics)), key=lambda i: val_metrics[i].get('f1_score', 0))
                summary['best_f1_epoch'] = val_metrics[best_f1_idx]
            
            if 'accuracy' in val_metrics[0]:
                best_acc_idx = max(range(len(val_metrics)), key=lambda i: val_metrics[i].get('accuracy', 0))
                summary['best_accuracy_epoch'] = val_metrics[best_acc_idx]
            
            if 'loss' in val_metrics[0]:
                best_loss_idx = min(range(len(val_metrics)), key=lambda i: val_metrics[i].get('loss', float('inf')))
                summary['best_loss_epoch'] = val_metrics[best_loss_idx]
        
        summary_file = os.path.join(self.output_dir, f"{self.experiment_name}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"💾 Metrics summary saved to {summary_file}")
        
        return summary
    
    def close(self):
        """Close all tracking backends"""
        if self.use_mlflow and self.mlflow_run:
            self.mlflow.end_run()
        
        if self.use_tensorboard:
            self.tensorboard_writer.close()
        
        if self.use_wandb:
            self.wandb.finish()
        
        # Save final summary
        self.save_metrics_summary()
        
        logger.info("✅ Metrics tracking closed")


def export_metrics_to_prometheus(metrics: Dict[str, float], output_file: str = "metrics.prom"):
    """
    Export metrics in Prometheus format for monitoring
    
    Args:
        metrics: Dictionary of metrics
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                # Convert to Prometheus format
                metric_name = name.replace(' ', '_').replace('-', '_').lower()
                f.write(f"# TYPE {metric_name} gauge\n")
                f.write(f"{metric_name} {value}\n")
    
    logger.info(f"📊 Prometheus metrics exported to {output_file}")


def create_metrics_report(
    metrics_history: Dict[str, List[Dict]],
    output_file: str = "metrics_report.md"
):
    """
    Create a markdown report of training metrics
    
    Args:
        metrics_history: History of metrics
        output_file: Output markdown file
    """
    with open(output_file, 'w') as f:
        f.write("# ML Model Training Report\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")
        
        for phase in ['train', 'validation', 'test']:
            if not metrics_history.get(phase):
                continue
            
            f.write(f"## {phase.capitalize()} Metrics\n\n")
            
            latest = metrics_history[phase][-1]
            f.write("### Latest Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            for key, value in latest.items():
                if isinstance(value, (int, float)):
                    f.write(f"| {key} | {value:.4f} |\n")
            
            f.write("\n")
        
        f.write("\n---\nEnd of Report\n")
    
    logger.info(f"📄 Metrics report saved to {output_file}")
