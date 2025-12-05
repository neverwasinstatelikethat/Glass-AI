"""
Complete Training Pipeline for Glass Production ML Models
Generates datasets, trains LSTM/GNN models with PyTorch, exports to ONNX
With comprehensive metrics tracking (TensorBoard, Prometheus)

Target Metrics:
- LSTM: MAE < 0.05, R² > 0.90
- GNN: Accuracy > 85%, F1 > 80%
"""

import logging
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import onnxruntime as ort
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Import dataset generator
from training.synthetic_training_data_generator import (
    SyntheticTrainingDataGenerator,
    DatasetConfig
)

# Import models
from models.lstm_predictor.attention_lstm import create_enhanced_lstm
from models.gnn_sensor_network.gnn_model import create_dynamic_gnn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {DEVICE}")


# ============= PYTORCH DATASETS =============
class LSTMDataset(Dataset):
    """PyTorch Dataset for LSTM training"""
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        # Only use first 6 values (defect probabilities)
        self.labels = torch.FloatTensor(labels[:, :6])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class GNNDataset(Dataset):
    """PyTorch Dataset for GNN training - Binary classification"""
    def __init__(self, node_features_list: List[np.ndarray], labels: np.ndarray):
        self.node_features_list = node_features_list
        # Binary classification: -1 (no anomaly) -> 0, any sensor_id -> 1 (has anomaly)
        binary_labels = (labels != -1).astype(np.int64)
        self.labels = torch.LongTensor(binary_labels)
    
    def __len__(self):
        return len(self.node_features_list)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.node_features_list[idx]), self.labels[idx]


# ============= GNN WRAPPER FOR ONNX EXPORT =============
class LSTMWrapper(nn.Module):
    """Wrapper for LSTM model for ONNX export (returns tensor, not dict)"""
    def __init__(self, lstm_model):
        super().__init__()
        self.lstm = lstm_model
    
    def forward(self, x):
        output = self.lstm(x)
        return output['prediction']


class GNNWrapper(nn.Module):
    """Wrapper for GNN model for ONNX export"""
    def __init__(self, gnn_model, classifier, num_nodes=20):
        super().__init__()
        self.gnn = gnn_model
        self.classifier = classifier
        self.num_nodes = num_nodes
    
    def forward(self, node_features):
        # Create ring edge index
        num_nodes = node_features.shape[0]
        edge_index = torch.zeros(2, num_nodes, dtype=torch.long, device=node_features.device)
        for i in range(num_nodes):
            edge_index[0, i] = i
            edge_index[1, i] = (i + 1) % num_nodes
        
        edge_attr = torch.ones(num_nodes, 3, device=node_features.device)
        gnn_out, _ = self.gnn(node_features, edge_index, edge_attr, return_attention=False)
        pooled = gnn_out.mean(dim=0, keepdim=True)
        return self.classifier(pooled)


class ONNXModelEvaluator:
    """Evaluator for pre-trained ONNX models with comprehensive metrics tracking"""
    
    def __init__(
        self,
        lstm_model_path: str = "models/lstm_predictor/lstm_model.onnx",
        gnn_model_path: str = "models/gnn_sensor_network/gnn_model.onnx",
        vit_model_path: str = "models/vision_transformer/vit_model.onnx",
        metrics_tracker = None
    ):
        """Initialize ONNX model evaluator with pre-trained models"""
        self.metrics_tracker = metrics_tracker
        
        # Load ONNX models
        logger.info("📦 Loading ONNX models...")
        
        if os.path.exists(lstm_model_path):
            self.lstm_session = ort.InferenceSession(lstm_model_path)
            logger.info(f"✅ LSTM model loaded: {lstm_model_path}")
        else:
            logger.warning(f"⚠️ LSTM model not found: {lstm_model_path}")
            self.lstm_session = None
        
        if os.path.exists(gnn_model_path):
            self.gnn_session = ort.InferenceSession(gnn_model_path)
            logger.info(f"✅ GNN model loaded: {gnn_model_path}")
        else:
            logger.warning(f"⚠️ GNN model not found: {gnn_model_path}")
            self.gnn_session = None
        
        if os.path.exists(vit_model_path):
            self.vit_session = ort.InferenceSession(vit_model_path)
            logger.info(f"✅ ViT model loaded: {vit_model_path}")
        else:
            logger.warning(f"⚠️ ViT model not found: {vit_model_path}")
            self.vit_session = None
        
    def evaluate_lstm_model(
        self,
        test_sequences: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate LSTM ONNX model on test data
        
        Args:
            test_sequences: Test sequences (num_samples, sequence_length, feature_dim)
            test_labels: Test labels (num_samples, output_dim)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.lstm_session is None:
            logger.error("❌ LSTM model not loaded")
            return {}
        
        logger.info(f"🔍 Evaluating LSTM model on {len(test_sequences)} samples...")
        
        # Get input/output names
        input_name = self.lstm_session.get_inputs()[0].name
        output_name = self.lstm_session.get_outputs()[0].name
        
        # Run inference
        predictions = []
        batch_size = 32
        for i in range(0, len(test_sequences), batch_size):
            batch = test_sequences[i:i+batch_size].astype(np.float32)
            output = self.lstm_session.run([output_name], {input_name: batch})
            predictions.append(output[0])
        
        predictions = np.concatenate(predictions, axis=0)
        
        # Extract only the first 6 values from labels (defect probabilities)
        # Ignore the last 3 values (time horizon predictions: 1h, 4h, 24h)
        test_labels_defects = test_labels[:, :6]
        
        # Calculate regression metrics
        mse = np.mean((predictions - test_labels_defects) ** 2)
        mae = np.mean(np.abs(predictions - test_labels_defects))
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((test_labels_defects - predictions) ** 2) / np.sum((test_labels_defects - test_labels_defects.mean()) ** 2))
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2)
        }
        
        # Log metrics
        if self.metrics_tracker:
            self.metrics_tracker.log_metrics(metrics, step=0, phase='test')
        
        logger.info(f"✅ LSTM Evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return metrics
    
    def evaluate_gnn_model(
        self,
        test_node_features: List[np.ndarray],
        test_edge_indices: List[np.ndarray],
        test_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate GNN ONNX model on test data
        
        Args:
            test_node_features: List of node feature arrays (num_nodes, feature_dim)
            test_edge_indices: List of edge index arrays (not used for this ONNX model)
            test_labels: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.gnn_session is None:
            logger.error("❌ GNN model not loaded")
            return {}
        
        logger.info(f"🔍 Evaluating GNN model on {len(test_node_features)} graphs...")
        
        # Get input/output names
        input_name = self.gnn_session.get_inputs()[0].name
        output_name = self.gnn_session.get_outputs()[0].name
        
        # NOTE: The GNN ONNX model (see generate_models.py) only takes node_features
        # It creates edge indices internally in the wrapper
        
        # Run inference
        predictions = []
        for i in range(len(test_node_features)):
            # Only pass node features (the wrapper creates edges internally)
            node_features = test_node_features[i].astype(np.float32)
            output = self.gnn_session.run([output_name], {input_name: node_features})
            predictions.append(np.argmax(output[0]))
        
        predictions = np.array(predictions)
        
        # Adjust labels for classification (shift -1 to 0, others to 1+)
        adjusted_labels = test_labels + 1
        adjusted_preds = predictions
        
        # Filter out ignore index
        valid_mask = adjusted_labels != 0
        
        if valid_mask.sum() > 0:
            acc = accuracy_score(adjusted_labels[valid_mask], adjusted_preds[valid_mask]) * 100
            f1 = f1_score(adjusted_labels[valid_mask], adjusted_preds[valid_mask], average='weighted', zero_division=0) * 100
            precision = precision_score(adjusted_labels[valid_mask], adjusted_preds[valid_mask], average='weighted', zero_division=0) * 100
            recall = recall_score(adjusted_labels[valid_mask], adjusted_preds[valid_mask], average='weighted', zero_division=0) * 100
        else:
            acc = f1 = precision = recall = 0.0
        
        metrics = {
            'accuracy': float(acc),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall)
        }
        
        # Log metrics
        if self.metrics_tracker:
            self.metrics_tracker.log_metrics(metrics, step=0, phase='test')
            # Don't log confusion matrix to avoid plotting errors with TensorBoard
            # self.metrics_tracker.log_confusion_matrix(
            #     adjusted_labels[valid_mask],
            #     adjusted_preds[valid_mask],
            #     step=0
            # )
        
        logger.info(f"✅ GNN Evaluation - Acc: {acc:.2f}%, F1: {f1:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%")
        
        return metrics


def generate_datasets(output_dir: str = "training/datasets", small_test: bool = False):
    """
    Generate training datasets with proper class balance
    
    Args:
        output_dir: Directory to save datasets
        small_test: If True, generate smaller but still meaningful datasets
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SyntheticTrainingDataGenerator(seed=42)
    
    # Configure dataset sizes - need enough samples for 21-class classification
    if small_test:
        lstm_samples = 5000  # Increased for better learning
        gnn_samples = 2000   # Need more for 21-class classification
        logger.info("🧪 Generating TEST datasets (smaller but balanced)...")
    else:
        lstm_samples = 100000
        gnn_samples = 10000
        logger.info("🔄 Generating FULL training datasets...")
    
    # Generate LSTM dataset with sequence_length=30 to match ONNX model
    logger.info("📊 Generating LSTM dataset (sequence_length=30)...")
    lstm_config = DatasetConfig(
        num_samples=lstm_samples,
        positive_ratio=0.20,  # More defects for better learning
        feature_dim=20,
        sequence_length=30  # Match ONNX model input shape
    )
    lstm_dataset = generator.generate_lstm_dataset(lstm_config)
    generator.save_dataset(lstm_dataset, f"{output_dir}/lstm_training_data.npz", 'lstm')
    
    lstm_stats = generator.get_dataset_statistics(lstm_dataset, 'lstm')
    logger.info(f"LSTM dataset stats: {lstm_stats}")
    
    # Generate GNN dataset with 50% anomaly rate for balanced training
    logger.info("📊 Generating GNN dataset...")
    gnn_config = DatasetConfig(
        num_samples=gnn_samples,
        positive_ratio=0.50,  # 50% anomaly rate for balanced binary classification
        feature_dim=10  # Temporal statistics per sensor
    )
    gnn_dataset = generator.generate_gnn_dataset(gnn_config)
    generator.save_dataset(gnn_dataset, f"{output_dir}/gnn_training_data.npz", 'gnn')
    
    gnn_stats = generator.get_dataset_statistics(gnn_dataset, 'gnn')
    logger.info(f"GNN dataset stats: {gnn_stats}")
    
    logger.info("✅ All datasets generated successfully")
    
    return lstm_dataset, gnn_dataset


def evaluate_all_models(
    datasets_dir: str = "training/datasets",
    models_dir: str = "models",
    metrics_dir: str = "ml_metrics",
    small_test: bool = False,
    use_tensorboard: bool = True
):
    """Evaluate ONNX models (after training)"""
    # This function remains for backward compatibility
    # Use train_all_models() for full training pipeline
    pass


# ============= TRAINING FUNCTIONS =============
def train_lstm_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    writer: SummaryWriter = None
) -> Tuple[nn.Module, Dict]:
    """
    Train LSTM model for defect prediction
    
    Target: MAE < 0.05, R² > 0.90
    """
    logger.info(f"\n{'='*60}")
    logger.info("📊 TRAINING LSTM MODEL")
    logger.info(f"{'='*60}")
    
    # Create model
    model = create_enhanced_lstm(
        input_size=20,
        hidden_size=128,
        num_layers=2,
        output_size=6,  # 6 defect types
        use_quantile=False,  # Direct prediction
        dropout=0.2
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(sequences)
            # Model returns dict with 'prediction' key
            outputs = output['prediction']
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                output = model(sequences)
                outputs = output['prediction']
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        val_mae = np.mean(np.abs(all_preds - all_labels))
        ss_res = np.sum((all_labels - all_preds) ** 2)
        ss_tot = np.sum((all_labels - all_labels.mean()) ** 2)
        val_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('LSTM/train_loss', train_loss, epoch)
            writer.add_scalar('LSTM/val_loss', val_loss, epoch)
            writer.add_scalar('LSTM/val_mae', val_mae, epoch)
            writer.add_scalar('LSTM/val_r2', val_r2, epoch)
            writer.add_scalar('LSTM/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    logger.info(f"\n✅ LSTM Training Complete! Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"   Final MAE: {history['val_mae'][-1]:.4f}, R²: {history['val_r2'][-1]:.4f}")
    
    return model, history


def train_gnn_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int = 2,  # Binary: 0 = no anomaly, 1 = anomaly
    epochs: int = 50,
    lr: float = 0.002,
    writer: SummaryWriter = None
) -> Tuple[nn.Module, nn.Module, Dict]:
    """
    Train GNN model for anomaly detection (Binary Classification)
    
    Target: Accuracy > 85%, F1 > 80%
    """
    logger.info(f"\n{'='*60}")
    logger.info("📊 TRAINING GNN MODEL (Binary Classification)")
    logger.info(f"{'='*60}")
    
    # Create GNN model
    gnn_model = create_dynamic_gnn(
        num_sensors=20,
        input_dim=10,
        hidden_dim=64,
        output_dim=64,
        edge_dim=3
    ).to(DEVICE)
    
    # Binary classifier head
    classifier = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.BatchNorm1d(128),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes)
    ).to(DEVICE)
    
    # Class weights for balanced learning
    # Give much higher weight to anomaly class to force learning
    class_weights = torch.tensor([0.2, 3.0], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    params = list(gnn_model.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*10, epochs=epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    best_val_acc = 0.0
    best_gnn_state = None
    best_clf_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        # Training
        gnn_model.train()
        classifier.train()
        train_loss = 0.0
        
        for node_features, labels in train_loader:
            node_features, labels = node_features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Process each graph in batch
            batch_outputs = []
            num_nodes = node_features.size(1)
            
            # Create edge index (ring topology)
            edge_index = torch.zeros(2, num_nodes, dtype=torch.long, device=DEVICE)
            for i in range(num_nodes):
                edge_index[0, i] = i
                edge_index[1, i] = (i + 1) % num_nodes
            edge_attr = torch.ones(num_nodes, 3, device=DEVICE)
            
            for i in range(node_features.size(0)):
                gnn_out, _ = gnn_model(node_features[i], edge_index, edge_attr, return_attention=False)
                pooled = gnn_out.mean(dim=0, keepdim=True)
                batch_outputs.append(pooled)
            
            batch_outputs = torch.cat(batch_outputs, dim=0)
            logits = classifier(batch_outputs)
            
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()  # OneCycleLR steps per batch
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        gnn_model.eval()
        classifier.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for node_features, labels in val_loader:
                node_features, labels = node_features.to(DEVICE), labels.to(DEVICE)
                
                num_nodes = node_features.size(1)
                edge_index = torch.zeros(2, num_nodes, dtype=torch.long, device=DEVICE)
                for i in range(num_nodes):
                    edge_index[0, i] = i
                    edge_index[1, i] = (i + 1) % num_nodes
                edge_attr = torch.ones(num_nodes, 3, device=DEVICE)
                
                batch_outputs = []
                for i in range(node_features.size(0)):
                    gnn_out, _ = gnn_model(node_features[i], edge_index, edge_attr, return_attention=False)
                    pooled = gnn_out.mean(dim=0, keepdim=True)
                    batch_outputs.append(pooled)
                
                batch_outputs = torch.cat(batch_outputs, dim=0)
                logits = classifier(batch_outputs)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate binary classification metrics
        val_acc = accuracy_score(all_labels, all_preds) * 100
        val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0) * 100
        val_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0) * 100
        val_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0) * 100
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('GNN/train_loss', train_loss, epoch)
            writer.add_scalar('GNN/val_loss', val_loss, epoch)
            writer.add_scalar('GNN/val_accuracy', val_acc, epoch)
            writer.add_scalar('GNN/val_f1', val_f1, epoch)
            writer.add_scalar('GNN/val_precision', val_precision, epoch)
            writer.add_scalar('GNN/val_recall', val_recall, epoch)
            writer.add_scalar('GNN/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Note: OneCycleLR is already stepped per batch, no need to step here
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_gnn_state = gnn_model.state_dict().copy()
            best_clf_state = classifier.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.2f}%, "
                       f"Precision: {val_precision:.2f}%, Recall: {val_recall:.2f}%")
    
    # Load best model
    gnn_model.load_state_dict(best_gnn_state)
    classifier.load_state_dict(best_clf_state)
    
    logger.info(f"\n✅ GNN Training Complete! Best Val Acc: {best_val_acc:.2f}%")
    logger.info(f"   Final Acc: {history['val_acc'][-1]:.2f}%, F1: {history['val_f1'][-1]:.2f}%")
    
    return gnn_model, classifier, history


def export_to_onnx(lstm_model, gnn_model, gnn_classifier, models_dir: str):
    """Export trained models to ONNX format"""
    logger.info("\n📦 Exporting models to ONNX...")
    
    lstm_model.eval()
    gnn_model.eval()
    gnn_classifier.eval()
    
    # Export LSTM (with wrapper to return tensor instead of dict)
    lstm_path = f"{models_dir}/lstm_predictor/lstm_model.onnx"
    os.makedirs(os.path.dirname(lstm_path), exist_ok=True)
    
    lstm_wrapper = LSTMWrapper(lstm_model).to(DEVICE)
    lstm_wrapper.eval()
    
    dummy_input = torch.randn(1, 30, 20).to(DEVICE)
    torch.onnx.export(
        lstm_wrapper,
        dummy_input,
        lstm_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=12
    )
    logger.info(f"✅ LSTM exported to {lstm_path}")
    
    # Export GNN (with wrapper)
    gnn_path = f"{models_dir}/gnn_sensor_network/gnn_model.onnx"
    os.makedirs(os.path.dirname(gnn_path), exist_ok=True)
    
    gnn_wrapper = GNNWrapper(gnn_model, gnn_classifier).to(DEVICE)
    gnn_wrapper.eval()
    
    dummy_gnn_input = torch.randn(20, 10).to(DEVICE)
    torch.onnx.export(
        gnn_wrapper,
        dummy_gnn_input,
        gnn_path,
        input_names=['node_features'],
        output_names=['output'],
        opset_version=16
    )
    logger.info(f"✅ GNN exported to {gnn_path}")


def train_all_models(
    datasets_dir: str = "training/datasets",
    models_dir: str = "models",
    metrics_dir: str = "ml_metrics",
    epochs: int = 30,
    batch_size: int = 64,
    small_test: bool = False
):
    """
    Main training pipeline: generates data, trains models, exports to ONNX
    
    Args:
        datasets_dir: Directory for datasets
        models_dir: Directory for ONNX models
        metrics_dir: Directory for metrics/TensorBoard
        epochs: Number of training epochs
        batch_size: Batch size for training
        small_test: Use small datasets for quick testing
    """
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=f"{metrics_dir}/tensorboard/training")
    logger.info(f"\n📊 TensorBoard: tensorboard --logdir={os.path.abspath(metrics_dir)}/tensorboard")
    
    # Generate or load datasets
    lstm_path = f"{datasets_dir}/lstm_training_data.npz"
    gnn_path = f"{datasets_dir}/gnn_training_data.npz"
    
    if not os.path.exists(lstm_path) or not os.path.exists(gnn_path):
        logger.info("📦 Generating datasets...")
        lstm_dataset, gnn_dataset = generate_datasets(datasets_dir, small_test=small_test)
    else:
        logger.info("📦 Loading existing datasets...")
        lstm_data = np.load(lstm_path)
        lstm_dataset = (lstm_data['sequences'], lstm_data['labels'])
        
        gnn_data = np.load(gnn_path, allow_pickle=True)
        node_features_array = gnn_data['node_features']
        node_features_list = [node_features_array[i] for i in range(len(node_features_array))]
        gnn_dataset = (node_features_list, gnn_data['labels'])
    
    # ===== TRAIN LSTM =====
    sequences, labels = lstm_dataset
    split_idx = int(len(sequences) * 0.8)
    
    train_lstm_ds = LSTMDataset(sequences[:split_idx], labels[:split_idx])
    val_lstm_ds = LSTMDataset(sequences[split_idx:], labels[split_idx:])
    
    train_lstm_loader = DataLoader(train_lstm_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_lstm_loader = DataLoader(val_lstm_ds, batch_size=batch_size, num_workers=0)
    
    lstm_model, lstm_history = train_lstm_model(
        train_lstm_loader, val_lstm_loader,
        epochs=epochs, writer=writer
    )
    
    # ===== TRAIN GNN =====
    node_features_list, gnn_labels = gnn_dataset[0], gnn_dataset[1] if len(gnn_dataset) == 2 else gnn_dataset[2]
    split_idx = int(len(node_features_list) * 0.8)
    
    train_gnn_ds = GNNDataset(node_features_list[:split_idx], gnn_labels[:split_idx])
    val_gnn_ds = GNNDataset(node_features_list[split_idx:], gnn_labels[split_idx:])
    
    train_gnn_loader = DataLoader(train_gnn_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_gnn_loader = DataLoader(val_gnn_ds, batch_size=batch_size, num_workers=0)
    
    gnn_model, gnn_classifier, gnn_history = train_gnn_model(
        train_gnn_loader, val_gnn_loader,
        epochs=epochs, writer=writer
    )
    
    # Export to ONNX
    export_to_onnx(lstm_model, gnn_model, gnn_classifier, models_dir)
    
    # Close TensorBoard writer
    writer.close()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"LSTM - Final MAE: {lstm_history['val_mae'][-1]:.4f}, R²: {lstm_history['val_r2'][-1]:.4f}")
    logger.info(f"GNN  - Final Acc: {gnn_history['val_acc'][-1]:.2f}%, F1: {gnn_history['val_f1'][-1]:.2f}%")
    logger.info("="*60)
    logger.info(f"📊 View TensorBoard: tensorboard --logdir={os.path.abspath(metrics_dir)}/tensorboard")
    logger.info("="*60)
    
    return {
        'lstm': lstm_history,
        'gnn': gnn_history
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Glass Production ML Models")
    parser.add_argument("--train", action="store_true", help="Train models (default action)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--small-test", action="store_true", help="Use small datasets for quick testing")
    parser.add_argument("--datasets-dir", type=str, default="training/datasets", help="Datasets directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--metrics-dir", type=str, default="ml_metrics", help="Metrics output directory")
    
    args = parser.parse_args()
    
    logger.info("🚀 Glass Production ML Training Pipeline")
    logger.info(f"Configuration:")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Small Test Mode: {args.small_test}")
    logger.info(f"  - Device: {DEVICE}")
    
    train_all_models(
        datasets_dir=args.datasets_dir,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        small_test=args.small_test
    )
