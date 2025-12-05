"""
Synthetic Training Data Generator for ML Models
Generates labeled datasets with known parameter-defect correlations for model training.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    num_samples: int
    positive_ratio: float  # Ratio of samples with defects
    feature_dim: int
    sequence_length: Optional[int] = None  # For time series data


class SyntheticTrainingDataGenerator:
    """
    Generate synthetic training datasets for ML models
    with realistic parameter-defect correlations
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize synthetic data generator
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
        
        # Define parameter ranges based on task.md specifications
        self.parameter_ranges = {
            'furnace_temperature': (1200, 1700, 1500, 50),  # min, max, mean, std
            'furnace_pressure': (0, 50, 15, 5),
            'melt_level': (0, 5000, 2500, 300),
            'belt_speed': (50, 200, 150, 20),
            'mold_temperature': (200, 600, 320, 40),
            'forming_pressure': (0, 120, 50, 15),
            'cooling_rate': (1, 10, 3.5, 1.5),
            'annealing_temp': (400, 700, 580, 40),
            'gas_o2': (0, 25, 5, 2),
            'gas_co2': (0, 25, 8, 3),
            'humidity': (20, 80, 50, 10),
            'viscosity': (100, 10000, 1200, 500),
            'conveyor_speed': (50, 200, 145, 20),
            'fuel_flow': (0, 100, 75, 10),
            'air_flow': (0, 100, 80, 10),
            'burner_power': (0, 100, 85, 8),
            'feed_rate': (0, 5000, 2500, 400),
            'product_thickness': (1, 20, 5, 2),
            'surface_quality': (0, 1, 0.85, 0.1),
            'internal_stress': (0, 100, 20, 10)
        }
        
        # Define defect types
        self.defect_types = ['crack', 'bubble', 'chip', 'cloudiness', 'deformation', 'stain']
        
        # Define parameter-defect causation rules (from design doc)
        self.causation_rules = {
            'crack': [
                ('furnace_temperature', '>', 1600, 0.85),
                ('cooling_rate', '>', 7, 0.90),
                ('internal_stress', '>', 60, 0.80)
            ],
            'cloudiness': [
                ('furnace_temperature', '<', 1450, 0.78),
                ('gas_o2', '<', 3, 0.70)
            ],
            'deformation': [
                ('belt_speed', '>', 180, 0.82),
                ('mold_temperature', '<', 280, 0.75),
                ('viscosity', '<', 500, 0.70)
            ],
            'bubble': [
                ('forming_pressure', 'var', 15, 0.88),  # variance > threshold
                ('furnace_temperature', '>', 1580, 0.75)
            ],
            'chip': [
                ('belt_speed', '>', 170, 0.70),
                ('surface_quality', '<', 0.7, 0.75)
            ],
            'stain': [
                ('humidity', '>', 70, 0.65),
                ('surface_quality', '<', 0.8, 0.70)
            ]
        }
        
        logger.info("✅ Synthetic Training Data Generator initialized")
    
    def generate_lstm_dataset(self, config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time series dataset for LSTM model
        
        Args:
            config: Dataset configuration
            
        Returns:
            Tuple of (sequences, labels) where:
                sequences: shape (num_samples, sequence_length, feature_dim)
                labels: shape (num_samples, len(defect_types) + 3) 
                       [defect probabilities (6), 1h_defect, 4h_defect, 24h_defect]
        """
        logger.info(f"Generating LSTM dataset: {config.num_samples} sequences, "
                   f"length {config.sequence_length}, features {config.feature_dim}")
        
        sequences = []
        labels = []
        
        for i in range(config.num_samples):
            # Generate time series sequence
            sequence = self._generate_time_series_sequence(
                config.sequence_length, 
                config.feature_dim
            )
            
            # Calculate defect probabilities based on parameter values
            last_timestep = sequence[-1]
            defect_probs = self._calculate_defect_probabilities(last_timestep)
            
            # Determine if defect occurs in future horizons
            defect_1h = 1 if np.random.random() < defect_probs.max() * 0.8 else 0
            defect_4h = 1 if np.random.random() < defect_probs.max() * 0.6 else 0
            defect_24h = 1 if np.random.random() < defect_probs.max() * 0.4 else 0
            
            # Combine into label
            label = np.concatenate([defect_probs, [defect_1h, defect_4h, defect_24h]])
            
            sequences.append(sequence)
            labels.append(label)
        
        sequences = np.array(sequences, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        
        logger.info(f"✅ Generated LSTM dataset: sequences {sequences.shape}, labels {labels.shape}")
        
        return sequences, labels
    
    def generate_gnn_dataset(self, config: DatasetConfig) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Generate graph dataset for GNN model with LEARNABLE patterns
        
        Args:
            config: Dataset configuration
            
        Returns:
            Tuple of (node_features_list, edge_indices_list, labels_array)
        """
        logger.info(f"Generating GNN dataset: {config.num_samples} graphs")
        
        num_sensors = 20
        node_features_list = []
        edge_indices_list = []
        labels_list = []
        
        for i in range(config.num_samples):
            # Generate base node features (sensor statistics)
            node_features = np.random.randn(num_sensors, config.feature_dim).astype(np.float32)
            
            # Normalize to reasonable range
            node_features = node_features * 0.3
            
            # Determine if this sample has an anomaly
            anomaly_prob = np.random.random()
            if anomaly_prob < config.positive_ratio:  # Use config positive_ratio
                # Pick which sensor has anomaly
                anomaly_sensor = np.random.randint(0, num_sensors)
                label = anomaly_sensor
                
                # CREATE CLEAR ANOMALY SIGNAL in the features!
                # The anomalous sensor will have distinct feature patterns:
                # 1. Higher magnitude values
                node_features[anomaly_sensor] *= 3.0
                node_features[anomaly_sensor] += 2.0  # Shift mean
                
                # 2. First few features are strongly positive (anomaly signature)
                node_features[anomaly_sensor, 0:3] = np.abs(node_features[anomaly_sensor, 0:3]) + 1.5
                
                # 3. Add correlation pattern - neighboring sensors also affected
                if anomaly_sensor > 0:
                    node_features[anomaly_sensor - 1] *= 1.5
                    node_features[anomaly_sensor - 1, 0] += 0.5
                if anomaly_sensor < num_sensors - 1:
                    node_features[anomaly_sensor + 1] *= 1.5
                    node_features[anomaly_sensor + 1, 0] += 0.5
            else:
                label = -1  # No anomaly
                # Normal sensors have balanced features around 0
            
            # Generate edge index (sensor correlations)
            edge_index = self._generate_sensor_graph(num_sensors)
            
            node_features_list.append(node_features)
            edge_indices_list.append(edge_index)
            labels_list.append(label)
        
        # Convert labels to numpy array
        labels = np.array(labels_list, dtype=np.int64)
        
        logger.info(f"✅ Generated GNN dataset: {len(node_features_list)} graphs")
        
        return node_features_list, edge_indices_list, labels
    
    def save_dataset(self, dataset: Tuple, filepath: str, dataset_type: str):
        """
        Save generated dataset to disk
        
        Args:
            dataset: Generated dataset tuple
            filepath: Path to save dataset
            dataset_type: Type of dataset ('lstm', 'gnn')
        """
        try:
            if dataset_type == 'lstm':
                sequences, labels = dataset
                np.savez_compressed(
                    filepath,
                    sequences=sequences,
                    labels=labels,
                    metadata={
                        'dataset_type': 'lstm',
                        'num_samples': len(sequences),
                        'sequence_length': sequences.shape[1],
                        'feature_dim': sequences.shape[2],
                        'generated_at': datetime.utcnow().isoformat()
                    }
                )
            elif dataset_type == 'gnn':
                node_features_list, edge_indices_list, labels = dataset
                
                # For GNN, we need to save lists of arrays differently
                # Save node features as a single array (they all have the same shape)
                node_features_array = np.stack(node_features_list)
                
                # For edge indices, we need to save them separately or pad them
                # Let's pad them to the maximum number of edges
                max_edges = max(edge_index.shape[1] if edge_index.size > 0 else 0 for edge_index in edge_indices_list)
                
                if max_edges > 0:
                    # Pad all edge indices to the same shape
                    padded_edge_indices = []
                    for edge_index in edge_indices_list:
                        if edge_index.size == 0:
                            # Empty edge index, create a zero-filled array
                            padded_edge_index = np.zeros((2, max_edges), dtype=np.int64)
                        elif edge_index.shape[1] < max_edges:
                            # Pad with -1 (will be ignored in GNN processing)
                            padding = np.full((2, max_edges - edge_index.shape[1]), -1, dtype=np.int64)
                            padded_edge_index = np.concatenate([edge_index, padding], axis=1)
                        else:
                            # Already the right size or larger (shouldn't happen)
                            padded_edge_index = edge_index[:, :max_edges]
                        padded_edge_indices.append(padded_edge_index)
                    
                    edge_indices_array = np.stack(padded_edge_indices)
                else:
                    # All graphs are empty
                    edge_indices_array = np.empty((len(edge_indices_list), 2, 0), dtype=np.int64)
                
                np.savez_compressed(
                    filepath,
                    node_features=node_features_array,
                    edge_indices=edge_indices_array,
                    labels=labels,
                    metadata={
                        'dataset_type': 'gnn',
                        'num_samples': len(node_features_list),
                        'num_sensors': node_features_list[0].shape[0] if node_features_list else 0,
                        'feature_dim': node_features_list[0].shape[1] if node_features_list else 0,
                        'max_edges': max_edges,
                        'generated_at': datetime.utcnow().isoformat()
                    }
                )
            
            logger.info(f"✅ Dataset saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error saving dataset: {e}")
    
    def _generate_time_series_sequence(self, length: int, feature_dim: int) -> np.ndarray:
        """Generate a single time series sequence with realistic patterns"""
        sequence = np.zeros((length, feature_dim), dtype=np.float32)
        
        # Get parameter names (first feature_dim parameters)
        param_names = list(self.parameter_ranges.keys())[:feature_dim]
        
        for t in range(length):
            for i, param_name in enumerate(param_names):
                min_val, max_val, mean, std = self.parameter_ranges[param_name]
                
                if t == 0:
                    # Initial value
                    value = np.random.normal(mean, std)
                else:
                    # Add temporal correlation (drift)
                    drift = np.random.normal(0, std * 0.1)
                    value = sequence[t-1, i] + drift
                
                # Clip to valid range
                value = np.clip(value, min_val, max_val)
                sequence[t, i] = value
        
        return sequence
    
    def _calculate_defect_probabilities(self, parameters: np.ndarray) -> np.ndarray:
        """Calculate defect probabilities based on parameter values"""
        param_names = list(self.parameter_ranges.keys())[:len(parameters)]
        param_dict = {name: parameters[i] for i, name in enumerate(param_names)}
        
        defect_probs = np.zeros(len(self.defect_types), dtype=np.float32)
        
        for defect_idx, defect_type in enumerate(self.defect_types):
            rules = self.causation_rules.get(defect_type, [])
            
            max_prob = 0.0
            for param_name, operator, threshold, confidence in rules:
                if param_name not in param_dict:
                    continue
                
                value = param_dict[param_name]
                
                # Check if rule is satisfied
                if operator == '>':
                    if value > threshold:
                        deviation = (value - threshold) / threshold
                        prob = confidence * min(1.0, deviation * 0.5)
                        max_prob = max(max_prob, prob)
                elif operator == '<':
                    if value < threshold:
                        deviation = (threshold - value) / threshold
                        prob = confidence * min(1.0, deviation * 0.5)
                        max_prob = max(max_prob, prob)
                elif operator == 'var':
                    # For variance check (not implemented in single timestep)
                    prob = confidence * 0.3
                    max_prob = max(max_prob, prob)
            
            # Add base probability
            defect_probs[defect_idx] = max_prob + np.random.uniform(0, 0.05)
        
        # Normalize to [0, 1]
        defect_probs = np.clip(defect_probs, 0.0, 1.0)
        
        return defect_probs
    
    def _generate_sensor_graph(self, num_sensors: int) -> np.ndarray:
        """Generate sensor correlation graph"""
        edges = []
        
        # Create edges based on sensor proximity and functional relationships
        for i in range(num_sensors):
            # Connect to nearby sensors
            for j in range(max(0, i-3), min(num_sensors, i+4)):
                if i != j:
                    # Add edge with some probability based on distance
                    distance = abs(i - j)
                    prob = 0.9 if distance == 1 else (0.6 if distance == 2 else 0.3)
                    
                    if np.random.random() < prob:
                        edges.append([i, j])
        
        if not edges:
            # Ensure at least some edges - connect each sensor to its immediate neighbor
            for i in range(num_sensors - 1):
                edges.append([i, i+1])
        
        # Convert to numpy array and transpose to get shape (2, num_edges)
        if edges:
            edge_array = np.array(edges, dtype=np.int64)
            return edge_array.T  # Shape: (2, num_edges)
        else:
            # Return empty edge array with correct shape
            return np.empty((2, 0), dtype=np.int64)
    
    def get_dataset_statistics(self, dataset: Tuple, dataset_type: str) -> Dict:
        """Get statistics about generated dataset"""
        stats = {}
        
        if dataset_type == 'lstm':
            sequences, labels = dataset
            stats = {
                'num_samples': len(sequences),
                'sequence_length': sequences.shape[1],
                'feature_dim': sequences.shape[2],
                'positive_samples': int(np.sum(labels[:, -3:].max(axis=1))),
                'positive_ratio': float(np.mean(labels[:, -3:].max(axis=1))),
                'avg_defect_prob': float(np.mean(labels[:, :6])),
                'dataset_type': 'lstm'
            }
        elif dataset_type == 'gnn':
            node_features_list, edge_indices_list, labels = dataset
            stats = {
                'num_samples': len(node_features_list),
                'num_sensors': node_features_list[0].shape[0] if node_features_list else 0,
                'feature_dim': node_features_list[0].shape[1] if node_features_list else 0,
                'num_anomalies': int(np.sum(labels >= 0)),
                'anomaly_ratio': float(np.mean(labels >= 0)),
                'dataset_type': 'gnn'
            }
        
        return stats


def create_training_datasets(output_dir: str = "training/datasets"):
    """
    Create all training datasets for ML models
    
    Args:
        output_dir: Directory to save datasets
    """
    import os
    os.makedirs(output_dir, exist_ok=True)    
    generator = SyntheticTrainingDataGenerator(seed=42)
    
    # Generate LSTM dataset (100,000 sequences as per design doc)
    logger.info("🔄 Generating LSTM training dataset...")
    lstm_config = DatasetConfig(
        num_samples=100000,
        positive_ratio=0.15,
        feature_dim=20,
        sequence_length=120  # 2 hours of minute-level data
    )
    lstm_dataset = generator.generate_lstm_dataset(lstm_config)
    generator.save_dataset(lstm_dataset, f"{output_dir}/lstm_training_data.npz", 'lstm')
    
    lstm_stats = generator.get_dataset_statistics(lstm_dataset, 'lstm')
    logger.info(f"LSTM dataset stats: {lstm_stats}")
    
    # Generate GNN dataset (10,000 graph snapshots)
    logger.info("🔄 Generating GNN training dataset...")
    gnn_config = DatasetConfig(
        num_samples=10000,
        positive_ratio=0.15,
        feature_dim=10  # Temporal statistics per sensor
    )
    gnn_dataset = generator.generate_gnn_dataset(gnn_config)
    generator.save_dataset(gnn_dataset, f"{output_dir}/gnn_training_data.npz", 'gnn')
    
    gnn_stats = generator.get_dataset_statistics(gnn_dataset, 'gnn')
    logger.info(f"GNN dataset stats: {gnn_stats}")
    
    logger.info("✅ All training datasets generated successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_training_datasets()
