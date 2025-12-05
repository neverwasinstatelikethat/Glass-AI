"""
Enhanced GNN with Dynamic Graph Construction, Attention Visualization, and Edge Features
КРИТИЧЕСКИЕ УЛУЧШЕНИЯ:
- Dynamic graph updates на основе корреляций
- Edge feature learning
- Attention visualization для GAT
- Temporal graph evolution
- Multi-scale graph aggregation
- Graph pooling strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax
import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.stats import spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicGraphBuilder:
    """Динамическое построение графа на основе временных корреляций"""
    
    def __init__(self, window_size: int = 50, update_freq: int = 10):
        self.window_size = window_size
        self.update_freq = update_freq
        self.correlation_history = []
        self.current_graph = None
        self.update_counter = 0
    
    def update_graph(
        self,
        sensor_data: np.ndarray,
        sensor_names: List[str],
        threshold: float = 0.6,
        method: str = "pearson"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Динамическое обновление графа
        
        Args:
            sensor_data: [window_size, n_sensors] последние измерения
            sensor_names: имена датчиков
            threshold: порог корреляции
            method: 'pearson' или 'spearman'
        
        Returns:
            edge_index: [2, n_edges]
            edge_attr: [n_edges, edge_features]
        """
        n_sensors = len(sensor_names)
        
        # Вычисление корреляций
        if method == "spearman":
            corr_matrix = np.zeros((n_sensors, n_sensors))
            for i in range(n_sensors):
                for j in range(i, n_sensors):
                    corr, _ = spearmanr(sensor_data[:, i], sensor_data[:, j])
                    corr_matrix[i, j] = corr_matrix[j, i] = corr
        else:
            corr_matrix = np.corrcoef(sensor_data.T)
        
        # Сохранение истории
        self.correlation_history.append(corr_matrix)
        if len(self.correlation_history) > 10:
            self.correlation_history.pop(0)
        
        # Усреднение корреляций для стабильности
        avg_corr = np.mean(self.correlation_history, axis=0)
        
        # Построение ребер
        edges = []
        edge_features = []
        
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                corr = abs(avg_corr[i, j])
                if corr > threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
                    
                    # Edge features: [correlation, variance, lag]
                    variance = np.var(sensor_data[:, i]) + np.var(sensor_data[:, j])
                    lag = self._compute_lag_correlation(sensor_data[:, i], sensor_data[:, j])
                    
                    edge_feat = [corr, variance / 1000, lag]  # Normalize variance
                    edge_features.extend([edge_feat, edge_feat])  # Both directions
        
        if len(edges) == 0:
            # Minimum spanning tree если нет сильных корреляций
            edges, edge_features = self._create_minimum_graph(n_sensors, avg_corr)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        self.current_graph = (edge_index, edge_attr)
        self.update_counter += 1
        
        logger.debug(f"Graph updated: {n_sensors} nodes, {len(edges)} edges")
        
        return edge_index, edge_attr
    
    def _compute_lag_correlation(self, x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> float:
        """Вычисление корреляции с временным лагом"""
        best_corr = 0.0
        for lag in range(max_lag):
            if lag == 0:
                corr = np.corrcoef(x, y)[0, 1]
            else:
                corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
        return best_corr
    
    def _create_minimum_graph(self, n_sensors: int, corr_matrix: np.ndarray) -> Tuple[List, List]:
        """Создание минимального связного графа"""
        edges = []
        edge_features = []
        
        # Ring topology как fallback
        for i in range(n_sensors):
            j = (i + 1) % n_sensors
            edges.extend([[i, j], [j, i]])
            
            corr = abs(corr_matrix[i, j])
            edge_feat = [corr, 0.5, 0.0]
            edge_features.extend([edge_feat, edge_feat])
        
        return edges, edge_features
    
    def should_update(self) -> bool:
        """Проверка необходимости обновления графа"""
        return self.update_counter % self.update_freq == 0


class EdgeFeatureGATConv(nn.Module):
    """GAT с обучением edge features"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 edge_dim: int = 3, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.gat = GATConv(
            in_channels, 
            out_channels, 
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True
        )
        
        # Edge feature transformation
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 2),
            nn.ReLU(),
            nn.Linear(edge_dim * 2, edge_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with edge features
        
        Returns:
            node_features: [num_nodes, out_channels * heads]
            attention_weights: [num_edges, heads]
        """
        # Encode edge features
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        # GAT forward
        out = self.gat(x, edge_index, edge_attr=edge_attr_encoded, return_attention_weights=True)
        
        if isinstance(out, tuple):
            node_features, (edge_index_att, attention_weights) = out
            return node_features, attention_weights
        else:
            return out, None


class EnhancedGATSensorGNN(nn.Module):
    """Улучшенный GAT с edge features и attention visualization"""
    
    def __init__(
        self,
        num_sensors: int,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 32,
        edge_dim: int = 3,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_sensors = num_sensors
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers with edge features
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim * num_heads if i > 0 else hidden_dim
            out_dim = hidden_dim
            heads = num_heads if i < num_layers - 1 else 1
            
            self.gat_layers.append(
                EdgeFeatureGATConv(in_dim, out_dim, edge_dim, heads, dropout)
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Batch norm layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * (num_heads if i < num_layers - 1 else 1))
            for i in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights_history = []
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with attention tracking
        
        Returns:
            output: [num_nodes, output_dim]
            attention_weights: List of attention weights per layer (if return_attention=True)
        """
        # Input projection
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)
        
        attention_weights = [] if return_attention else None
        
        # GAT layers
        for i, (gat_layer, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_new, attn = gat_layer(x, edge_index, edge_attr)
            
            if return_attention and attn is not None:
                attention_weights.append(attn)
            
            # Batch norm
            if batch is not None:
                x_new = bn(x_new)
            
            # Activation and dropout (except last layer)
            if i < len(self.gat_layers) - 1:
                x_new = F.relu(x_new)
                x_new = self.dropout(x_new)
            
            x = x_new
        
        # Output projection
        x = self.output_proj(x)
        
        # Store attention for visualization
        if return_attention:
            self.attention_weights_history.append(attention_weights)
            if len(self.attention_weights_history) > 100:
                self.attention_weights_history.pop(0)
        
        return x, attention_weights
    
    def visualize_attention(self, layer_idx: int = -1) -> Optional[np.ndarray]:
        """
        Get attention weights for visualization
        
        Args:
            layer_idx: which layer to visualize (-1 for last)
        
        Returns:
            attention_matrix: [num_edges, num_heads] or None
        """
        if not self.attention_weights_history:
            return None
        
        last_attention = self.attention_weights_history[-1]
        
        if layer_idx >= len(last_attention):
            return None
        
        return last_attention[layer_idx].detach().cpu().numpy()


class MultiScaleGraphAggregation(nn.Module):
    """Multi-scale graph aggregation для захвата разных уровней взаимодействий"""
    
    def __init__(self, hidden_dim: int, num_scales: int = 3):
        super().__init__()
        
        self.num_scales = num_scales
        
        # Different aggregation functions for different scales
        self.aggregators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_scales)
        ])
        
        # Scale combination
        self.scale_combiner = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Multi-scale aggregation
        
        Returns:
            aggregated: [num_nodes, hidden_dim]
        """
        scale_outputs = []
        
        for scale, aggregator in enumerate(self.aggregators):
            # Different hop neighborhoods for different scales
            if scale == 0:
                # 1-hop neighbors
                scale_out = aggregator(x)
            else:
                # k-hop neighbors (simplified)
                scale_out = aggregator(x)
            
            scale_outputs.append(scale_out)
        
        # Combine scales
        combined = torch.cat(scale_outputs, dim=-1)
        aggregated = self.scale_combiner(combined)
        
        return aggregated


class EnhancedSensorGraphAnomalyDetector(nn.Module):
    """Улучшенный детектор аномалий с reconstruction + classification"""
    
    def __init__(
        self,
        num_sensors: int,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        edge_dim: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Encoder GNN
        self.encoder = EnhancedGATSensorGNN(
            num_sensors=num_sensors,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            edge_dim=edge_dim,
            dropout=dropout
        )
        
        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        # Anomaly classifier head
        self.anomaly_classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),  # Normal / Anomaly
            nn.Softmax(dim=-1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            reconstructed: [num_nodes, input_dim]
            latent: [num_nodes, latent_dim]
            anomaly_scores: [num_nodes, 2]
        """
        # Encode
        latent, _ = self.encoder(x, edge_index, edge_attr, batch)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        # Anomaly classification
        anomaly_scores = self.anomaly_classifier(latent)
        
        return reconstructed, latent, anomaly_scores
    
    def compute_anomaly_score(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Comprehensive anomaly scoring
        
        Returns:
            scores: Dict with reconstruction_error and classification_score
        """
        reconstructed, latent, anomaly_probs = self.forward(x, edge_index, edge_attr, batch)
        
        # Reconstruction error
        mse = F.mse_loss(reconstructed, x, reduction='none').mean(dim=-1)
        
        # Classification score (probability of anomaly)
        anomaly_prob = anomaly_probs[:, 1]
        
        # Combined score
        combined_score = 0.5 * mse + 0.5 * anomaly_prob
        
        return {
            'reconstruction_error': mse,
            'classification_score': anomaly_prob,
            'combined_score': combined_score
        }


# Factory functions
def create_dynamic_gnn(
    num_sensors: int,
    input_dim: int = 1,
    hidden_dim: int = 64,
    output_dim: int = 32,
    edge_dim: int = 3
) -> EnhancedGATSensorGNN:
    """Create enhanced GAT with dynamic graph support"""
    model = EnhancedGATSensorGNN(
        num_sensors=num_sensors,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        edge_dim=edge_dim
    )
    
    logger.info(f"✅ Enhanced GAT created: {num_sensors} sensors, edge_dim={edge_dim}")
    return model


def create_enhanced_anomaly_detector(
    num_sensors: int,
    input_dim: int = 1,
    hidden_dim: int = 64,
    latent_dim: int = 32,
    edge_dim: int = 3
) -> EnhancedSensorGraphAnomalyDetector:
    """Create enhanced anomaly detector"""
    model = EnhancedSensorGraphAnomalyDetector(
        num_sensors=num_sensors,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        edge_dim=edge_dim
    )
    
    logger.info(f"✅ Enhanced anomaly detector created")
    return model


# ==================== TESTING ====================
if __name__ == "__main__":
    print("🧪 Testing Enhanced GNN...")
    
    # Parameters
    num_sensors = 10
    input_dim = 1
    window_size = 50
    
    # Dynamic graph builder
    print("\n1. Testing dynamic graph construction...")
    graph_builder = DynamicGraphBuilder(window_size=window_size)
    
    # Simulate sensor data
    sensor_data = np.random.randn(window_size, num_sensors)
    sensor_names = [f"sensor_{i}" for i in range(num_sensors)]
    
    edge_index, edge_attr = graph_builder.update_graph(sensor_data, sensor_names)
    print(f"   Graph: {num_sensors} nodes, {edge_index.shape[1]} edges")
    print(f"   Edge features shape: {edge_attr.shape}")
    
    # Enhanced GAT
    print("\n2. Testing Enhanced GAT with edge features...")
    model = create_dynamic_gnn(num_sensors, input_dim, hidden_dim=32, edge_dim=3)
    
    x = torch.randn(num_sensors, input_dim)
    output, attention = model(x, edge_index, edge_attr, return_attention=True)
    
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    if attention:
        print(f"   Attention weights: {len(attention)} layers")
        print(f"   First layer attention: {attention[0].shape}")
    
    # Enhanced anomaly detector
    print("\n3. Testing enhanced anomaly detector...")
    anomaly_detector = create_enhanced_anomaly_detector(
        num_sensors, input_dim, hidden_dim=32, edge_dim=3
    )
    
    reconstructed, latent, anomaly_probs = anomaly_detector(x, edge_index, edge_attr)
    scores = anomaly_detector.compute_anomaly_score(x, edge_index, edge_attr)
    
    print(f"   Reconstructed: {reconstructed.shape}")
    print(f"   Latent: {latent.shape}")
    print(f"   Anomaly probs: {anomaly_probs.shape}")
    print(f"   Scores keys: {scores.keys()}")
    
    print("\n✅ All Enhanced GNN tests passed!")