"""
Dynamic Graph Builder for Sensor Networks
Constructs and updates sensor connectivity graphs based on temporal correlations and physical proximity
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialProximityGraph:
    """
    Builds graph edges based on physical sensor locations
    """
    
    def __init__(self, max_distance: float = 10.0):
        """
        Args:
            max_distance: Maximum distance for connecting sensors
        """
        self.max_distance = max_distance
    
    def build_graph(
        self, 
        sensor_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph based on spatial proximity
        
        Args:
            sensor_positions: [n_sensors, 3] array of 3D positions
            
        Returns:
            edge_index: [2, n_edges] edge connectivity
            edge_attr: [n_edges, 1] edge distances
        """
        n_sensors = sensor_positions.shape[0]
        
        # Compute pairwise distances
        distances = squareform(pdist(sensor_positions, metric='euclidean'))
        
        # Create edges for sensors within max_distance
        edges = []
        edge_attrs = []
        
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                if distances[i, j] <= self.max_distance:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
                    edge_attrs.append([distances[i, j]])
                    edge_attrs.append([distances[i, j]])
        
        if len(edges) == 0:
            # Connect each sensor to its nearest neighbor if no connections
            for i in range(n_sensors):
                nearest_j = np.argmin(distances[i] + np.eye(n_sensors)[i] * 1e10)
                edges.append([i, nearest_j])
                edges.append([nearest_j, i])
                edge_attrs.append([distances[i, nearest_j]])
                edge_attrs.append([distances[i, nearest_j]])
        
        edge_index = np.array(edges).T.astype(np.int64)
        edge_attr = np.array(edge_attrs, dtype=np.float32)
        
        return edge_index, edge_attr


class TemporalCorrelationGraph:
    """
    Builds graph edges based on temporal correlations between sensors
    """
    
    def __init__(
        self, 
        correlation_window: int = 100,
        correlation_threshold: float = 0.7,
        correlation_method: str = "pearson"
    ):
        """
        Args:
            correlation_window: Window size for computing correlations
            correlation_threshold: Minimum correlation for edge creation
            correlation_method: "pearson" or "spearman"
        """
        self.correlation_window = correlation_window
        self.correlation_threshold = correlation_threshold
        self.correlation_method = correlation_method
        self.history = []
    
    def update_history(self, sensor_data: np.ndarray):
        """
        Update historical data buffer
        
        Args:
            sensor_data: [n_sensors] latest sensor readings
        """
        self.history.append(sensor_data)
        if len(self.history) > self.correlation_window:
            self.history.pop(0)
    
    def build_graph(
        self, 
        sensor_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build graph based on temporal correlations
        
        Args:
            sensor_data: [n_sensors] latest sensor readings (optional, uses history if not provided)
            
        Returns:
            edge_index: [2, n_edges] edge connectivity
            edge_attr: [n_edges, 2] edge correlations and lags
        """
        # Update history if new data provided
        if sensor_data is not None:
            self.update_history(sensor_data)
        
        # Need sufficient history
        if len(self.history) < 10:
            # Return empty graph with minimal connections
            return self._build_minimal_graph()
        
        # Convert history to array
        history_array = np.array(self.history)  # [time_steps, n_sensors]
        n_sensors = history_array.shape[1]
        
        # Compute correlations
        if self.correlation_method == "spearman":
            correlations = np.zeros((n_sensors, n_sensors))
            for i in range(n_sensors):
                for j in range(i, n_sensors):
                    if len(history_array[:, i]) > 1 and len(history_array[:, j]) > 1:
                        corr, _ = spearmanr(history_array[:, i], history_array[:, j])
                    else:
                        corr = 0.0
                    correlations[i, j] = correlations[j, i] = corr
        else:
            correlations = np.corrcoef(history_array.T)
        
        # Handle NaN values
        correlations = np.nan_to_num(correlations, nan=0.0)
        
        # Create edges for highly correlated sensors
        edges = []
        edge_attrs = []
        
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                corr = abs(correlations[i, j])
                if corr >= self.correlation_threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
                    # Compute temporal lag
                    lag = self._compute_temporal_lag(history_array[:, i], history_array[:, j])
                    edge_attrs.append([corr, lag])
                    edge_attrs.append([corr, -lag])  # Opposite lag for reverse direction
        
        if len(edges) == 0:
            # Return minimal graph if no strong correlations
            return self._build_minimal_graph(n_sensors)
        
        edge_index = np.array(edges).T.astype(np.int64)
        edge_attr = np.array(edge_attrs, dtype=np.float32)
        
        return edge_index, edge_attr
    
    def _compute_temporal_lag(self, x: np.ndarray, y: np.ndarray, max_lag: int = 5) -> float:
        """
        Compute optimal temporal lag between two signals
        
        Args:
            x, y: Time series data
            max_lag: Maximum lag to consider
            
        Returns:
            Optimal lag value
        """
        best_corr = 0.0
        best_lag = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                x_aligned = x[-lag:]
                y_aligned = y[:lag]
            elif lag > 0:
                x_aligned = x[:-lag]
                y_aligned = y[lag:]
            else:
                x_aligned = x
                y_aligned = y
            
            if len(x_aligned) > 1 and len(y_aligned) > 1:
                try:
                    corr, _ = pearsonr(x_aligned, y_aligned)
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                except:
                    continue
        
        return float(best_lag)
    
    def _build_minimal_graph(self, n_sensors: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build minimal connected graph (ring topology)
        """
        if n_sensors is None and len(self.history) > 0:
            n_sensors = len(self.history[0])
        elif n_sensors is None:
            n_sensors = 10  # Default
        
        edges = []
        edge_attrs = []
        
        # Ring topology
        for i in range(n_sensors):
            j = (i + 1) % n_sensors
            edges.append([i, j])
            edges.append([j, i])
            edge_attrs.append([0.5, 0.0])  # Default correlation and lag
            edge_attrs.append([0.5, 0.0])
        
        edge_index = np.array(edges).T.astype(np.int64)
        edge_attr = np.array(edge_attrs, dtype=np.float32)
        
        return edge_index, edge_attr


class MultiModalGraphBuilder:
    """
    Combines multiple graph construction approaches
    """
    
    def __init__(
        self,
        sensor_positions: Optional[np.ndarray] = None,
        spatial_weight: float = 0.3,
        temporal_weight: float = 0.7
    ):
        """
        Args:
            sensor_positions: [n_sensors, 3] sensor 3D positions
            spatial_weight: Weight for spatial proximity edges
            temporal_weight: Weight for temporal correlation edges
        """
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        
        # Initialize builders
        self.spatial_builder = None
        if sensor_positions is not None:
            self.spatial_builder = SpatialProximityGraph()
        
        self.temporal_builder = TemporalCorrelationGraph()
        
        logger.info("Initialized MultiModalGraphBuilder")
    
    def build_graph(
        self, 
        sensor_data: np.ndarray,
        sensor_positions: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build combined graph from multiple sources
        
        Args:
            sensor_data: [n_sensors] latest sensor readings
            sensor_positions: [n_sensors, 3] sensor positions (optional)
            
        Returns:
            edge_index: [2, n_edges] combined edge connectivity
            edge_attr: [n_edges, 3] combined edge attributes [distance, correlation, lag]
        """
        n_sensors = len(sensor_data)
        
        # Build spatial graph
        spatial_edges, spatial_attrs = None, None
        if self.spatial_builder is not None and sensor_positions is not None:
            spatial_edges, spatial_attrs = self.spatial_builder.build_graph(sensor_positions)
            spatial_attrs = spatial_attrs.squeeze(-1)  # [n_edges]
        elif sensor_positions is not None:
            # Create spatial builder on-the-fly
            spatial_builder = SpatialProximityGraph()
            spatial_edges, spatial_attrs = spatial_builder.build_graph(sensor_positions)
            spatial_attrs = spatial_attrs.squeeze(-1)
        
        # Build temporal graph
        self.temporal_builder.update_history(sensor_data)
        temporal_edges, temporal_attrs = self.temporal_builder.build_graph()
        
        # Combine graphs
        if spatial_edges is not None:
            # Merge edges (union)
            all_edges = np.concatenate([spatial_edges.T, temporal_edges.T], axis=0)
            # Remove duplicates
            unique_edges, unique_indices = np.unique(all_edges, axis=0, return_index=True)
            edge_index = unique_edges.T
            
            # Combine attributes (this is simplified - in practice would need more sophisticated merging)
            # For now, we'll pad spatial attributes with zeros for missing dimensions
            combined_attrs = []
            for i in range(unique_edges.shape[0]):
                edge = unique_edges[i]
                # Check if this edge exists in spatial graph
                spatial_match = np.where((spatial_edges.T == edge).all(axis=1))[0]
                temporal_match = np.where((temporal_edges.T == edge).all(axis=1))[0]
                
                if len(spatial_match) > 0 and len(temporal_match) > 0:
                    # Edge exists in both
                    spatial_val = spatial_attrs[spatial_match[0]]
                    temporal_vals = temporal_attrs[temporal_match[0]]
                    combined_attr = [
                        spatial_val * self.spatial_weight,
                        temporal_vals[0] * self.temporal_weight,
                        temporal_vals[1]
                    ]
                elif len(spatial_match) > 0:
                    # Only spatial
                    spatial_val = spatial_attrs[spatial_match[0]]
                    combined_attr = [spatial_val, 0.0, 0.0]
                else:
                    # Only temporal
                    temporal_vals = temporal_attrs[temporal_match[0]]
                    combined_attr = [0.0, temporal_vals[0], temporal_vals[1]]
                
                combined_attrs.append(combined_attr)
            
            edge_attr = np.array(combined_attrs, dtype=np.float32)
        else:
            # Only temporal graph
            edge_index = temporal_edges
            # Pad temporal attributes to 3 dimensions
            padded_attrs = np.pad(temporal_attrs, ((0, 0), (0, 1)), mode='constant')
            edge_attr = padded_attrs
        
        return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_attr, dtype=torch.float)
    
    def update_graph(
        self, 
        sensor_data: np.ndarray,
        sensor_positions: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update graph with new sensor data
        
        Args:
            sensor_data: Latest sensor readings
            sensor_positions: Sensor positions (if changed)
            
        Returns:
            Updated edge_index and edge_attr tensors
        """
        return self.build_graph(sensor_data, sensor_positions)


def create_graph_builder(
    sensor_positions: Optional[np.ndarray] = None,
    **kwargs
) -> MultiModalGraphBuilder:
    """
    Factory function for creating graph builder
    
    Args:
        sensor_positions: Sensor 3D positions
        **kwargs: Other parameters
        
    Returns:
        MultiModalGraphBuilder instance
    """
    builder = MultiModalGraphBuilder(
        sensor_positions=sensor_positions,
        **kwargs
    )
    
    return builder


if __name__ == "__main__":
    # Test graph builders
    
    # Create sample sensor data
    n_sensors = 8
    sensor_positions = np.random.rand(n_sensors, 3) * 20  # Random 3D positions
    sensor_data = np.random.rand(n_sensors)  # Random sensor readings
    
    print("Testing SpatialProximityGraph...")
    spatial_builder = SpatialProximityGraph(max_distance=15.0)
    spatial_edges, spatial_attrs = spatial_builder.build_graph(sensor_positions)
    print(f"Spatial graph: {spatial_edges.shape[1]} edges")
    
    print("\nTesting TemporalCorrelationGraph...")
    temporal_builder = TemporalCorrelationGraph()
    # Add some history
    for _ in range(50):
        temporal_builder.update_history(np.random.rand(n_sensors))
    temporal_edges, temporal_attrs = temporal_builder.build_graph(sensor_data)
    print(f"Temporal graph: {temporal_edges.shape[1]} edges")
    
    print("\nTesting MultiModalGraphBuilder...")
    multimodal_builder = create_graph_builder(sensor_positions)
    edge_index, edge_attr = multimodal_builder.build_graph(sensor_data, sensor_positions)
    print(f"Combined graph: {edge_index.shape[1]} edges")
    print(f"Edge attributes shape: {edge_attr.shape}")
    
    # Test incremental updates
    print("\nTesting incremental updates...")
    for i in range(5):
        new_data = np.random.rand(n_sensors)
        edge_index, edge_attr = multimodal_builder.update_graph(new_data)
        print(f"Update {i+1}: {edge_index.shape[1]} edges")