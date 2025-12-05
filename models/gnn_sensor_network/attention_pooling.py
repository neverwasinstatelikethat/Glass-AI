"""
Attention-based Pooling for Graph Neural Networks
Implements various attention mechanisms for aggregating node information in GNNs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalAttentionPooling(nn.Module):
    """
    Global attention pooling for graph representations
    Computes attention weights for each node and aggregates accordingly
    """
    
    def __init__(self, input_dim: int, attention_dim: int = 64):
        """
        Args:
            input_dim: Dimension of node features
            attention_dim: Dimension of attention computation
        """
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # Attention computation layers
        self.attention_linear = nn.Linear(input_dim, attention_dim)
        self.attention_context = nn.Parameter(torch.randn(attention_dim))
        
        # Output projection
        self.output_projection = nn.Linear(input_dim, input_dim)
        
        self._init_weights()
        logger.info(f"Initialized GlobalAttentionPooling with dims: {input_dim}->{attention_dim}")
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.attention_linear.weight)
        nn.init.constant_(self.attention_linear.bias, 0)
        nn.init.normal_(self.attention_context, mean=0, std=0.1)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0)
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute global attention-pooled representation
        
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            Graph-level representation [batch_size, input_dim]
        """
        # Compute attention scores for each node
        attention_logits = self.attention_linear(x)  # [num_nodes, attention_dim]
        attention_scores = torch.matmul(
            torch.tanh(attention_logits), 
            self.attention_context
        )  # [num_nodes]
        
        # Apply softmax within each graph
        attention_weights = self._sparse_softmax(attention_scores, batch)  # [num_nodes]
        
        # Weighted sum of node features
        weighted_features = x * attention_weights.unsqueeze(-1)  # [num_nodes, input_dim]
        graph_representations = global_sum_pool(weighted_features, batch)  # [batch_size, input_dim]
        
        # Output projection
        output = self.output_projection(graph_representations)
        
        return output
    
    def _sparse_softmax(self, scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Apply softmax within each graph in the batch
        
        Args:
            scores: Attention scores [num_nodes]
            batch: Batch indices [num_nodes]
            
        Returns:
            Normalized attention weights [num_nodes]
        """
        # Subtract max for numerical stability
        max_scores = scatter_max(scores, batch)[0][batch]
        scores_stable = scores - max_scores
        
        # Exponentiate
        exp_scores = torch.exp(scores_stable)
        
        # Sum within each graph
        sum_exp_scores = scatter_add(exp_scores, batch)[batch]
        
        # Normalize
        attention_weights = exp_scores / (sum_exp_scores + 1e-8)
        
        return attention_weights


def global_sum_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Global sum pooling helper function
    
    Args:
        x: Node features [num_nodes, feature_dim]
        batch: Batch indices [num_nodes]
        
    Returns:
        Pooled features [batch_size, feature_dim]
    """
    # Manual implementation since torch_geometric might not be available in this context
    batch_size = batch.max().item() + 1
    feature_dim = x.shape[1]
    pooled = torch.zeros(batch_size, feature_dim, device=x.device, dtype=x.dtype)
    
    for i in range(batch_size):
        mask = batch == i
        pooled[i] = x[mask].sum(dim=0)
    
    return pooled


def scatter_max(src: torch.Tensor, index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple scatter max implementation"""
    # This is a simplified version - in practice, use torch_scatter.scatter_max
    unique_indices = torch.unique(index)
    max_values = torch.zeros_like(unique_indices, dtype=src.dtype, device=src.device)
    max_indices = torch.zeros_like(unique_indices, dtype=torch.long, device=src.device)
    
    for i, idx in enumerate(unique_indices):
        mask = index == idx
        max_values[i], max_indices[i] = src[mask].max(dim=0)
    
    return max_values, max_indices


def scatter_add(src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Simple scatter add implementation"""
    # This is a simplified version - in practice, use torch_scatter.scatter_add
    unique_indices = torch.unique(index)
    summed = torch.zeros_like(unique_indices, dtype=src.dtype, device=src.device)
    
    for i, idx in enumerate(unique_indices):
        mask = index == idx
        summed[i] = src[mask].sum()
    
    return summed


class HierarchicalAttentionPooling(nn.Module):
    """
    Hierarchical attention pooling with multiple attention heads
    """
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        hidden_dim: int = 64
    ):
        """
        Args:
            input_dim: Dimension of input features
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension for attention computation
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Multi-head attention layers
        self.attention_heads = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_heads)
        ])
        
        # Context vectors for each head
        self.context_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim)) for _ in range(num_heads)
        ])
        
        # Head combination
        self.head_combination = nn.Linear(num_heads * input_dim, input_dim)
        
        self._init_weights()
        logger.info(f"Initialized HierarchicalAttentionPooling with {num_heads} heads")
    
    def _init_weights(self):
        """Initialize weights"""
        for head in self.attention_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, 0)
        
        for context in self.context_vectors:
            nn.init.normal_(context, mean=0, std=0.1)
        
        nn.init.xavier_uniform_(self.head_combination.weight)
        nn.init.constant_(self.head_combination.bias, 0)
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical attention pooling
        
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            Pooled graph representation [batch_size, input_dim]
        """
        head_outputs = []
        
        # Compute attention for each head
        for head, context in zip(self.attention_heads, self.context_vectors):
            # Compute attention scores
            attention_logits = head(x)  # [num_nodes, hidden_dim]
            attention_scores = torch.matmul(
                torch.tanh(attention_logits),
                context
            )  # [num_nodes]
            
            # Apply softmax within each graph
            attention_weights = self._sparse_softmax(attention_scores, batch)  # [num_nodes]
            
            # Weighted sum for this head
            weighted_features = x * attention_weights.unsqueeze(-1)  # [num_nodes, input_dim]
            head_output = global_sum_pool(weighted_features, batch)  # [batch_size, input_dim]
            head_outputs.append(head_output)
        
        # Concatenate and combine head outputs
        concatenated = torch.cat(head_outputs, dim=-1)  # [batch_size, num_heads * input_dim]
        combined = self.head_combination(concatenated)  # [batch_size, input_dim]
        
        return combined


class Set2SetPooling(nn.Module):
    """
    Set2Set pooling - iterative attention-based aggregation
    Based on the paper "Order Matters: Sequence to sequence for sets"
    """
    
    def __init__(
        self,
        input_dim: int,
        processing_steps: int = 4,
        num_layers: int = 1
    ):
        """
        Args:
            input_dim: Dimension of input features
            processing_steps: Number of attention iterations
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        
        # LSTM for processing attention outputs
        self.lstm = nn.LSTM(
            input_size=input_dim * 2,  # [input, previous output]
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=False
        )
        
        # Attention computation
        self.attention_linear = nn.Linear(input_dim, input_dim)
        
        self._init_weights()
        logger.info(f"Initialized Set2SetPooling with {processing_steps} steps")
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.attention_linear.weight)
        nn.init.constant_(self.attention_linear.bias, 0)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Set2Set pooling
        
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            Pooled representation [batch_size, input_dim * 2]
        """
        batch_size = batch.max().item() + 1
        
        # Initialize LSTM hidden state
        h = torch.zeros(self.num_layers, batch_size, self.input_dim, 
                       device=x.device, dtype=x.dtype)
        c = torch.zeros(self.num_layers, batch_size, self.input_dim, 
                       device=x.device, dtype=x.dtype)
        
        # Global average as initial input
        q_star = global_mean_pool(x, batch)  # [batch_size, input_dim]
        
        # Iterative attention processing
        for _ in range(self.processing_steps):
            # Duplicate q_star for each node in its graph
            q_star_expanded = q_star[batch]  # [num_nodes, input_dim]
            
            # Compute attention weights
            attention_input = self.attention_linear(x)  # [num_nodes, input_dim]
            attention_scores = torch.sum(attention_input * q_star_expanded, dim=1)  # [num_nodes]
            attention_weights = self._sparse_softmax(attention_scores, batch)  # [num_nodes]
            
            # Compute attention-weighted sum
            r = global_sum_pool(x * attention_weights.unsqueeze(-1), batch)  # [batch_size, input_dim]
            
            # LSTM step
            lstm_input = torch.cat([q_star, r], dim=-1).unsqueeze(0)  # [1, batch_size, input_dim * 2]
            _, (h, c) = self.lstm(lstm_input, (h, c))
            q_star = h[-1]  # [batch_size, input_dim]
        
        # Final output concatenates q_star and r from last step
        r_final = global_sum_pool(x * attention_weights.unsqueeze(-1), batch)
        output = torch.cat([q_star, r_final], dim=-1)  # [batch_size, input_dim * 2]
        
        return output


class AdaptiveGraphPooling(nn.Module):
    """
    Adaptive pooling that selects the best pooling strategy based on graph properties
    """
    
    def __init__(
        self,
        input_dim: int,
        strategies: list = ['mean', 'max', 'attention', 'set2set']
    ):
        """
        Args:
            input_dim: Dimension of input features
            strategies: List of pooling strategies to use
        """
        super().__init__()
        self.input_dim = input_dim
        self.strategies = strategies
        
        # Initialize pooling modules
        self.pooling_modules = nn.ModuleDict()
        if 'attention' in strategies:
            self.pooling_modules['attention'] = GlobalAttentionPooling(input_dim)
        if 'set2set' in strategies:
            self.pooling_modules['set2set'] = Set2SetPooling(input_dim)
        
        # Strategy selector network
        self.strategy_selector = nn.Sequential(
            nn.Linear(input_dim * len(strategies), input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, len(strategies)),
            nn.Softmax(dim=-1)
        )
        
        logger.info(f"Initialized AdaptiveGraphPooling with strategies: {strategies}")
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Adaptive pooling
        
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            Adaptively pooled representation
        """
        # Apply all pooling strategies
        pooled_results = []
        
        # Mean pooling
        if 'mean' in self.strategies:
            pooled_results.append(global_mean_pool(x, batch))
        
        # Max pooling
        if 'max' in self.strategies:
            pooled_results.append(global_max_pool(x, batch))
        
        # Attention pooling
        if 'attention' in self.strategies:
            pooled_results.append(self.pooling_modules['attention'](x, batch))
        
        # Set2Set pooling
        if 'set2set' in self.strategies:
            pooled_results.append(self.pooling_modules['set2set'](x, batch))
        
        # Stack results
        stacked = torch.stack(pooled_results, dim=1)  # [batch_size, num_strategies, feature_dim]
        
        # Compute strategy weights
        flattened = stacked.view(stacked.shape[0], -1)  # [batch_size, num_strategies * feature_dim]
        strategy_weights = self.strategy_selector(flattened)  # [batch_size, num_strategies]
        
        # Weighted combination
        weighted = stacked * strategy_weights.unsqueeze(-1)  # [batch_size, num_strategies, feature_dim]
        output = weighted.sum(dim=1)  # [batch_size, feature_dim]
        
        return output


def create_attention_pooling(
    input_dim: int,
    pooling_type: str = "global_attention",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating attention pooling modules
    
    Args:
        input_dim: Input feature dimension
        pooling_type: Type of pooling ('global_attention', 'hierarchical', 'set2set', 'adaptive')
        **kwargs: Additional parameters
        
    Returns:
        Attention pooling module
    """
    if pooling_type == "global_attention":
        return GlobalAttentionPooling(input_dim, **kwargs)
    elif pooling_type == "hierarchical":
        return HierarchicalAttentionPooling(input_dim, **kwargs)
    elif pooling_type == "set2set":
        return Set2SetPooling(input_dim, **kwargs)
    elif pooling_type == "adaptive":
        return AdaptiveGraphPooling(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown pooling type: {pooling_type}")


if __name__ == "__main__":
    # Test attention pooling modules
    
    # Create sample data
    num_nodes = 100
    input_dim = 64
    batch_size = 4
    
    # Node features
    x = torch.randn(num_nodes, input_dim)
    
    # Batch assignments (4 graphs with varying sizes)
    batch_assignments = []
    nodes_per_graph = [20, 30, 25, 25]
    for i, count in enumerate(nodes_per_graph):
        batch_assignments.extend([i] * count)
    batch = torch.tensor(batch_assignments, dtype=torch.long)
    
    print("Testing GlobalAttentionPooling...")
    global_attention = create_attention_pooling(input_dim, "global_attention")
    pooled_global = global_attention(x, batch)
    print(f"Global attention pooled shape: {pooled_global.shape}")
    
    print("\nTesting HierarchicalAttentionPooling...")
    hierarchical = create_attention_pooling(input_dim, "hierarchical", num_heads=4)
    pooled_hierarchical = hierarchical(x, batch)
    print(f"Hierarchical pooled shape: {pooled_hierarchical.shape}")
    
    print("\nTesting Set2SetPooling...")
    set2set = create_attention_pooling(input_dim, "set2set")
    pooled_set2set = set2set(x, batch)
    print(f"Set2Set pooled shape: {pooled_set2set.shape}")
    
    print("\nTesting AdaptiveGraphPooling...")
    adaptive = create_attention_pooling(input_dim, "adaptive")
    pooled_adaptive = adaptive(x, batch)
    print(f"Adaptive pooled shape: {pooled_adaptive.shape}")