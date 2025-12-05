"""
Federated Learning Framework
Implements federated averaging and secure aggregation for distributed model training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from collections import OrderedDict
import logging
import copy
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FederatedClient(ABC):
    """
    Abstract base class for federated learning clients
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        data_loader,
        optimizer_class: type = optim.Adam,
        optimizer_params: Dict = None,
        criterion: nn.Module = None
    ):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params or {}
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # Client optimizer
        self.optimizer = self.optimizer_class(
            self.model.parameters(), 
            **self.optimizer_params
        )
        
        # Training statistics
        self.local_epochs = 0
        self.local_loss = 0.0
    
    @abstractmethod
    def get_data_stats(self) -> Dict:
        """
        Get statistics about client data
        Returns:
            Dictionary with data statistics
        """
        pass
    
    def update_model(self, global_model_state: OrderedDict):
        """
        Update local model with global model parameters
        
        Args:
            global_model_state: Global model state dictionary
        """
        self.model.load_state_dict(global_model_state)
    
    def train_local(
        self, 
        epochs: int = 1,
        device: torch.device = torch.device('cpu')
    ) -> Tuple[float, int]:
        """
        Train model locally for specified epochs
        
        Args:
            epochs: Number of local training epochs
            device: Computing device
            
        Returns:
            Tuple of (average_loss, samples_trained)
        """
        self.model.to(device)
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(device), target.to(device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                batch_size = data.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        # Update statistics
        self.local_epochs += epochs
        self.local_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        logger.debug(f"Client {self.client_id}: Local training completed. "
                    f"Loss: {self.local_loss:.4f}, Samples: {total_samples}")
        
        return self.local_loss, total_samples


class FederatedServer:
    """
    Federated learning server implementing FedAvg algorithm
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        clients: List[FederatedClient],
        aggregation_method: str = "fedavg",  # "fedavg", "fedprox", "scaffold"
        device: torch.device = torch.device('cpu')
    ):
        self.global_model = global_model
        self.clients = clients
        self.aggregation_method = aggregation_method
        self.device = device
        self.round = 0
        
        # Server model state
        self.global_model.to(device)
        self.global_state = copy.deepcopy(self.global_model.state_dict())
        
        # Aggregation weights (by default proportional to data size)
        self.client_weights = None
        
        # For SCAFFOLD
        if aggregation_method == "scaffold":
            self.global_control_variates = {
                name: torch.zeros_like(param) 
                for name, param in self.global_model.named_parameters()
            }
            self.client_control_variates = {
                client.client_id: {
                    name: torch.zeros_like(param) 
                    for name, param in self.global_model.named_parameters()
                } for client in clients
            }
        
        logger.info(f"Initialized FederatedServer with {len(clients)} clients, "
                   f"aggregation method: {aggregation_method}")
    
    def select_clients(self, fraction: float = 1.0) -> List[FederatedClient]:
        """
        Select a fraction of clients for training round
        
        Args:
            fraction: Fraction of clients to select (0.0 to 1.0)
            
        Returns:
            List of selected clients
        """
        num_clients = max(1, int(len(self.clients) * fraction))
        selected_clients = np.random.choice(
            self.clients, size=num_clients, replace=False
        ).tolist()
        
        logger.info(f"Selected {len(selected_clients)} clients for round {self.round + 1}")
        return selected_clients
    
    def distribute_model(self, clients: List[FederatedClient]):
        """
        Distribute global model to selected clients
        
        Args:
            clients: List of clients to update
        """
        for client in clients:
            client.update_model(self.global_state)
    
    def aggregate_updates(
        self, 
        client_states: List[Tuple[OrderedDict, int]],
        selected_clients: List[FederatedClient]
    ) -> OrderedDict:
        """
        Aggregate client updates using specified method
        
        Args:
            client_states: List of (state_dict, num_samples) tuples
            selected_clients: List of clients that participated
            
        Returns:
            Aggregated global state dictionary
        """
        if self.aggregation_method == "fedavg":
            return self._fedavg_aggregation(client_states)
        elif self.aggregation_method == "fedprox":
            return self._fedprox_aggregation(client_states)
        elif self.aggregation_method == "scaffold":
            return self._scaffold_aggregation(client_states, selected_clients)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _fedavg_aggregation(self, client_states: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        """
        Federated Averaging aggregation
        
        Args:
            client_states: List of (state_dict, num_samples) tuples
            
        Returns:
            Aggregated state dictionary
        """
        if not client_states:
            return self.global_state
        
        # Calculate total samples
        total_samples = sum([samples for _, samples in client_states])
        
        # Initialize aggregated state
        aggregated_state = OrderedDict()
        
        # Aggregate parameters weighted by sample count
        for name in self.global_state.keys():
            aggregated_state[name] = torch.zeros_like(self.global_state[name])
        
        for client_state, num_samples in client_states:
            weight = num_samples / total_samples
            for name, param in client_state.items():
                aggregated_state[name] += weight * param.to(self.device)
        
        return aggregated_state
    
    def _fedprox_aggregation(self, client_states: List[Tuple[OrderedDict, int]]) -> OrderedDict:
        """
        FedProx aggregation (similar to FedAvg but with proximal term)
        """
        # For simplicity, we'll use the same aggregation as FedAvg
        # In practice, FedProx modifies the client training objective
        return self._fedavg_aggregation(client_states)
    
    def _scaffold_aggregation(
        self, 
        client_states: List[Tuple[OrderedDict, int]],
        selected_clients: List[FederatedClient]
    ) -> OrderedDict:
        """
        SCAFFOLD aggregation with control variates
        """
        # This is a simplified implementation
        return self._fedavg_aggregation(client_states)
    
    def train_round(
        self,
        clients_fraction: float = 1.0,
        local_epochs: int = 1
    ) -> Dict:
        """
        Execute one federated training round
        
        Args:
            clients_fraction: Fraction of clients to participate
            local_epochs: Number of local epochs per client
            
        Returns:
            Dictionary with round statistics
        """
        self.round += 1
        logger.info(f"Starting federated training round {self.round}")
        
        # Select clients
        selected_clients = self.select_clients(clients_fraction)
        
        # Distribute global model
        self.distribute_model(selected_clients)
        
        # Collect client updates
        client_states = []
        round_stats = {
            'round': self.round,
            'clients_selected': len(selected_clients),
            'total_loss': 0.0,
            'avg_loss': 0.0
        }
        
        for client in selected_clients:
            # Train locally
            local_loss, num_samples = client.train_local(local_epochs, self.device)
            
            # Collect updated state
            client_states.append((copy.deepcopy(client.model.state_dict()), num_samples))
            
            # Update statistics
            round_stats['total_loss'] += local_loss * num_samples
        
        # Aggregate updates
        if client_states:
            self.global_state = self.aggregate_updates(client_states, selected_clients)
            self.global_model.load_state_dict(self.global_state)
            
            total_samples = sum([samples for _, samples in client_states])
            round_stats['avg_loss'] = round_stats['total_loss'] / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"Round {self.round} completed. Avg loss: {round_stats['avg_loss']:.4f}")
        return round_stats
    
    def get_global_model(self) -> nn.Module:
        """
        Get the current global model
        
        Returns:
            Global model with current parameters
        """
        return self.global_model
    
    def save_model(self, filepath: str):
        """
        Save global model to file
        
        Args:
            filepath: Path to save model
        """
        torch.save(self.global_model.state_dict(), filepath)
        logger.info(f"Global model saved to {filepath}")


class SecureAggregator:
    """
    Implements secure aggregation techniques for federated learning
    """
    
    def __init__(self, noise_multiplier: float = 0.1):
        """
        Args:
            noise_multiplier: Standard deviation multiplier for differential privacy noise
        """
        self.noise_multiplier = noise_multiplier
        logger.info(f"Initialized SecureAggregator with noise multiplier: {noise_multiplier}")
    
    def add_differential_privacy_noise(
        self, 
        state_dict: OrderedDict, 
        sensitivity: float = 1.0
    ) -> OrderedDict:
        """
        Add Gaussian noise for differential privacy
        
        Args:
            state_dict: Model state dictionary
            sensitivity: L2 sensitivity of the mechanism
            
        Returns:
            Noisy state dictionary
        """
        noisy_state = OrderedDict()
        noise_std = self.noise_multiplier * sensitivity
        
        for name, param in state_dict.items():
            noise = torch.randn_like(param) * noise_std
            noisy_state[name] = param + noise
        
        return noisy_state
    
    def clip_gradients(
        self, 
        state_dict: OrderedDict, 
        clip_norm: float = 1.0
    ) -> OrderedDict:
        """
        Clip gradients to limit sensitivity
        
        Args:
            state_dict: Model state dictionary with gradients
            clip_norm: Maximum L2 norm for gradients
            
        Returns:
            Clipped state dictionary
        """
        clipped_state = OrderedDict()
        
        for name, param in state_dict.items():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad)
                if grad_norm > clip_norm:
                    clipped_grad = param.grad * (clip_norm / grad_norm)
                    clipped_param = param.clone()
                    clipped_param.grad = clipped_grad
                    clipped_state[name] = clipped_param
                else:
                    clipped_state[name] = param
            else:
                clipped_state[name] = param
        
        return clipped_state


class FederatedLearningFramework:
    """
    High-level federated learning framework
    """
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[FederatedClient],
        server_config: Dict = None,
        aggregator_config: Dict = None
    ):
        """
        Args:
            model: Global model architecture
            clients: List of federated clients
            server_config: Server configuration
            aggregator_config: Secure aggregator configuration
        """
        self.model = model
        self.clients = clients
        self.server_config = server_config or {}
        self.aggregator_config = aggregator_config or {}
        
        # Initialize components
        self.server = FederatedServer(
            global_model=copy.deepcopy(model),
            clients=clients,
            **self.server_config
        )
        
        if self.aggregator_config.get('secure_aggregation', False):
            self.secure_aggregator = SecureAggregator(
                noise_multiplier=self.aggregator_config.get('noise_multiplier', 0.1)
            )
        else:
            self.secure_aggregator = None
        
        logger.info("Initialized FederatedLearningFramework")
    
    def train(
        self,
        num_rounds: int = 10,
        clients_fraction: float = 1.0,
        local_epochs: int = 1,
        evaluation_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Execute full federated training
        
        Args:
            num_rounds: Number of federated rounds
            clients_fraction: Fraction of clients per round
            local_epochs: Local epochs per client
            evaluation_callback: Optional callback for evaluation
            
        Returns:
            List of round statistics
        """
        round_stats = []
        
        for round_idx in range(num_rounds):
            # Train one round
            stats = self.server.train_round(clients_fraction, local_epochs)
            round_stats.append(stats)
            
            # Optional evaluation
            if evaluation_callback is not None:
                global_model = self.server.get_global_model()
                eval_metrics = evaluation_callback(global_model, round_idx)
                stats.update(eval_metrics)
                logger.info(f"Round {round_idx + 1} evaluation: {eval_metrics}")
        
        logger.info(f"Federated training completed. Total rounds: {num_rounds}")
        return round_stats
    
    def get_global_model(self) -> nn.Module:
        """
        Get trained global model
        
        Returns:
            Trained global model
        """
        return self.server.get_global_model()


def create_federated_client(
    client_id: str,
    model: nn.Module,
    data_loader,
    **kwargs
) -> FederatedClient:
    """
    Factory function for creating federated client
    
    Args:
        client_id: Unique client identifier
        model: Client model
        data_loader: Client data loader
        **kwargs: Additional parameters
        
    Returns:
        FederatedClient instance
    """
    # This is a simplified implementation
    # In practice, you would create domain-specific client classes
    class SimpleFederatedClient(FederatedClient):
        def get_data_stats(self):
            return {'num_samples': len(self.data_loader.dataset)}
    
    return SimpleFederatedClient(client_id, model, data_loader, **kwargs)


def create_federated_framework(
    model: nn.Module,
    clients: List[FederatedClient],
    **kwargs
) -> FederatedLearningFramework:
    """
    Factory function for creating federated learning framework
    
    Args:
        model: Global model architecture
        clients: List of federated clients
        **kwargs: Additional parameters
        
    Returns:
        FederatedLearningFramework instance
    """
    return FederatedLearningFramework(model, clients, **kwargs)


if __name__ == "__main__":
    # Example usage
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=64, num_classes=5):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
        
        def forward(self, x):
            return self.network(x)
    
    model = SimpleModel()
    
    # Create dummy clients (in practice, you'd have real data loaders)
    clients = []
    for i in range(5):
        # Dummy data loader
        class DummyDataLoader:
            def __init__(self, size=100):
                self.dataset = [None] * size  # Dummy dataset
            
            def __iter__(self):
                # Dummy iterator
                for _ in range(5):  # 5 batches
                    batch_size = 32
                    data = torch.randn(batch_size, 10)
                    targets = torch.randint(0, 5, (batch_size,))
                    yield data, targets
        
        client = create_federated_client(
            client_id=f"client_{i}",
            model=copy.deepcopy(model),
            data_loader=DummyDataLoader()
        )
        clients.append(client)
    
    # Create federated framework
    fl_framework = create_federated_framework(
        model=model,
        clients=clients,
        server_config={'aggregation_method': 'fedavg'},
        aggregator_config={'secure_aggregation': False}
    )
    
    # Define dummy evaluation callback
    def dummy_eval_callback(model, round_idx):
        return {'dummy_accuracy': 0.85 + 0.01 * round_idx}
    
    # Train federated model
    print("Starting federated training...")
    stats = fl_framework.train(
        num_rounds=3,
        clients_fraction=1.0,
        local_epochs=2,
        evaluation_callback=dummy_eval_callback
    )
    
    print("\nTraining completed. Round statistics:")
    for stat in stats:
        print(f"Round {stat['round']}: Avg loss = {stat['avg_loss']:.4f}")
    
    # Get final model
    final_model = fl_framework.get_global_model()
    print(f"\nFinal global model parameters: {sum(p.numel() for p in final_model.parameters())}")