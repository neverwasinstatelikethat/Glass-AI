"""
Continuous Learning Framework
Implements lifelong learning with catastrophic forgetting prevention and model adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import deque
import logging
from abc import ABC, abstractmethod
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperienceReplayBuffer:
    """
    Experience replay buffer for storing and sampling past experiences
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, experience: Tuple):
        """
        Add experience to buffer
        
        Args:
            experience: Tuple of (state, action, reward, next_state, done)
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return list(np.random.choice(self.buffer, size=batch_size, replace=False))
    
    def __len__(self):
        return len(self.buffer)


class RegularizationMethod(ABC):
    """
    Abstract base class for regularization methods to prevent catastrophic forgetting
    """
    
    @abstractmethod
    def compute_regularization_loss(
        self, 
        model: nn.Module, 
        previous_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute regularization loss to prevent forgetting
        
        Args:
            model: Current model
            previous_model: Previous model state
            
        Returns:
            Regularization loss tensor
        """
        pass


class ElasticWeightConsolidation(RegularizationMethod):
    """
    Elastic Weight Consolidation (EWC) regularization
    """
    
    def __init__(self, importance_coeff: float = 1000.0):
        """
        Args:
            importance_coeff: Coefficient for importance weighting
        """
        self.importance_coeff = importance_coeff
        self.fisher_information = {}
        self.optimal_params = {}
    
    def compute_fisher_information(
        self, 
        model: nn.Module, 
        dataloader,
        criterion: nn.Module,
        device: torch.device
    ):
        """
        Compute Fisher Information Matrix diagonal approximation
        
        Args:
            model: Model to compute Fisher information for
            dataloader: DataLoader with previous task data
            criterion: Loss function
            device: Computing device
        """
        model.eval()
        model.to(device)
        
        # Initialize Fisher information
        fisher_info = {}
        for name, param in model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)
        
        # Compute gradients on previous task data
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.data.pow(2)
        
        # Average over dataset
        dataset_size = len(dataloader.dataset)
        for name in fisher_info:
            fisher_info[name] /= dataset_size
            # Store Fisher information and optimal parameters
            self.fisher_information[name] = fisher_info[name]
            self.optimal_params[name] = model.state_dict()[name].clone()
        
        model.train()
    
    def compute_regularization_loss(
        self, 
        model: nn.Module, 
        previous_model: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Compute EWC regularization loss
        
        Args:
            model: Current model
            previous_model: Not used in EWC (uses stored Fisher info)
            
        Returns:
            EWC regularization loss
        """
        if not self.fisher_information:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_information:
                fisher = self.fisher_information[name]
                opt_param = self.optimal_params[name]
                loss += (fisher * (param - opt_param).pow(2)).sum()
        
        return self.importance_coeff / 2 * loss


class MemoryAwareSynapses(RegularizationMethod):
    """
    Memory Aware Synapses (MAS) regularization
    """
    
    def __init__(self, importance_coeff: float = 1.0):
        """
        Args:
            importance_coeff: Coefficient for importance weighting
        """
        self.importance_coeff = importance_coeff
        self.omega = {}
    
    def compute_importance(
        self, 
        model: nn.Module, 
        dataloader,
        device: torch.device
    ):
        """
        Compute parameter importance using MAS
        
        Args:
            model: Model to compute importance for
            dataloader: DataLoader with previous task data
            device: Computing device
        """
        model.eval()
        model.to(device)
        
        # Initialize importance
        omega = {}
        for name, param in model.named_parameters():
            omega[name] = torch.zeros_like(param)
        
        # Compute gradients of output w.r.t. parameters
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            model.zero_grad()
            output = model(data)
            
            # Compute L2 norm of output
            output_norm = torch.norm(output, p=2, dim=1).mean()
            output_norm.backward()
            
            # Accumulate gradient magnitudes
            for name, param in model.named_parameters():
                if param.grad is not None:
                    omega[name] += param.grad.data.abs()
        
        # Average over dataset and store
        dataset_size = len(dataloader.dataset)
        for name in omega:
            self.omega[name] = omega[name] / dataset_size
        
        model.train()
    
    def compute_regularization_loss(
        self, 
        model: nn.Module, 
        previous_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute MAS regularization loss
        
        Args:
            model: Current model
            previous_model: Previous model state
            
        Returns:
            MAS regularization loss
        """
        if not self.omega:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        loss = 0.0
        prev_state = previous_model.state_dict()
        
        for name, param in model.named_parameters():
            if name in self.omega:
                omega = self.omega[name]
                prev_param = prev_state[name]
                loss += (omega * (param - prev_param).pow(2)).sum()
        
        return self.importance_coeff / 2 * loss


class ProgressiveNeuralNetworks:
    """
    Progressive Neural Networks approach for continual learning
    """
    
    def __init__(self, base_model: nn.Module, num_tasks: int = 10):
        """
        Args:
            base_model: Base model architecture
            num_tasks: Expected number of tasks
        """
        self.base_model = base_model
        self.num_tasks = num_tasks
        self.task_columns = nn.ModuleList([copy.deepcopy(base_model)])
        self.lateral_connections = nn.ModuleList()
        
        logger.info(f"Initialized ProgressiveNeuralNetworks with {num_tasks} expected tasks")
    
    def add_task_column(self):
        """
        Add new column for new task
        """
        new_column = copy.deepcopy(self.base_model)
        self.task_columns.append(new_column)
        
        # Add lateral connections from all previous columns
        current_task = len(self.task_columns) - 1
        lateral_layer = nn.ModuleList([
            self._create_lateral_connection(prev_column, new_column)
            for prev_column in self.task_columns[:-1]
        ])
        self.lateral_connections.append(lateral_layer)
        
        logger.info(f"Added task column {current_task}")
    
    def _create_lateral_connection(self, from_model: nn.Module, to_model: nn.Module) -> nn.Module:
        """
        Create lateral connection between two model columns
        
        Args:
            from_model: Source model
            to_model: Target model
            
        Returns:
            Lateral connection module
        """
        # Simplified implementation - in practice, this would be more complex
        connections = nn.ModuleDict()
        
        # Create linear adapters for each layer
        for (name1, layer1), (name2, layer2) in zip(
            from_model.named_modules(), to_model.named_modules()
        ):
            if isinstance(layer1, (nn.Linear, nn.Conv2d)):
                if hasattr(layer1, 'out_features'):
                    adapter = nn.Linear(layer1.out_features, layer2.in_features)
                elif hasattr(layer1, 'out_channels'):
                    adapter = nn.Conv2d(
                        layer1.out_channels, layer2.in_channels,
                        kernel_size=1, stride=1, padding=0
                    )
                else:
                    continue
                connections[name1 + '_to_' + name2] = adapter
        
        return connections
    
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass through specific task column
        
        Args:
            x: Input tensor
            task_id: Task identifier
            
        Returns:
            Output tensor
        """
        if task_id >= len(self.task_columns):
            raise ValueError(f"Task ID {task_id} exceeds number of columns {len(self.task_columns)}")
        
        # Forward through main column
        output = self.task_columns[task_id](x)
        
        # Add contributions from lateral connections (if not first task)
        if task_id > 0:
            for prev_task in range(task_id):
                prev_output = self.task_columns[prev_task](x)
                # Apply lateral connections
                lateral_conn = self.lateral_connections[task_id - 1][prev_task]
                # This is simplified - actual implementation would depend on layer types
                adapted_output = prev_output  # Placeholder
                for adapter in lateral_conn.values():
                    if isinstance(adapter, nn.Linear) and len(adapted_output.shape) == 2:
                        adapted_output = adapter(adapted_output)
                    elif isinstance(adapter, nn.Conv2d) and len(adapted_output.shape) == 4:
                        adapted_output = adapter(adapted_output)
                output += adapted_output
        
        return output


class ContinualLearningOptimizer:
    """
    Optimizer wrapper for continual learning with adaptive learning rates
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        adaptation_strategy: str = "stable",  # "stable", "adaptive", "conservative"
        learning_rate_decay: float = 0.9,
        min_lr: float = 1e-6
    ):
        """
        Args:
            optimizer: Base optimizer
            adaptation_strategy: Learning rate adaptation strategy
            learning_rate_decay: Decay factor for learning rate
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.adaptation_strategy = adaptation_strategy
        self.learning_rate_decay = learning_rate_decay
        self.min_lr = min_lr
        self.task_iteration = 0
        self.task_switches = []
        
        # Store original learning rates
        self.original_lrs = []
        for param_group in self.optimizer.param_groups:
            self.original_lrs.append(param_group['lr'])
    
    def step(self):
        """Perform optimization step"""
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
    
    def switch_task(self):
        """Adjust learning rates when switching to new task"""
        self.task_iteration += 1
        self.task_switches.append(self.task_iteration)
        
        if self.adaptation_strategy == "adaptive":
            # Reduce learning rate for all parameters
            for i, param_group in enumerate(self.optimizer.param_groups):
                new_lr = max(
                    self.original_lrs[i] * (self.learning_rate_decay ** self.task_iteration),
                    self.min_lr
                )
                param_group['lr'] = new_lr
        elif self.adaptation_strategy == "conservative":
            # More aggressive decay
            for i, param_group in enumerate(self.optimizer.param_groups):
                new_lr = max(
                    self.original_lrs[i] * (0.5 ** self.task_iteration),
                    self.min_lr
                )
                param_group['lr'] = new_lr
        
        logger.info(f"Switched to task {self.task_iteration}, adjusted learning rates")


class ContinualLearningFramework:
    """
    Main continual learning framework integrating multiple approaches
    """
    
    def __init__(
        self,
        model: nn.Module,
        regularization_method: RegularizationMethod,
        optimizer: optim.Optimizer,
        experience_replay: bool = True,
        replay_buffer_size: int = 10000,
        progressive_networks: bool = False,
        num_expected_tasks: int = 10
    ):
        """
        Args:
            model: Base model
            regularization_method: Method to prevent catastrophic forgetting
            optimizer: Optimizer
            experience_replay: Whether to use experience replay
            replay_buffer_size: Size of experience replay buffer
            progressive_networks: Whether to use progressive networks
            num_expected_tasks: Expected number of tasks (for progressive networks)
        """
        self.model = model
        self.regularization_method = regularization_method
        self.optimizer_wrapper = ContinualLearningOptimizer(optimizer)
        self.progressive_networks = progressive_networks
        
        # Experience replay
        if experience_replay:
            self.replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        else:
            self.replay_buffer = None
        
        # Progressive networks
        if progressive_networks:
            self.prog_nn = ProgressiveNeuralNetworks(model, num_expected_tasks)
        else:
            self.prog_nn = None
        
        # Task tracking
        self.current_task = 0
        self.task_data_loaders = {}
        self.task_criterions = {}
        
        logger.info(f"Initialized ContinualLearningFramework with "
                   f"replay={experience_replay}, progressive={progressive_networks}")
    
    def register_task(
        self, 
        task_id: int, 
        data_loader,
        criterion: nn.Module
    ):
        """
        Register a new task
        
        Args:
            task_id: Task identifier
            data_loader: DataLoader for task data
            criterion: Loss function for task
        """
        self.task_data_loaders[task_id] = data_loader
        self.task_criterions[task_id] = criterion
        
        # Add new column for progressive networks
        if self.prog_nn and task_id >= len(self.prog_nn.task_columns):
            self.prog_nn.add_task_column()
        
        logger.info(f"Registered task {task_id}")
    
    def switch_to_task(self, task_id: int):
        """
        Switch to learning a specific task
        
        Args:
            task_id: Task identifier
        """
        if task_id not in self.task_data_loaders:
            raise ValueError(f"Task {task_id} not registered")
        
        self.current_task = task_id
        self.optimizer_wrapper.switch_task()
        logger.info(f"Switched to task {task_id}")
    
    def train_task(
        self,
        task_id: int,
        epochs: int = 10,
        device: torch.device = torch.device('cpu'),
        replay_batch_size: int = 32,
        ewc_lambda: float = 100.0
    ):
        """
        Train on a specific task
        
        Args:
            task_id: Task identifier
            epochs: Number of epochs
            device: Computing device
            replay_batch_size: Batch size for replay samples
            ewc_lambda: EWC regularization strength
        """
        if task_id not in self.task_data_loaders:
            raise ValueError(f"Task {task_id} not registered")
        
        self.switch_to_task(task_id)
        
        data_loader = self.task_data_loaders[task_id]
        criterion = self.task_criterions[task_id]
        
        # Store previous model for regularization
        previous_model = copy.deepcopy(self.model)
        
        self.model.to(device)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            total_samples = 0
            
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                if self.prog_nn:
                    output = self.prog_nn(data, task_id)
                else:
                    output = self.model(data)
                
                # Compute task loss
                task_loss = criterion(output, target)
                
                # Compute regularization loss
                reg_loss = self.regularization_method.compute_regularization_loss(
                    self.model, previous_model
                )
                
                # Combined loss
                loss = task_loss + ewc_lambda * reg_loss
                
                # Experience replay loss
                if self.replay_buffer and len(self.replay_buffer) > 0:
                    replay_samples = self.replay_buffer.sample(replay_batch_size)
                    if replay_samples:
                        replay_loss = self._compute_replay_loss(
                            replay_samples, device, criterion
                        )
                        loss += 0.1 * replay_loss  # Replay loss coefficient
                
                # Backward pass
                self.optimizer_wrapper.zero_grad()
                loss.backward()
                self.optimizer_wrapper.step()
                
                # Store experience for replay
                if self.replay_buffer:
                    self._store_batch_experience(data, target, output)
                
                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
            
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            logger.info(f"Task {task_id}, Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def _compute_replay_loss(
        self, 
        replay_samples: List[Tuple], 
        device: torch.device,
        criterion: nn.Module
    ) -> torch.Tensor:
        """
        Compute loss on replay samples
        
        Args:
            replay_samples: List of replay samples
            device: Computing device
            criterion: Loss function
            
        Returns:
            Replay loss tensor
        """
        if not replay_samples:
            return torch.tensor(0.0, device=device)
        
        # Unpack samples
        states, actions, rewards, next_states, dones = zip(*replay_samples)
        
        # Convert to tensors (simplified)
        states_tensor = torch.stack(states).to(device)
        actions_tensor = torch.stack(actions).to(device)
        
        # Forward pass
        if self.prog_nn:
            output = self.prog_nn(states_tensor, self.current_task)
        else:
            output = self.model(states_tensor)
        
        # Compute loss (simplified - assumes actions are targets)
        replay_loss = criterion(output, actions_tensor)
        
        return replay_loss
    
    def _store_batch_experience(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        outputs: torch.Tensor
    ):
        """
        Store batch experience in replay buffer
        
        Args:
            states: Batch of states
            actions: Batch of actions/targets
            outputs: Batch of model outputs
        """
        if not self.replay_buffer:
            return
        
        batch_size = states.size(0)
        for i in range(batch_size):
            experience = (
                states[i].detach().cpu(),
                actions[i].detach().cpu(),
                0.0,  # Reward (not used in supervised learning)
                states[i].detach().cpu(),  # Next state (same for supervised)
                False  # Done flag
            )
            self.replay_buffer.push(experience)
    
    def evaluate_task(
        self, 
        task_id: int, 
        data_loader,
        device: torch.device = torch.device('cpu')
    ) -> Dict[str, float]:
        """
        Evaluate model on specific task
        
        Args:
            task_id: Task identifier
            data_loader: DataLoader for evaluation
            device: Computing device
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        self.model.to(device)
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        criterion = self.task_criterions.get(task_id, nn.CrossEntropyLoss())
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                
                if self.prog_nn:
                    output = self.prog_nn(data, task_id)
                else:
                    output = self.model(data)
                
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }


def create_continual_learning_framework(
    model: nn.Module,
    method: str = "ewc",  # "ewc", "mas", "prog_nn"
    optimizer: Optional[optim.Optimizer] = None,
    **kwargs
) -> ContinualLearningFramework:
    """
    Factory function for creating continual learning framework
    
    Args:
        model: Base model
        method: Continual learning method
        optimizer: Optimizer (creates Adam if None)
        **kwargs: Additional parameters
        
    Returns:
        ContinualLearningFramework instance
    """
    # Create regularization method
    if method == "ewc":
        reg_method = ElasticWeightConsolidation(
            importance_coeff=kwargs.get('importance_coeff', 1000.0)
        )
    elif method == "mas":
        reg_method = MemoryAwareSynapses(
            importance_coeff=kwargs.get('importance_coeff', 1.0)
        )
    else:
        reg_method = ElasticWeightConsolidation()
    
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=kwargs.get('lr', 1e-3))
    
    # Create framework
    framework = ContinualLearningFramework(
        model=model,
        regularization_method=reg_method,
        optimizer=optimizer,
        experience_replay=kwargs.get('experience_replay', True),
        replay_buffer_size=kwargs.get('replay_buffer_size', 10000),
        progressive_networks=(method == "prog_nn"),
        num_expected_tasks=kwargs.get('num_expected_tasks', 10)
    )
    
    return framework


if __name__ == "__main__":
    # Example usage
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=784, hidden_dim=256, num_classes=10):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_classes)
            )
        
        def forward(self, x):
            return self.network(x.view(x.size(0), -1))
    
    # Create model and framework
    model = SimpleModel()
    
    cl_framework = create_continual_learning_framework(
        model=model,
        method="ewc",
        lr=1e-3,
        importance_coeff=1000.0,
        experience_replay=True,
        replay_buffer_size=5000
    )
    
    # Create dummy data loaders for multiple tasks
    class DummyDataLoader:
        def __init__(self, task_id, size=1000):
            self.dataset = [(torch.randn(28, 28), torch.randint(0, 5, (1,)).item()) 
                           for _ in range(size)]
            self.task_id = task_id
        
        def __iter__(self):
            for i in range(0, len(self.dataset), 32):  # Batch size 32
                batch = self.dataset[i:i+32]
                data = torch.stack([item[0] for item in batch])
                targets = torch.tensor([item[1] for item in batch])
                yield data, targets
        
        def __len__(self):
            return len(self.dataset) // 32  # Number of batches
    
    # Register tasks
    for task_id in range(3):
        data_loader = DummyDataLoader(task_id)
        criterion = nn.CrossEntropyLoss()
        cl_framework.register_task(task_id, data_loader, criterion)
    
    # Train on tasks sequentially
    print("Starting continual learning training...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for task_id in range(3):
        print(f"\nTraining on task {task_id}...")
        cl_framework.train_task(
            task_id=task_id,
            epochs=2,
            device=device,
            ewc_lambda=100.0
        )
        
        # Evaluate on current task
        eval_metrics = cl_framework.evaluate_task(
            task_id, 
            cl_framework.task_data_loaders[task_id],
            device
        )
        print(f"Task {task_id} evaluation: {eval_metrics}")
    
    print("\nContinual learning training completed!")