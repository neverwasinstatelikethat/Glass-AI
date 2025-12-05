"""
Batch Predictor for Large-Scale Inference
Handles batch processing of data with optimized memory usage and parallel processing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Iterator, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchDataset(Dataset):
    """
    Dataset wrapper for batch inference
    """
    
    def __init__(self, data: List[Any], transform: Optional[Callable] = None):
        """
        Args:
            data: List of data samples
            transform: Optional transform function
        """
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class BatchPredictor:
    """
    Batch predictor for efficient large-scale inference
    """
    
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        num_workers: int = 4,
        device: torch.device = torch.device('cpu'),
        pin_memory: bool = False,
        prefetch_factor: int = 2
    ):
        """
        Args:
            model: Model for inference
            batch_size: Batch size for processing
            num_workers: Number of worker processes for data loading
            device: Computing device
            pin_memory: Whether to use pinned memory
            prefetch_factor: Number of batches to prefetch per worker
        """
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        
        # Move model to device and set to evaluation mode
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Initialized BatchPredictor with batch_size={batch_size}, "
                   f"workers={num_workers}, device={device}")
    
    def predict(
        self,
        data: List[Any],
        transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        return_probabilities: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform batch prediction on data
        
        Args:
            data: List of data samples
            transform: Transform function for preprocessing
            collate_fn: Custom collate function
            return_probabilities: Whether to return class probabilities
            progress_callback: Callback function for progress updates
            
        Returns:
            List of prediction dictionaries
        """
        # Create dataset
        dataset = BatchDataset(data, transform)
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn
        )
        
        # Process batches
        predictions = []
        total_batches = len(dataloader)
        
        logger.info(f"Processing {len(data)} samples in {total_batches} batches")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [item.to(self.device) if hasattr(item, 'to') else item for item in batch]
                elif hasattr(batch, 'to'):
                    batch = batch.to(self.device)
                
                # Forward pass
                start_time = time.time()
                outputs = self.model(batch)
                inference_time = time.time() - start_time
                
                # Process outputs
                batch_predictions = self._process_outputs(
                    outputs, return_probabilities
                )
                
                # Add metadata
                for i, pred in enumerate(batch_predictions):
                    pred.update({
                        'batch_idx': batch_idx,
                        'sample_idx': batch_idx * self.batch_size + i,
                        'inference_time': inference_time / len(batch_predictions)
                    })
                
                predictions.extend(batch_predictions)
                
                # Progress callback
                if progress_callback:
                    progress_callback(batch_idx + 1, total_batches)
                
                # Clear cache periodically
                if batch_idx % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        
        logger.info(f"Completed batch prediction. Processed {len(predictions)} samples")
        return predictions
    
    def _process_outputs(
        self, 
        outputs: torch.Tensor, 
        return_probabilities: bool
    ) -> List[Dict[str, Any]]:
        """
        Process model outputs into prediction dictionaries
        
        Args:
            outputs: Model outputs tensor
            return_probabilities: Whether to include probabilities
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        # Handle different output formats
        if isinstance(outputs, dict):
            # Dictionary output (e.g., from models with multiple heads)
            logits = outputs.get('prediction', outputs.get('logits', next(iter(outputs.values()))))
        else:
            # Tensor output
            logits = outputs
        
        # Convert to probabilities if needed
        if return_probabilities:
            if len(logits.shape) == 2:  # Classification logits
                probabilities = torch.softmax(logits, dim=1)
            else:
                probabilities = torch.sigmoid(logits)
        else:
            probabilities = None
        
        # Get predictions
        if len(logits.shape) == 2 and logits.shape[1] > 1:  # Multi-class
            predictions_tensor = torch.argmax(logits, dim=1)
        else:  # Binary or regression
            predictions_tensor = (torch.sigmoid(logits) > 0.5).float() if len(logits.shape) == 2 else logits
        
        # Convert to list of dictionaries
        for i in range(len(predictions_tensor)):
            pred_dict = {
                'prediction': predictions_tensor[i].item()
            }
            
            if return_probabilities and probabilities is not None:
                if len(probabilities.shape) == 2:
                    pred_dict['probabilities'] = probabilities[i].cpu().numpy()
                else:
                    pred_dict['probability'] = probabilities[i].item()
            
            predictions.append(pred_dict)
        
        return predictions
    
    def predict_generator(
        self,
        data_generator: Iterator[List[Any]],
        transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        return_probabilities: bool = False
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Generator-based prediction for streaming data
        
        Args:
            data_generator: Generator yielding batches of data
            transform: Transform function
            collate_fn: Collate function
            return_probabilities: Whether to return probabilities
            
        Yields:
            Lists of prediction dictionaries
        """
        for batch_data in data_generator:
            # Create temporary dataset and dataloader
            dataset = BatchDataset(batch_data, transform)
            dataloader = DataLoader(
                dataset,
                batch_size=min(self.batch_size, len(batch_data)),
                shuffle=False,
                num_workers=0,  # No workers for generator mode
                collate_fn=collate_fn
            )
            
            batch_predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    # Move to device
                    if isinstance(batch, (list, tuple)):
                        batch = [item.to(self.device) if hasattr(item, 'to') else item for item in batch]
                    elif hasattr(batch, 'to'):
                        batch = batch.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch)
                    
                    # Process outputs
                    predictions = self._process_outputs(outputs, return_probabilities)
                    batch_predictions.extend(predictions)
            
            yield batch_predictions


class ParallelBatchPredictor:
    """
    Parallel batch predictor using multiple processes
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        batch_size: int = 32,
        num_processes: int = None,
        device_ids: List[str] = None
    ):
        """
        Args:
            model_factory: Function that creates and returns a model
            batch_size: Batch size per process
            num_processes: Number of processes (defaults to CPU count)
            device_ids: List of device IDs (e.g., ['cuda:0', 'cuda:1'])
        """
        self.model_factory = model_factory
        self.batch_size = batch_size
        self.num_processes = num_processes or mp.cpu_count()
        self.device_ids = device_ids or ['cpu'] * self.num_processes
        
        logger.info(f"Initialized ParallelBatchPredictor with {self.num_processes} processes")
    
    def predict(
        self,
        data: List[Any],
        transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        return_probabilities: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform parallel batch prediction
        
        Args:
            data: List of data samples
            transform: Transform function
            collate_fn: Collate function
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of prediction dictionaries
        """
        # Split data into chunks
        chunk_size = max(1, len(data) // self.num_processes)
        data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        # Create process arguments
        process_args = [
            (chunk, i, transform, collate_fn, return_probabilities)
            for i, chunk in enumerate(data_chunks)
        ]
        
        # Process in parallel
        predictions = []
        with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
            future_to_chunk = {
                executor.submit(self._process_chunk, args): i 
                for i, args in enumerate(process_args)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_predictions = future.result()
                predictions.extend(chunk_predictions)
        
        # Sort by sample index to maintain order
        predictions.sort(key=lambda x: x.get('sample_idx', 0))
        
        logger.info(f"Completed parallel batch prediction on {len(data)} samples")
        return predictions
    
    def _process_chunk(self, args: Tuple) -> List[Dict[str, Any]]:
        """
        Process a chunk of data in a separate process
        
        Args:
            args: Tuple of (data_chunk, chunk_id, transform, collate_fn, return_probabilities)
            
        Returns:
            List of predictions for the chunk
        """
        data_chunk, chunk_id, transform, collate_fn, return_probabilities = args
        
        # Create model and predictor for this process
        model = self.model_factory()
        device_id = self.device_ids[chunk_id % len(self.device_ids)]
        device = torch.device(device_id)
        
        predictor = BatchPredictor(
            model=model,
            batch_size=self.batch_size,
            device=device
        )
        
        # Process chunk
        chunk_predictions = predictor.predict(
            data=data_chunk,
            transform=transform,
            collate_fn=collate_fn,
            return_probabilities=return_probabilities
        )
        
        return chunk_predictions


class MemoryEfficientPredictor:
    """
    Memory-efficient predictor with gradient checkpointing and mixed precision
    """
    
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 32,
        device: torch.device = torch.device('cpu'),
        use_amp: bool = False,
        max_memory_mb: int = 1024
    ):
        """
        Args:
            model: Model for inference
            batch_size: Batch size
            device: Computing device
            use_amp: Whether to use automatic mixed precision
            max_memory_mb: Maximum memory usage in MB
        """
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.max_memory_mb = max_memory_mb
        
        # Setup AMP scaler if needed
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Initialized MemoryEfficientPredictor with AMP={use_amp}")
    
    def predict(
        self,
        data: List[Any],
        transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        return_probabilities: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Memory-efficient prediction with automatic batch size adjustment
        
        Args:
            data: List of data samples
            transform: Transform function
            collate_fn: Collate function
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of prediction dictionaries
        """
        # Start with requested batch size
        current_batch_size = self.batch_size
        predictions = []
        
        # Process in chunks to manage memory
        for i in range(0, len(data), current_batch_size):
            chunk_data = data[i:i + current_batch_size]
            
            try:
                # Create dataset and dataloader
                dataset = BatchDataset(chunk_data, transform)
                dataloader = DataLoader(
                    dataset,
                    batch_size=len(chunk_data),  # Process whole chunk at once
                    shuffle=False,
                    collate_fn=collate_fn
                )
                
                # Process chunk
                chunk_predictions = self._predict_chunk(
                    dataloader, return_probabilities
                )
                
                # Add offset to sample indices
                for j, pred in enumerate(chunk_predictions):
                    pred['sample_idx'] = i + j
                
                predictions.extend(chunk_predictions)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and current_batch_size > 1:
                    # Reduce batch size and retry
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(f"OOM error, reducing batch size to {current_batch_size}")
                    # Retry with smaller batch size
                    i -= current_batch_size  # Go back to reprocess with smaller batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        return predictions
    
    def _predict_chunk(
        self,
        dataloader: DataLoader,
        return_probabilities: bool
    ) -> List[Dict[str, Any]]:
        """
        Predict on a single data chunk
        
        Args:
            dataloader: DataLoader for the chunk
            return_probabilities: Whether to return probabilities
            
        Returns:
            List of predictions
        """
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                if isinstance(batch, (list, tuple)):
                    batch = [item.to(self.device) if hasattr(item, 'to') else item for item in batch]
                elif hasattr(batch, 'to'):
                    batch = batch.to(self.device)
                
                # Forward pass with AMP if enabled
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)
                
                # Process outputs
                batch_predictions = self._process_outputs(outputs, return_probabilities)
                predictions.extend(batch_predictions)
        
        return predictions
    
    def _process_outputs(
        self, 
        outputs: torch.Tensor, 
        return_probabilities: bool
    ) -> List[Dict[str, Any]]:
        """
        Process model outputs (same as BatchPredictor)
        """
        # This is identical to BatchPredictor._process_outputs
        # For brevity, we'll just call the parent method in practice
        # Here's a simplified version:
        
        if isinstance(outputs, dict):
            logits = outputs.get('prediction', outputs.get('logits', next(iter(outputs.values()))))
        else:
            logits = outputs
        
        if return_probabilities:
            if len(logits.shape) == 2:
                probabilities = torch.softmax(logits, dim=1)
            else:
                probabilities = torch.sigmoid(logits)
        else:
            probabilities = None
        
        if len(logits.shape) == 2 and logits.shape[1] > 1:
            predictions_tensor = torch.argmax(logits, dim=1)
        else:
            predictions_tensor = (torch.sigmoid(logits) > 0.5).float() if len(logits.shape) == 2 else logits
        
        predictions = []
        for i in range(len(predictions_tensor)):
            pred_dict = {'prediction': predictions_tensor[i].item()}
            if return_probabilities and probabilities is not None:
                if len(probabilities.shape) == 2:
                    pred_dict['probabilities'] = probabilities[i].cpu().numpy()
                else:
                    pred_dict['probability'] = probabilities[i].item()
            predictions.append(pred_dict)
        
        return predictions


def create_batch_predictor(
    model: nn.Module,
    predictor_type: str = "standard",  # "standard", "parallel", "memory_efficient"
    **kwargs
) -> Any:
    """
    Factory function for creating batch predictors
    
    Args:
        model: Model for inference
        predictor_type: Type of predictor
        **kwargs: Additional parameters
        
    Returns:
        Batch predictor instance
    """
    if predictor_type == "standard":
        return BatchPredictor(model, **kwargs)
    elif predictor_type == "parallel":
        return ParallelBatchPredictor(lambda: model, **kwargs)
    elif predictor_type == "memory_efficient":
        return MemoryEfficientPredictor(model, **kwargs)
    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}")


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
    
    # Create model
    model = SimpleModel()
    
    # Create sample data
    sample_data = [torch.randn(28, 28) for _ in range(1000)]
    
    print("Testing Standard BatchPredictor...")
    # Standard batch predictor
    standard_predictor = create_batch_predictor(
        model,
        predictor_type="standard",
        batch_size=32,
        device=torch.device('cpu')
    )
    
    # Define progress callback
    def progress_callback(current, total):
        if current % 10 == 0 or current == total:
            print(f"Progress: {current}/{total} batches")
    
    # Predict
    predictions = standard_predictor.predict(
        data=sample_data[:100],  # Smaller sample for demo
        progress_callback=progress_callback,
        return_probabilities=True
    )
    
    print(f"Standard predictor results: {len(predictions)} predictions")
    print(f"Sample prediction: {predictions[0]}")
    
    print("\nTesting MemoryEfficientPredictor...")
    # Memory efficient predictor
    mem_predictor = create_batch_predictor(
        model,
        predictor_type="memory_efficient",
        batch_size=64,
        use_amp=False
    )
    
    mem_predictions = mem_predictor.predict(
        data=sample_data[:100],
        return_probabilities=True
    )
    
    print(f"Memory efficient predictor results: {len(mem_predictions)} predictions")
    
    # Compare results (should be similar)
    diff = abs(predictions[0]['prediction'] - mem_predictions[0]['prediction'])
    print(f"Difference between predictors: {diff}")