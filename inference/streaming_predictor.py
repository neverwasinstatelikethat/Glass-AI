"""
Streaming Predictor for Real-Time Inference
Handles continuous data streams with low-latency predictions and adaptive buffering
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any, Iterator
import logging
import time
import threading
import queue
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingBuffer:
    """
    Circular buffer for streaming data with configurable size and overlap
    """
    
    def __init__(
        self, 
        buffer_size: int = 1000,
        overlap_size: int = 100,
        drop_oldest: bool = True
    ):
        """
        Args:
            buffer_size: Maximum number of items in buffer
            overlap_size: Number of items to overlap between windows
            drop_oldest: Whether to drop oldest items when buffer is full
        """
        self.buffer_size = buffer_size
        self.overlap_size = overlap_size
        self.drop_oldest = drop_oldest
        
        # Circular buffer
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        
        # Buffer statistics
        self.total_items_added = 0
        self.total_items_dropped = 0
    
    def add_item(self, item: Any) -> bool:
        """
        Add item to buffer
        
        Args:
            item: Item to add
            
        Returns:
            True if item was added, False if buffer was full and drop_oldest is False
        """
        with self.lock:
            if len(self.buffer) >= self.buffer_size:
                if not self.drop_oldest:
                    return False
                # Remove oldest items to make space for overlap
                for _ in range(self.overlap_size):
                    if self.buffer:
                        self.buffer.popleft()
                        self.total_items_dropped += 1
            
            self.buffer.append(item)
            self.total_items_added += 1
            return True
    
    def get_window(self, window_size: int) -> List[Any]:
        """
        Get sliding window of items
        
        Args:
            window_size: Size of window to retrieve
            
        Returns:
            List of items in window
        """
        with self.lock:
            if len(self.buffer) < window_size:
                return list(self.buffer)
            return list(self.buffer)[-window_size:]
    
    def get_overlap_window(self) -> List[Any]:
        """
        Get overlap window for continuity
        
        Returns:
            List of overlapping items
        """
        with self.lock:
            if len(self.buffer) < self.overlap_size:
                return list(self.buffer)
            return list(self.buffer)[-self.overlap_size:]
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def stats(self) -> Dict[str, int]:
        """Get buffer statistics"""
        with self.lock:
            return {
                'current_size': len(self.buffer),
                'total_added': self.total_items_added,
                'total_dropped': self.total_items_dropped,
                'buffer_capacity': self.buffer_size
            }


class StreamingPredictor:
    """
    Streaming predictor for real-time inference on continuous data streams
    """
    
    def __init__(
        self,
        model: nn.Module,
        window_size: int = 100,
        stride: int = 50,
        buffer_size: int = 1000,
        overlap_size: int = 10,
        device: torch.device = torch.device('cpu'),
        prediction_callback: Optional[Callable] = None,
        batch_processing: bool = True,
        batch_size: int = 32
    ):
        """
        Args:
            model: Model for inference
            window_size: Size of input windows for prediction
            stride: Stride between consecutive windows
            buffer_size: Size of circular buffer
            overlap_size: Overlap between windows for continuity
            device: Computing device
            prediction_callback: Callback function for predictions
            batch_processing: Whether to process in batches
            batch_size: Batch size for batch processing
        """
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.buffer_size = buffer_size
        self.overlap_size = overlap_size
        self.device = device
        self.prediction_callback = prediction_callback
        self.batch_processing = batch_processing
        self.batch_size = batch_size
        
        # Initialize buffer
        self.buffer = StreamingBuffer(buffer_size, overlap_size)
        
        # Model setup
        self.model.to(device)
        self.model.eval()
        
        # Processing state
        self.last_processed_index = 0
        self.is_running = False
        self.processing_thread = None
        
        # Performance metrics
        self.prediction_times = deque(maxlen=1000)
        self.throughput_rates = deque(maxlen=1000)
        
        logger.info(f"Initialized StreamingPredictor with window_size={window_size}, "
                   f"stride={stride}, buffer_size={buffer_size}")
    
    def add_data(self, data: Any) -> bool:
        """
        Add data to streaming buffer
        
        Args:
            data: Data item to add
            
        Returns:
            True if data was added, False otherwise
        """
        success = self.buffer.add_item(data)
        if not success:
            logger.warning("Failed to add data to buffer (buffer full)")
        return success
    
    def process_stream(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Process data stream and generate predictions
        
        Args:
            timeout: Timeout for processing (None for no timeout)
            
        Returns:
            List of prediction results
        """
        predictions = []
        start_time = time.time()
        
        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                break
            
            # Get current buffer window
            current_window = self.buffer.get_window(self.window_size)
            
            # Check if we have enough data and haven't processed this window
            if len(current_window) >= self.window_size:
                current_buffer_size = self.buffer.size()
                if current_buffer_size > self.last_processed_index + self.stride:
                    # Process window
                    window_predictions = self._process_window(current_window)
                    
                    # Update last processed index
                    self.last_processed_index = current_buffer_size
                    
                    # Add predictions
                    predictions.extend(window_predictions)
                    
                    # Call callback if provided
                    if self.prediction_callback:
                        for pred in window_predictions:
                            self.prediction_callback(pred)
                
                # Small delay to prevent busy waiting
                time.sleep(0.001)
            else:
                # Not enough data, small delay
                time.sleep(0.01)
        
        return predictions
    
    def start_processing_thread(self):
        """
        Start background processing thread
        """
        if self.is_running:
            logger.warning("Processing thread already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Started streaming processing thread")
    
    def stop_processing_thread(self):
        """
        Stop background processing thread
        """
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        logger.info("Stopped streaming processing thread")
    
    def _processing_loop(self):
        """
        Background processing loop
        """
        while self.is_running:
            try:
                # Get current window
                current_window = self.buffer.get_window(self.window_size)
                
                if len(current_window) >= self.window_size:
                    current_buffer_size = self.buffer.size()
                    if current_buffer_size > self.last_processed_index + self.stride:
                        # Process window
                        start_time = time.time()
                        window_predictions = self._process_window(current_window)
                        processing_time = time.time() - start_time
                        
                        # Update metrics
                        self.prediction_times.append(processing_time)
                        if processing_time > 0:
                            self.throughput_rates.append(len(window_predictions) / processing_time)
                        
                        # Update last processed index
                        self.last_processed_index = current_buffer_size
                        
                        # Call callback for each prediction
                        if self.prediction_callback:
                            for pred in window_predictions:
                                try:
                                    self.prediction_callback(pred)
                                except Exception as e:
                                    logger.error(f"Error in prediction callback: {e}")
                
                # Adaptive sleep based on processing speed
                if len(self.prediction_times) > 0:
                    avg_time = np.mean(list(self.prediction_times))
                    sleep_time = max(0.001, avg_time / 10)
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _process_window(self, window_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Process a window of data
        
        Args:
            window_data: List of data items in window
            
        Returns:
            List of prediction dictionaries
        """
        start_time = time.time()
        
        try:
            if self.batch_processing and len(window_data) > self.batch_size:
                # Process in batches
                predictions = []
                for i in range(0, len(window_data), self.batch_size):
                    batch = window_data[i:i + self.batch_size]
                    batch_predictions = self._process_batch(batch)
                    predictions.extend(batch_predictions)
            else:
                # Process single batch
                predictions = self._process_batch(window_data)
            
            # Add timing information
            processing_time = time.time() - start_time
            for pred in predictions:
                pred['processing_time'] = processing_time / len(predictions)
                pred['timestamp'] = time.time()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error processing window: {e}")
            return []
    
    def _process_batch(self, batch_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Process a batch of data
        
        Args:
            batch_data: List of data items in batch
            
        Returns:
            List of prediction dictionaries
        """
        if not batch_data:
            return []
        
        try:
            with torch.no_grad():
                # Convert to tensor (assuming data is compatible)
                if isinstance(batch_data[0], (torch.Tensor, np.ndarray)):
                    batch_tensor = torch.stack([
                        torch.tensor(item) if isinstance(item, np.ndarray) else item 
                        for item in batch_data
                    ])
                else:
                    # Assume data needs preprocessing
                    batch_tensor = torch.tensor(batch_data)
                
                # Move to device
                batch_tensor = batch_tensor.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_tensor)
                
                # Process outputs
                predictions = self._process_outputs(outputs)
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return []
    
    def _process_outputs(self, outputs: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Process model outputs into prediction dictionaries
        
        Args:
            outputs: Model outputs tensor
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        # Handle different output formats
        if isinstance(outputs, dict):
            logits = outputs.get('prediction', outputs.get('logits', next(iter(outputs.values()))))
        else:
            logits = outputs
        
        # Convert to predictions
        if len(logits.shape) == 2 and logits.shape[1] > 1:  # Multi-class
            predictions_tensor = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)
        elif len(logits.shape) == 2:  # Binary
            predictions_tensor = (torch.sigmoid(logits) > 0.5).float()
            probabilities = torch.sigmoid(logits)
        else:  # Regression or other
            predictions_tensor = logits
            probabilities = None
        
        # Convert to list of dictionaries
        for i in range(len(predictions_tensor)):
            pred_dict = {
                'prediction': predictions_tensor[i].item(),
                'index': self.last_processed_index + i
            }
            
            if probabilities is not None:
                if len(probabilities.shape) == 2 and probabilities.shape[1] > 1:
                    pred_dict['probabilities'] = probabilities[i].cpu().numpy()
                else:
                    pred_dict['probability'] = probabilities[i].item()
            
            predictions.append(pred_dict)
        
        return predictions
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics
        
        Returns:
            Dictionary with buffer statistics
        """
        return self.buffer.stats()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        with threading.Lock():
            return {
                'avg_prediction_time': np.mean(list(self.prediction_times)) if self.prediction_times else 0.0,
                'avg_throughput': np.mean(list(self.throughput_rates)) if self.throughput_rates else 0.0,
                'prediction_time_std': np.std(list(self.prediction_times)) if self.prediction_times else 0.0,
                'throughput_std': np.std(list(self.throughput_rates)) if self.throughput_rates else 0.0
            }
    
    def reset(self):
        """
        Reset predictor state
        """
        self.buffer.clear()
        self.last_processed_index = 0
        self.prediction_times.clear()
        self.throughput_rates.clear()
        logger.info("Reset streaming predictor")


class AsyncStreamingPredictor:
    """
    Asynchronous streaming predictor using asyncio
    """
    
    def __init__(
        self,
        model: nn.Module,
        window_size: int = 100,
        stride: int = 50,
        buffer_size: int = 1000,
        device: torch.device = torch.device('cpu'),
        prediction_callback: Optional[Callable] = None
    ):
        """
        Args:
            model: Model for inference
            window_size: Size of input windows
            stride: Stride between windows
            buffer_size: Size of circular buffer
            device: Computing device
            prediction_callback: Async callback function for predictions
        """
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.buffer_size = buffer_size
        self.device = device
        self.prediction_callback = prediction_callback
        
        # Initialize buffer
        self.buffer = StreamingBuffer(buffer_size, 0)  # No overlap for async
        
        # Model setup
        self.model.to(device)
        self.model.eval()
        
        # Async state
        self.is_running = False
        self.processing_task = None
        
        logger.info(f"Initialized AsyncStreamingPredictor with window_size={window_size}")
    
    async def add_data_async(self, data: Any):
        """
        Add data asynchronously
        
        Args:
            data: Data item to add
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.buffer.add_item, data)
    
    async def start_processing(self):
        """
        Start async processing
        """
        if self.is_running:
            logger.warning("Processing already running")
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Started async streaming processing")
    
    async def stop_processing(self):
        """
        Stop async processing
        """
        self.is_running = False
        if self.processing_task:
            await self.processing_task
        logger.info("Stopped async streaming processing")
    
    async def _processing_loop(self):
        """
        Async processing loop
        """
        last_processed = 0
        
        while self.is_running:
            try:
                # Get current window
                current_window = self.buffer.get_window(self.window_size)
                
                if len(current_window) >= self.window_size:
                    current_size = self.buffer.size()
                    if current_size > last_processed + self.stride:
                        # Process window
                        predictions = await self._process_window_async(current_window)
                        
                        # Update last processed
                        last_processed = current_size
                        
                        # Call callback
                        if self.prediction_callback:
                            for pred in predictions:
                                await self.prediction_callback(pred)
                
                # Small delay
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in async processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_window_async(self, window_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Process window asynchronously
        
        Args:
            window_data: Window data
            
        Returns:
            List of predictions
        """
        loop = asyncio.get_event_loop()
        
        # Run processing in thread pool
        predictions = await loop.run_in_executor(
            None, self._process_window_sync, window_data
        )
        
        return predictions
    
    def _process_window_sync(self, window_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Synchronous window processing (runs in thread pool)
        
        Args:
            window_data: Window data
            
        Returns:
            List of predictions
        """
        try:
            with torch.no_grad():
                # Convert to tensor
                if isinstance(window_data[0], (torch.Tensor, np.ndarray)):
                    batch_tensor = torch.stack([
                        torch.tensor(item) if isinstance(item, np.ndarray) else item 
                        for item in window_data
                    ])
                else:
                    batch_tensor = torch.tensor(window_data)
                
                # Move to device
                batch_tensor = batch_tensor.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_tensor)
                
                # Process outputs
                predictions = []
                if isinstance(outputs, dict):
                    logits = outputs.get('prediction', next(iter(outputs.values())))
                else:
                    logits = outputs
                
                if len(logits.shape) == 2 and logits.shape[1] > 1:
                    predictions_tensor = torch.argmax(logits, dim=1)
                else:
                    predictions_tensor = (torch.sigmoid(logits) > 0.5).float()
                
                for i in range(len(predictions_tensor)):
                    predictions.append({
                        'prediction': predictions_tensor[i].item(),
                        'timestamp': time.time()
                    })
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error in sync window processing: {e}")
            return []


class AdaptiveStreamingPredictor(StreamingPredictor):
    """
    Adaptive streaming predictor that adjusts window size and stride based on load
    """
    
    def __init__(
        self,
        model: nn.Module,
        initial_window_size: int = 100,
        initial_stride: int = 50,
        min_window_size: int = 10,
        max_window_size: int = 1000,
        target_latency: float = 0.1,  # seconds
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        """
        Args:
            model: Model for inference
            initial_window_size: Initial window size
            initial_stride: Initial stride
            min_window_size: Minimum window size
            max_window_size: Maximum window size
            target_latency: Target processing latency in seconds
            adaptation_rate: Rate of adaptation (0.0 to 1.0)
            **kwargs: Additional arguments for StreamingPredictor
        """
        super().__init__(
            model=model,
            window_size=initial_window_size,
            stride=initial_stride,
            **kwargs
        )
        
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.target_latency = target_latency
        self.adaptation_rate = adaptation_rate
        
        # Adaptive parameters
        self.current_window_size = initial_window_size
        self.current_stride = initial_stride
        
        logger.info(f"Initialized AdaptiveStreamingPredictor with target latency={target_latency}s")
    
    def _process_window(self, window_data: List[Any]) -> List[Dict[str, Any]]:
        """
        Process window with adaptive sizing
        """
        start_time = time.time()
        predictions = super()._process_window(window_data)
        processing_time = time.time() - start_time
        
        # Adapt window size based on processing time
        self._adapt_window_size(processing_time)
        
        return predictions
    
    def _adapt_window_size(self, processing_time: float):
        """
        Adapt window size based on processing time
        
        Args:
            processing_time: Time taken to process last window
        """
        # Calculate error from target latency
        error = processing_time - self.target_latency
        
        # Adjust window size proportionally to error
        if abs(error) > 0.01:  # Only adapt if error is significant
            # Scale factor based on error and adaptation rate
            scale_factor = 1.0 - (error / self.target_latency) * self.adaptation_rate
            
            # Apply bounds
            scale_factor = max(0.5, min(2.0, scale_factor))
            
            # Update window size
            new_window_size = int(self.current_window_size * scale_factor)
            new_window_size = max(self.min_window_size, min(self.max_window_size, new_window_size))
            
            # Update stride proportionally
            stride_ratio = self.current_stride / self.current_window_size
            new_stride = int(new_window_size * stride_ratio)
            new_stride = max(1, min(new_window_size // 2, new_stride))
            
            # Apply changes
            if new_window_size != self.current_window_size:
                self.current_window_size = new_window_size
                self.current_stride = new_stride
                self.window_size = new_window_size
                self.stride = new_stride
                
                logger.debug(f"Adapted window size: {new_window_size}, stride: {new_stride}")


def create_streaming_predictor(
    model: nn.Module,
    predictor_type: str = "standard",  # "standard", "async", "adaptive"
    **kwargs
) -> Any:
    """
    Factory function for creating streaming predictors
    
    Args:
        model: Model for inference
        predictor_type: Type of predictor
        **kwargs: Additional parameters
        
    Returns:
        Streaming predictor instance
    """
    if predictor_type == "standard":
        return StreamingPredictor(model, **kwargs)
    elif predictor_type == "async":
        return AsyncStreamingPredictor(model, **kwargs)
    elif predictor_type == "adaptive":
        return AdaptiveStreamingPredictor(model, **kwargs)
    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}")


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
    
    # Create model
    model = SimpleModel()
    
    print("Testing Standard StreamingPredictor...")
    # Standard streaming predictor
    streaming_predictor = create_streaming_predictor(
        model,
        predictor_type="standard",
        window_size=50,
        stride=25,
        buffer_size=500
    )
    
    # Define callback
    def prediction_callback(prediction):
        print(f"Prediction received: {prediction['prediction']:.2f} at index {prediction['index']}")
    
    streaming_predictor.prediction_callback = prediction_callback
    
    # Add some data
    print("Adding data to stream...")
    for i in range(200):
        data_point = torch.randn(10)  # 10-dimensional input
        streaming_predictor.add_data(data_point)
        time.sleep(0.01)  # Simulate data arrival
    
    # Process stream for a bit
    print("Processing stream...")
    predictions = streaming_predictor.process_stream(timeout=2.0)
    print(f"Generated {len(predictions)} predictions")
    
    # Show buffer stats
    buffer_stats = streaming_predictor.get_buffer_stats()
    print(f"Buffer stats: {buffer_stats}")
    
    # Show performance metrics
    perf_metrics = streaming_predictor.get_performance_metrics()
    print(f"Performance metrics: {perf_metrics}")
    
    print("\nTesting Adaptive StreamingPredictor...")
    # Adaptive streaming predictor
    adaptive_predictor = create_streaming_predictor(
        model,
        predictor_type="adaptive",
        initial_window_size=30,
        initial_stride=15,
        target_latency=0.05
    )
    
    # Add data to adaptive predictor
    print("Adding data to adaptive stream...")
    for i in range(100):
        data_point = torch.randn(10)
        adaptive_predictor.add_data(data_point)
        time.sleep(0.005)  # Faster data arrival
    
    # Process for a bit
    predictions = adaptive_predictor.process_stream(timeout=1.0)
    print(f"Adaptive predictor generated {len(predictions)} predictions")
    print(f"Final window size: {adaptive_predictor.current_window_size}")