"""
Real-time Synchronization for Digital Twin Visualization
Synchronizes 3D visualization with real-time sensor data and simulation updates
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    """Real-time sensor data structure"""
    timestamp: float
    sensor_id: str
    value: float
    unit: str
    position: Optional[Tuple[float, float, float]] = None
    quality: float = 1.0  # Data quality 0.0 to 1.0


@dataclass
class VisualizationUpdate:
    """Visualization update package"""
    timestamp: float
    temperature_field: Optional[np.ndarray] = None
    defects: Optional[Dict[str, float]] = None
    furnace_state: Optional[Dict[str, Any]] = None
    forming_state: Optional[Dict[str, Any]] = None
    simulation_time: Optional[float] = None


class RealTimeSynchronizer:
    """
    Real-time synchronizer for digital twin visualization
    Handles data streaming, synchronization, and update coordination
    """
    
    def __init__(
        self,
        update_rate: float = 30.0,  # FPS
        buffer_size: int = 1000,
        sync_tolerance: float = 0.1  # seconds
    ):
        """
        Args:
            update_rate: Target visualization update rate in FPS
            buffer_size: Size of data buffers
            sync_tolerance: Maximum time difference for synchronization
        """
        self.update_rate = update_rate
        self.sync_tolerance = sync_tolerance
        self.update_interval = 1.0 / update_rate
        
        # Data buffers
        self.sensor_buffer = deque(maxlen=buffer_size)
        self.simulation_buffer = deque(maxlen=buffer_size)
        self.visualization_buffer = deque(maxlen=buffer_size)
        
        # Time tracking
        self.last_update_time = 0.0
        self.simulation_time = 0.0
        self.real_time = time.time()
        
        # Callbacks
        self.visualization_callback: Optional[Callable] = None
        self.data_callbacks: Dict[str, Callable] = {}
        
        # Threading
        self.is_running = False
        self.sync_thread = None
        self.lock = threading.Lock()
        
        # Performance metrics
        self.update_times = deque(maxlen=100)
        self.frame_drops = 0
        
        logger.info(f"Initialized RealTimeSynchronizer: {update_rate} FPS")
    
    def register_data_callback(self, data_type: str, callback: Callable):
        """
        Register callback for specific data type
        
        Args:
            data_type: Type of data ('sensor', 'simulation', etc.)
            callback: Function to call when data arrives
        """
        self.data_callbacks[data_type] = callback
        logger.info(f"Registered callback for {data_type} data")
    
    def set_visualization_callback(self, callback: Callable):
        """
        Set callback for visualization updates
        
        Args:
            callback: Function to call for visualization updates
        """
        self.visualization_callback = callback
        logger.info("Set visualization callback")
    
    def add_sensor_data(self, sensor_data: SensorData):
        """
        Add sensor data to buffer
        
        Args:
            sensor_data: Sensor data to add
        """
        with self.lock:
            self.sensor_buffer.append(sensor_data)
        
        # Call data callback if registered
        if 'sensor' in self.data_callbacks:
            try:
                self.data_callbacks['sensor'](sensor_data)
            except Exception as e:
                logger.error(f"Error in sensor callback: {e}")
    
    def add_simulation_data(self, simulation_data: VisualizationUpdate):
        """
        Add simulation data to buffer
        
        Args:
            simulation_data: Simulation data to add
        """
        with self.lock:
            self.simulation_buffer.append(simulation_data)
            if simulation_data.simulation_time is not None:
                self.simulation_time = simulation_data.simulation_time
        
        # Call data callback if registered
        if 'simulation' in self.data_callbacks:
            try:
                self.data_callbacks['simulation'](simulation_data)
            except Exception as e:
                logger.error(f"Error in simulation callback: {e}")
    
    def start_synchronization(self):
        """
        Start real-time synchronization thread
        """
        if self.is_running:
            logger.warning("Synchronization already running")
            return
        
        self.is_running = True
        self.sync_thread = threading.Thread(target=self._synchronization_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        logger.info("Started real-time synchronization")
    
    def stop_synchronization(self):
        """
        Stop real-time synchronization
        """
        self.is_running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=2.0)
        logger.info("Stopped real-time synchronization")
    
    def _synchronization_loop(self):
        """
        Main synchronization loop running in separate thread
        """
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time for an update
                if current_time - self.last_update_time >= self.update_interval:
                    start_time = time.perf_counter()
                    
                    # Create synchronized update
                    update = self._create_synchronized_update()
                    
                    # Send to visualization
                    if update and self.visualization_callback:
                        try:
                            self.visualization_callback(update)
                        except Exception as e:
                            logger.error(f"Error in visualization callback: {e}")
                    
                    # Update timing
                    end_time = time.perf_counter()
                    update_duration = end_time - start_time
                    self.update_times.append(update_duration)
                    
                    self.last_update_time = current_time
                    
                    # Check for frame drops
                    if update_duration > self.update_interval:
                        self.frame_drops += 1
                        if self.frame_drops % 100 == 0:
                            logger.warning(f"Frame drops: {self.frame_drops}")
                
                # Small sleep to prevent busy waiting
                sleep_time = max(0.001, self.update_interval / 10)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in synchronization loop: {e}")
                time.sleep(0.1)
    
    def _create_synchronized_update(self) -> Optional[VisualizationUpdate]:
        """
        Create a synchronized visualization update from buffered data
        
        Returns:
            VisualizationUpdate or None if no data available
        """
        with self.lock:
            if not self.simulation_buffer:
                return None
            
            # Get latest simulation data
            latest_simulation = self.simulation_buffer[-1]
            
            # Synchronize with sensor data based on timestamps
            synchronized_sensors = self._synchronize_sensor_data(
                latest_simulation.timestamp
            )
            
            # Create update
            update = VisualizationUpdate(
                timestamp=latest_simulation.timestamp,
                temperature_field=latest_simulation.temperature_field,
                defects=latest_simulation.defects,
                furnace_state=latest_simulation.furnace_state,
                forming_state=latest_simulation.forming_state,
                simulation_time=latest_simulation.simulation_time
            )
            
            # Add to visualization buffer
            self.visualization_buffer.append(update)
            
            return update
    
    def _synchronize_sensor_data(self, target_time: float) -> List[SensorData]:
        """
        Synchronize sensor data to target time
        
        Args:
            target_time: Target timestamp for synchronization
            
        Returns:
            List of synchronized sensor data
        """
        synchronized = []
        
        # Find sensor data closest to target time
        best_matches = {}
        
        for sensor_data in self.sensor_buffer:
            time_diff = abs(sensor_data.timestamp - target_time)
            if time_diff <= self.sync_tolerance:
                sensor_id = sensor_data.sensor_id
                if (sensor_id not in best_matches or 
                    time_diff < abs(best_matches[sensor_id].timestamp - target_time)):
                    best_matches[sensor_id] = sensor_data
        
        return list(best_matches.values())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get synchronization performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        with self.lock:
            if self.update_times:
                avg_update_time = np.mean(self.update_times)
                std_update_time = np.std(self.update_times)
            else:
                avg_update_time = 0.0
                std_update_time = 0.0
            
            return {
                'update_rate': self.update_rate,
                'avg_update_time': avg_update_time,
                'std_update_time': std_update_time,
                'target_interval': self.update_interval,
                'frame_drops': self.frame_drops,
                'buffer_sizes': {
                    'sensor_buffer': len(self.sensor_buffer),
                    'simulation_buffer': len(self.simulation_buffer),
                    'visualization_buffer': len(self.visualization_buffer)
                },
                'current_simulation_time': self.simulation_time,
                'real_time': time.time()
            }
    
    def clear_buffers(self):
        """
        Clear all data buffers
        """
        with self.lock:
            self.sensor_buffer.clear()
            self.simulation_buffer.clear()
            self.visualization_buffer.clear()
            self.update_times.clear()
        logger.info("Cleared all data buffers")
    
    async def async_add_sensor_data(self, sensor_data: SensorData):
        """
        Asynchronously add sensor data (for async environments)
        
        Args:
            sensor_data: Sensor data to add
        """
        self.add_sensor_data(sensor_data)
    
    async def async_add_simulation_data(self, simulation_data: VisualizationUpdate):
        """
        Asynchronously add simulation data (for async environments)
        
        Args:
            simulation_data: Simulation data to add
        """
        self.add_simulation_data(simulation_data)


class WebSocketBroadcaster:
    """
    Broadcasts synchronized data to WebSocket clients for real-time visualization
    """
    
    def __init__(self, broadcast_interval: float = 0.033):  # ~30 FPS
        """
        Args:
            broadcast_interval: Interval between broadcasts in seconds
        """
        self.broadcast_interval = broadcast_interval
        self.clients = set()
        self.is_broadcasting = False
        self.broadcast_task = None
        
        # Data cache
        self.latest_update = None
        self.cache_lock = threading.Lock()
    
    def register_client(self, client):
        """
        Register a WebSocket client
        
        Args:
            client: WebSocket client connection
        """
        self.clients.add(client)
        logger.info(f"Registered WebSocket client. Total clients: {len(self.clients)}")
    
    def unregister_client(self, client):
        """
        Unregister a WebSocket client
        
        Args:
            client: WebSocket client connection
        """
        self.clients.discard(client)
        logger.info(f"Unregistered WebSocket client. Total clients: {len(self.clients)}")
    
    def update_data(self, visualization_update: VisualizationUpdate):
        """
        Update cached data for broadcasting
        
        Args:
            visualization_update: Latest visualization update
        """
        with self.cache_lock:
            self.latest_update = visualization_update
    
    async def start_broadcasting(self):
        """
        Start broadcasting data to clients
        """
        if self.is_broadcasting:
            logger.warning("Broadcasting already started")
            return
        
        self.is_broadcasting = True
        self.broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("Started WebSocket broadcasting")
    
    async def stop_broadcasting(self):
        """
        Stop broadcasting data to clients
        """
        self.is_broadcasting = False
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped WebSocket broadcasting")
    
    async def _broadcast_loop(self):
        """
        Main broadcasting loop
        """
        while self.is_broadcasting:
            try:
                # Get latest data
                with self.cache_lock:
                    if self.latest_update is None:
                        await asyncio.sleep(self.broadcast_interval)
                        continue
                    
                    # Convert to JSON-serializable format
                    data_dict = self._serialize_update(self.latest_update)
                    json_data = json.dumps(data_dict)
                
                # Broadcast to all clients
                disconnected_clients = set()
                for client in self.clients:
                    try:
                        await client.send(json_data)
                    except Exception as e:
                        logger.warning(f"Failed to send to client: {e}")
                        disconnected_clients.add(client)
                
                # Remove disconnected clients
                for client in disconnected_clients:
                    self.unregister_client(client)
                
                await asyncio.sleep(self.broadcast_interval)
                
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(0.1)
    
    def _serialize_update(self, update: VisualizationUpdate) -> Dict:
        """
        Serialize VisualizationUpdate to JSON-serializable dictionary
        
        Args:
            update: VisualizationUpdate to serialize
            
        Returns:
            Dictionary representation
        """
        result = {
            'timestamp': update.timestamp,
            'simulation_time': update.simulation_time,
        }
        
        # Handle numpy arrays
        if update.temperature_field is not None:
            # Downsample for transmission
            temp_field = update.temperature_field
            if len(temp_field.shape) >= 2:
                # Reduce resolution for transmission
                step = max(1, temp_field.shape[0] // 20)  # Max 20 points in each dimension
                downsampled = temp_field[::step, ::step]
                result['temperature_field'] = downsampled.tolist()
                result['temperature_shape'] = downsampled.shape
            else:
                result['temperature_field'] = temp_field.tolist()
                result['temperature_shape'] = temp_field.shape
        
        # Handle other fields
        if update.defects is not None:
            result['defects'] = update.defects
        
        if update.furnace_state is not None:
            result['furnace_state'] = update.furnace_state
            
        if update.forming_state is not None:
            result['forming_state'] = update.forming_state
        
        return result


def create_realtime_synchronizer(**kwargs) -> RealTimeSynchronizer:
    """
    Factory function to create a RealTimeSynchronizer instance
    
    Args:
        **kwargs: Parameters for RealTimeSynchronizer
        
    Returns:
        RealTimeSynchronizer instance
    """
    synchronizer = RealTimeSynchronizer(**kwargs)
    logger.info("Created RealTimeSynchronizer")
    return synchronizer


def create_websocket_broadcaster(**kwargs) -> WebSocketBroadcaster:
    """
    Factory function to create a WebSocketBroadcaster instance
    
    Args:
        **kwargs: Parameters for WebSocketBroadcaster
        
    Returns:
        WebSocketBroadcaster instance
    """
    broadcaster = WebSocketBroadcaster(**kwargs)
    logger.info("Created WebSocketBroadcaster")
    return broadcaster


if __name__ == "__main__":
    # Example usage
    print("Testing RealTimeSynchronizer...")
    
    # Create synchronizer
    synchronizer = create_realtime_synchronizer(update_rate=30.0)
    
    # Create broadcaster
    broadcaster = create_websocket_broadcaster()
    
    # Sample data callback
    def sample_viz_callback(update: VisualizationUpdate):
        print(f"Visualization update: t={update.timestamp:.3f}, "
              f"sim_time={update.simulation_time:.3f}")
    
    # Register callback
    synchronizer.set_visualization_callback(sample_viz_callback)
    
    # Start synchronization
    synchronizer.start_synchronization()
    
    # Generate sample data
    print("Generating sample data...")
    for i in range(100):
        # Add sensor data
        sensor_data = SensorData(
            timestamp=time.time(),
            sensor_id=f"temp_sensor_{i % 5}",
            value=1000 + np.random.normal(0, 50),
            unit="K",
            position=(np.random.random() * 50, 
                     np.random.random() * 5 - 2.5, 
                     np.random.random() * 3),
            quality=0.95
        )
        synchronizer.add_sensor_data(sensor_data)
        
        # Add simulation data every 10 iterations
        if i % 10 == 0:
            # Create sample temperature field
            temp_field = np.zeros((20, 10))
            for x in range(20):
                for y in range(10):
                    temp_field[x, y] = 1000 + 500 * np.exp(
                        -((x - 10)**2 + (y - 5)**2) / 20
                    )
            
            sim_data = VisualizationUpdate(
                timestamp=time.time(),
                temperature_field=temp_field,
                defects={
                    'crack': np.random.random() * 0.8,
                    'bubble': np.random.random() * 0.5,
                    'cloudiness': np.random.random() * 0.3
                },
                furnace_state={'temperature': 1500.0, 'melt_level': 0.8},
                forming_state={'belt_speed': 150.0, 'quality': 0.85},
                simulation_time=i * 0.1
            )
            synchronizer.add_simulation_data(sim_data)
        
        time.sleep(0.05)  # 20 FPS data generation
    
    # Get performance metrics
    metrics = synchronizer.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Update rate: {metrics['update_rate']} FPS")
    print(f"  Avg update time: {metrics['avg_update_time']*1000:.2f} ms")
    print(f"  Frame drops: {metrics['frame_drops']}")
    print(f"  Buffer sizes: {metrics['buffer_sizes']}")
    
    # Stop synchronization
    synchronizer.stop_synchronization()
    
    print("\nRealTimeSynchronizer testing completed!")