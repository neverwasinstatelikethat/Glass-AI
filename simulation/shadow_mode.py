"""
Shadow Mode Simulator for Digital Twin
Runs parallel simulation alongside real data stream to validate system behavior.

Workflow:
Real Sensor Data → Digital Twin Model → Predicted State → Compare with Actual State → Validation Metrics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a prediction made in shadow mode"""
    timestamp: datetime
    predicted_state: Dict
    actual_state: Optional[Dict] = None
    prediction_horizon_seconds: int = 300  # 5 minutes default
    prediction_error: Optional[Dict] = None
    
    def calculate_error(self):
        """Calculate prediction error when actual state becomes available"""
        if self.actual_state is None:
            return None
        
        errors = {}
        
        # Calculate errors for furnace parameters
        if 'furnace' in self.predicted_state and 'furnace' in self.actual_state:
            pred_furnace = self.predicted_state['furnace']
            actual_furnace = self.actual_state['furnace']
            
            if 'temperature' in pred_furnace and 'temperature' in actual_furnace:
                errors['furnace_temperature'] = abs(pred_furnace['temperature'] - actual_furnace['temperature'])
            
            if 'melt_level' in pred_furnace and 'melt_level' in actual_furnace:
                errors['melt_level'] = abs(pred_furnace['melt_level'] - actual_furnace['melt_level'])
        
        # Calculate errors for forming parameters
        if 'forming' in self.predicted_state and 'forming' in self.actual_state:
            pred_forming = self.predicted_state['forming']
            actual_forming = self.actual_state['forming']
            
            if 'belt_speed' in pred_forming and 'belt_speed' in actual_forming:
                errors['belt_speed'] = abs(pred_forming['belt_speed'] - actual_forming['belt_speed'])
            
            if 'mold_temp' in pred_forming and 'mold_temp' in actual_forming:
                errors['mold_temp'] = abs(pred_forming['mold_temp'] - actual_forming['mold_temp'])
        
        # Calculate errors for quality score
        if 'quality_score' in self.predicted_state and 'quality_score' in self.actual_state:
            errors['quality_score'] = abs(self.predicted_state['quality_score'] - self.actual_state['quality_score'])
        
        self.prediction_error = errors
        return errors


@dataclass
class ShadowModeMetrics:
    """Metrics for shadow mode performance"""
    total_predictions: int = 0
    correct_predictions: int = 0
    avg_temperature_error: float = 0.0
    avg_quality_error: float = 0.0
    avg_defect_prediction_accuracy: float = 0.0
    model_drift_detected: bool = False
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    def get_accuracy(self) -> float:
        """Get overall prediction accuracy"""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions


class ShadowModeSimulator:
    """
    Shadow Mode Simulator that runs parallel simulation alongside real production
    to validate Digital Twin behavior and detect model drift.
    """
    
    def __init__(self, digital_twin, prediction_window_seconds: int = 300):
        """
        Initialize Shadow Mode Simulator
        
        Args:
            digital_twin: Digital Twin instance to use for predictions
            prediction_window_seconds: How far ahead to predict (default 5 minutes)
        """
        self.digital_twin = digital_twin
        self.prediction_window = prediction_window_seconds
        
        # Store prediction records
        self.prediction_history: deque = deque(maxlen=1000)
        
        # Current state tracking
        self.current_real_state: Optional[Dict] = None
        self.current_predicted_state: Optional[Dict] = None
        
        # Metrics
        self.metrics = ShadowModeMetrics()
        
        # Thresholds for drift detection
        self.drift_thresholds = {
            'temperature_error': 50.0,  # °C
            'quality_error': 0.15,      # 15% quality deviation
            'consecutive_errors': 5     # Number of consecutive high errors
        }
        
        self.consecutive_high_errors = 0
        
        logger.info(f"✅ Shadow Mode Simulator initialized with {prediction_window_seconds}s prediction window")
    
    def update_with_real_data(self, sensor_data: Dict):
        """
        Update shadow mode with real sensor data
        
        Args:
            sensor_data: Real sensor measurements from production line
        """
        # Store current real state
        self.current_real_state = self._extract_state_from_sensors(sensor_data)
        
        # Update digital twin with real data
        self.digital_twin.update_with_real_data(sensor_data)
        
        # Make prediction for next time step
        predicted_state = self._predict_next_state(sensor_data)
        self.current_predicted_state = predicted_state
        
        # Create prediction record
        prediction_record = PredictionRecord(
            timestamp=datetime.utcnow(),
            predicted_state=predicted_state,
            prediction_horizon_seconds=self.prediction_window
        )
        
        self.prediction_history.append(prediction_record)
        
        # Match predictions with actual outcomes
        self._match_predictions_with_actuals()
        
        # Update metrics
        self._update_metrics()
        
        # Check for model drift
        self._check_model_drift()
    
    def _extract_state_from_sensors(self, sensor_data: Dict) -> Dict:
        """Extract state representation from sensor data"""
        state = {}
        
        if 'sensors' in sensor_data:
            sensors = sensor_data['sensors']
            
            # Extract furnace state
            if 'furnace' in sensors:
                furnace = sensors['furnace']
                state['furnace'] = {
                    'temperature': furnace.get('temperature', 0.0),
                    'pressure': furnace.get('pressure', 0.0),
                    'melt_level': furnace.get('melt_level', 0.0)
                }
            
            # Extract forming state
            if 'forming' in sensors:
                forming = sensors['forming']
                state['forming'] = {
                    'belt_speed': forming.get('belt_speed', 0.0),
                    'mold_temp': forming.get('mold_temperature', 0.0),
                    'pressure': forming.get('pressure', 0.0)
                }
        
        # Extract quality score if available
        if 'quality_score' in sensor_data:
            state['quality_score'] = sensor_data['quality_score']
        
        state['timestamp'] = datetime.utcnow()
        
        return state
    
    def _predict_next_state(self, current_sensor_data: Dict) -> Dict:
        """
        Predict next state using Digital Twin
        
        Args:
            current_sensor_data: Current sensor readings
            
        Returns:
            Predicted state after prediction_window seconds
        """
        # Get current digital twin state
        twin_state = self.digital_twin.get_current_state()
        
        # Extract current parameters
        sensors = current_sensor_data.get('sensors', {})
        furnace_data = sensors.get('furnace', {})
        forming_data = sensors.get('forming', {})
        
        # Assume parameters remain stable (simple prediction model)
        # In production, this would use trend analysis and ML models
        predicted_state = {
            'furnace': {
                'temperature': furnace_data.get('temperature', 1500.0),
                'pressure': furnace_data.get('pressure', 15.0),
                'melt_level': furnace_data.get('melt_level', 2500.0)
            },
            'forming': {
                'belt_speed': forming_data.get('belt_speed', 150.0),
                'mold_temp': forming_data.get('mold_temperature', 320.0),
                'pressure': forming_data.get('pressure', 50.0)
            },
            'quality_score': twin_state['forming'].get('quality_score', 0.85),
            'defects': twin_state.get('defects', {}),
            'timestamp': datetime.utcnow() + timedelta(seconds=self.prediction_window)
        }
        
        return predicted_state
    
    def _match_predictions_with_actuals(self):
        """Match old predictions with current actual state"""
        if self.current_real_state is None:
            return
        
        current_time = datetime.utcnow()
        
        # Find predictions that should have materialized by now
        for record in self.prediction_history:
            if record.actual_state is not None:
                continue  # Already matched
            
            # Check if prediction window has elapsed
            time_since_prediction = (current_time - record.timestamp).total_seconds()
            
            if time_since_prediction >= record.prediction_horizon_seconds - 30:  # 30s tolerance
                # This prediction should match current state
                record.actual_state = self.current_real_state.copy()
                record.calculate_error()
                
                logger.debug(f"Matched prediction from {record.timestamp} with actual state")
    
    def _update_metrics(self):
        """Update performance metrics based on prediction history"""
        # Count predictions with actual outcomes
        completed_predictions = [r for r in self.prediction_history if r.actual_state is not None]
        
        if not completed_predictions:
            return
        
        self.metrics.total_predictions = len(completed_predictions)
        
        # Calculate average errors
        temp_errors = []
        quality_errors = []
        
        for record in completed_predictions:
            if record.prediction_error:
                if 'furnace_temperature' in record.prediction_error:
                    temp_errors.append(record.prediction_error['furnace_temperature'])
                if 'quality_score' in record.prediction_error:
                    quality_errors.append(record.prediction_error['quality_score'])
        
        if temp_errors:
            self.metrics.avg_temperature_error = np.mean(temp_errors)
        
        if quality_errors:
            self.metrics.avg_quality_error = np.mean(quality_errors)
        
        # Count correct predictions (within tolerance)
        correct = 0
        for record in completed_predictions:
            if record.prediction_error:
                temp_error = record.prediction_error.get('furnace_temperature', 0)
                quality_error = record.prediction_error.get('quality_score', 0)
                
                # Consider prediction correct if errors are within thresholds
                if temp_error < 30.0 and quality_error < 0.1:
                    correct += 1
        
        self.metrics.correct_predictions = correct
        self.metrics.last_update = datetime.utcnow()
    
    def _check_model_drift(self):
        """Check if model is drifting from reality"""
        # Check if average errors exceed thresholds
        temp_drift = self.metrics.avg_temperature_error > self.drift_thresholds['temperature_error']
        quality_drift = self.metrics.avg_quality_error > self.drift_thresholds['quality_error']
        
        if temp_drift or quality_drift:
            self.consecutive_high_errors += 1
        else:
            self.consecutive_high_errors = 0
        
        # Detect drift if consecutive errors exceed threshold
        if self.consecutive_high_errors >= self.drift_thresholds['consecutive_errors']:
            if not self.metrics.model_drift_detected:
                self.metrics.model_drift_detected = True
                logger.warning(f"⚠️ Model drift detected! Avg temp error: {self.metrics.avg_temperature_error:.1f}°C, "
                             f"Avg quality error: {self.metrics.avg_quality_error:.3f}")
        else:
            self.metrics.model_drift_detected = False
    
    def get_validation_metrics(self) -> Dict:
        """
        Get validation metrics for shadow mode
        
        Returns:
            Dict with prediction accuracy and error metrics
        """
        return {
            'total_predictions': self.metrics.total_predictions,
            'prediction_accuracy': self.metrics.get_accuracy(),
            'avg_temperature_error': self.metrics.avg_temperature_error,
            'avg_quality_error': self.metrics.avg_quality_error,
            'model_drift_detected': self.metrics.model_drift_detected,
            'last_update': self.metrics.last_update.isoformat()
        }
    
    def get_state_comparison(self) -> Dict:
        """
        Compare current predicted state with actual state
        
        Returns:
            Dict with predicted vs actual comparison
        """
        if self.current_real_state is None or self.current_predicted_state is None:
            return {
                'available': False,
                'message': 'Insufficient data for comparison'
            }
        
        comparison = {
            'available': True,
            'timestamp': datetime.utcnow().isoformat(),
            'predicted': self.current_predicted_state,
            'actual': self.current_real_state,
            'deviation': {}
        }
        
        # Calculate deviations
        if 'furnace' in self.current_predicted_state and 'furnace' in self.current_real_state:
            pred_temp = self.current_predicted_state['furnace'].get('temperature', 0)
            actual_temp = self.current_real_state['furnace'].get('temperature', 0)
            comparison['deviation']['furnace_temperature'] = pred_temp - actual_temp
        
        if 'quality_score' in self.current_predicted_state and 'quality_score' in self.current_real_state:
            pred_quality = self.current_predicted_state['quality_score']
            actual_quality = self.current_real_state['quality_score']
            comparison['deviation']['quality_score'] = pred_quality - actual_quality
        
        return comparison
