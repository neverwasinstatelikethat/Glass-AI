"""
Production Line Simulator for Digital Twin
Complete simulation of glass production line including forming, annealing, and quality control
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductType(Enum):
    """Types of glass products"""
    FLAT_GLASS = "flat_glass"
    CONTAINER_GLASS = "container_glass"
    FIBER_GLASS = "fiber_glass"
    SPECIALTY_GLASS = "specialty_glass"


class QualityGrade(Enum):
    """Quality grades for glass products"""
    EXCELLENT = "excellent"  # A+ grade
    GOOD = "good"           # A grade
    ACCEPTABLE = "acceptable"  # B grade
    REJECT = "reject"       # C grade or reject


@dataclass
class GlassProduct:
    """Glass product with properties and quality metrics"""
    product_id: str
    product_type: ProductType
    dimensions: Tuple[float, float, float]  # length, width, thickness (m)
    temperature: float = 0.0  # K
    viscosity: float = 0.0    # Pa·s
    stress_level: float = 0.0  # MPa
    defects: Dict[str, float] = field(default_factory=dict)  # defect_type: severity
    quality_grade: QualityGrade = QualityGrade.REJECT
    production_time: float = 0.0
    forming_speed: float = 0.0  # m/min
    surface_roughness: float = 0.0  # μm
    optical_quality: float = 0.0  # 0.0 to 1.0


@dataclass
class ProductionParameters:
    """Production line parameters"""
    forming_speed: float = 150.0  # m/min
    mold_temperature: float = 320.0  # K
    annealing_temperature: float = 200.0  # K
    cooling_rate: float = 50.0  # K/min
    quality_threshold: float = 0.8  # Minimum quality score
    production_rate: float = 60.0  # pieces per hour


class ProductionSimulator:
    """
    Complete production line simulator with:
    - Glass forming simulation
    - Annealing process
    - Quality control
    - Production rate optimization
    """
    
    def __init__(
        self,
        product_type: ProductType = ProductType.FLAT_GLASS,
        production_params: Optional[ProductionParameters] = None,
        furnace_temperature: float = 1500.0  # K
    ):
        """
        Args:
            product_type: Type of glass product being produced
            production_params: Production parameters
            furnace_temperature: Initial furnace temperature
        """
        self.product_type = product_type
        self.params = production_params or ProductionParameters()
        self.furnace_temperature = furnace_temperature
        
        # Production state
        self.is_running = False
        self.production_count = 0
        self.total_products = 0
        self.rejected_products = 0
        self.current_product: Optional[GlassProduct] = None
        
        # Quality metrics
        self.quality_history = deque(maxlen=1000)
        self.production_rate_history = deque(maxlen=100)
        
        # Equipment state
        self.forming_line_temperature = furnace_temperature
        self.annealing_temperature = self.params.annealing_temperature
        self.cooling_system_efficiency = 1.0
        
        # Threading
        self.simulation_thread = None
        self.lock = threading.Lock()
        
        # Product specifications
        self.product_specs = {
            ProductType.FLAT_GLASS: {
                'nominal_thickness': 0.01,  # 10mm
                'nominal_width': 2.0,       # 2m
                'nominal_length': 3.0,      # 3m
                'target_temperature': 1400.0,
                'target_viscosity': 50.0
            },
            ProductType.CONTAINER_GLASS: {
                'nominal_thickness': 0.005,  # 5mm
                'nominal_width': 0.1,        # 10cm
                'nominal_length': 0.1,       # 10cm
                'target_temperature': 1300.0,
                'target_viscosity': 100.0
            }
        }
        
        logger.info(f"Initialized ProductionSimulator for {product_type.value}")
    
    def start_production(self):
        """
        Start production simulation
        """
        if self.is_running:
            logger.warning("Production already running")
            return
        
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._production_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        logger.info("Started production simulation")
    
    def stop_production(self):
        """
        Stop production simulation
        """
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=5.0)
        logger.info("Stopped production simulation")
    
    def _production_loop(self):
        """
        Main production simulation loop
        """
        last_product_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time to produce next product
                product_interval = 3600.0 / self.params.production_rate  # seconds per product
                if current_time - last_product_time >= product_interval:
                    # Produce new product
                    product = self._produce_product()
                    
                    with self.lock:
                        self.current_product = product
                        self.production_count += 1
                        self.total_products += 1
                        
                        # Update quality metrics
                        if product.quality_grade == QualityGrade.REJECT:
                            self.rejected_products += 1
                        
                        self.quality_history.append(product.quality_grade)
                        
                        # Update production rate
                        if len(self.production_rate_history) > 10:
                            rate = len([p for p in self.quality_history if p != QualityGrade.REJECT]) / 10
                            self.production_rate_history.append(rate)
                    
                    last_product_time = current_time
                    logger.info(f"Produced product {product.product_id}: {product.quality_grade.value}")
                
                # Update equipment states
                self._update_equipment_states()
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in production loop: {e}")
                time.sleep(1.0)
    
    def _produce_product(self) -> GlassProduct:
        """
        Simulate production of a single glass product
        
        Returns:
            Produced GlassProduct
        """
        # Create new product
        product_id = f"GLASS_{self.total_products:06d}"
        
        # Get product specifications
        specs = self.product_specs.get(self.product_type, self.product_specs[ProductType.FLAT_GLASS])
        
        # Simulate forming process
        forming_result = self._simulate_forming_process(specs)
        
        # Simulate annealing process
        annealing_result = self._simulate_annealing_process(forming_result)
        
        # Perform quality control
        quality_result = self._perform_quality_control(annealing_result)
        
        # Determine final quality grade
        quality_grade = self._determine_quality_grade(quality_result)
        
        # Create product object
        product = GlassProduct(
            product_id=product_id,
            product_type=self.product_type,
            dimensions=(specs['nominal_length'], specs['nominal_width'], specs['nominal_thickness']),
            temperature=quality_result['final_temperature'],
            viscosity=quality_result['final_viscosity'],
            stress_level=quality_result['stress_level'],
            defects=quality_result['defects'],
            quality_grade=quality_grade,
            production_time=time.time(),
            forming_speed=self.params.forming_speed,
            surface_roughness=quality_result['surface_roughness'],
            optical_quality=quality_result['optical_quality']
        )
        
        return product
    
    def _simulate_forming_process(self, specs: Dict) -> Dict[str, Any]:
        """
        Simulate glass forming process
        
        Args:
            specs: Product specifications
            
        Returns:
            Forming process results
        """
        # Simulate temperature evolution during forming
        initial_temp = self.furnace_temperature
        target_temp = specs['target_temperature']
        
        # Temperature change during forming (cooling)
        temp_drop = (initial_temp - target_temp) * (1 - np.exp(-0.1 * self.params.forming_speed / 100))
        forming_temp = initial_temp - temp_drop + np.random.normal(0, 10)  # Add noise
        
        # Viscosity based on temperature
        viscosity = specs['target_viscosity'] * np.exp((forming_temp - target_temp) / 200)
        viscosity *= np.random.normal(1.0, 0.1)  # Process variation
        
        # Stress development during forming
        stress_level = abs(forming_temp - target_temp) * 0.1 + np.random.normal(0, 5)
        stress_level = max(0, stress_level)
        
        return {
            'temperature': forming_temp,
            'viscosity': viscosity,
            'stress_level': stress_level,
            'forming_time': 60.0 / self.params.production_rate  # seconds
        }
    
    def _simulate_annealing_process(self, forming_result: Dict) -> Dict[str, Any]:
        """
        Simulate annealing process
        
        Args:
            forming_result: Results from forming process
            
        Returns:
            Annealing process results
        """
        # Annealing temperature control
        annealing_temp = self.annealing_temperature + np.random.normal(0, 5)
        
        # Cooling from forming temperature to annealing temperature
        temp_diff = forming_result['temperature'] - annealing_temp
        cooling_time = temp_diff / (self.params.cooling_rate * self.cooling_system_efficiency)
        cooling_time = max(0, cooling_time)
        
        # Final temperature after annealing
        final_temp = annealing_temp + np.random.normal(0, 2)
        
        # Stress relief during annealing
        stress_relief = forming_result['stress_level'] * np.exp(-cooling_time / 60)
        final_stress = stress_relief * 0.5 + np.random.normal(0, 2)  # Residual stress
        final_stress = max(0, final_stress)
        
        return {
            'initial_temperature': forming_result['temperature'],
            'final_temperature': final_temp,
            'cooling_time': cooling_time,
            'initial_stress': forming_result['stress_level'],
            'final_stress': final_stress,
            'annealing_temperature': annealing_temp
        }
    
    def _perform_quality_control(self, annealing_result: Dict) -> Dict[str, Any]:
        """
        Perform quality control checks
        
        Args:
            annealing_result: Results from annealing process
            
        Returns:
            Quality control results
        """
        # Simulate defect detection
        defects = {}
        
        # Crack probability based on stress
        crack_prob = min(1.0, annealing_result['final_stress'] / 50.0)
        if np.random.random() < crack_prob:
            defects['crack'] = np.random.uniform(0.1, 0.9)
        
        # Bubble probability based on temperature control
        bubble_prob = abs(annealing_result['final_temperature'] - 200) / 1000
        if np.random.random() < bubble_prob:
            defects['bubble'] = np.random.uniform(0.1, 0.7)
        
        # Surface roughness based on forming quality
        surface_roughness = 0.1 + abs(annealing_result['final_stress']) / 10 + np.random.normal(0, 0.05)
        surface_roughness = max(0.05, surface_roughness)
        
        # Optical quality based on multiple factors
        temp_stability = 1.0 - abs(annealing_result['final_temperature'] - 200) / 100
        stress_quality = 1.0 - annealing_result['final_stress'] / 100
        defect_impact = 1.0 - sum(defects.values()) / 10 if defects else 1.0
        
        optical_quality = np.clip(temp_stability * stress_quality * defect_impact, 0.0, 1.0)
        
        # Final viscosity
        final_viscosity = 1000 * np.exp(1400 / annealing_result['final_temperature'])
        
        return {
            'defects': defects,
            'final_temperature': annealing_result['final_temperature'],
            'final_viscosity': final_viscosity,
            'stress_level': annealing_result['final_stress'],
            'surface_roughness': surface_roughness,
            'optical_quality': optical_quality,
            'quality_score': optical_quality * (1.0 - sum(defects.values()) / 10)
        }
    
    def _determine_quality_grade(self, quality_result: Dict) -> QualityGrade:
        """
        Determine quality grade based on quality metrics
        
        Args:
            quality_result: Quality control results
            
        Returns:
            Quality grade
        """
        quality_score = quality_result['quality_score']
        
        if quality_score >= 0.95:
            return QualityGrade.EXCELLENT
        elif quality_score >= 0.85:
            return QualityGrade.GOOD
        elif quality_score >= self.params.quality_threshold:
            return QualityGrade.ACCEPTABLE
        else:
            return QualityGrade.REJECT
    
    def _update_equipment_states(self):
        """
        Update equipment states based on production
        """
        # Simulate equipment wear and efficiency changes
        self.cooling_system_efficiency *= (1 - np.random.random() * 0.001)
        self.cooling_system_efficiency = max(0.8, self.cooling_system_efficiency)
        
        # Small temperature fluctuations
        self.forming_line_temperature += np.random.normal(0, 2)
        self.annealing_temperature += np.random.normal(0, 1)
    
    def adjust_parameters(self, new_params: ProductionParameters):
        """
        Adjust production parameters in real-time
        
        Args:
            new_params: New production parameters
        """
        with self.lock:
            self.params = new_params
        logger.info("Adjusted production parameters")
    
    def get_production_status(self) -> Dict[str, Any]:
        """
        Get current production status
        
        Returns:
            Dictionary with production status
        """
        with self.lock:
            if self.total_products > 0:
                quality_rate = (self.total_products - self.rejected_products) / self.total_products
            else:
                quality_rate = 0.0
            
            return {
                'is_running': self.is_running,
                'production_count': self.production_count,
                'total_products': self.total_products,
                'rejected_products': self.rejected_products,
                'quality_rate': quality_rate,
                'current_product': self.current_product,
                'forming_temperature': self.forming_line_temperature,
                'annealing_temperature': self.annealing_temperature,
                'cooling_efficiency': self.cooling_system_efficiency,
                'production_rate': self.params.production_rate
            }
    
    def get_quality_statistics(self) -> Dict[str, Any]:
        """
        Get quality statistics
        
        Returns:
            Dictionary with quality statistics
        """
        with self.lock:
            if not self.quality_history:
                return {
                    'grade_distribution': {},
                    'average_quality_score': 0.0,
                    'defect_frequency': {}
                }
            
            # Grade distribution
            grade_counts = {}
            for grade in self.quality_history:
                grade_counts[grade.value] = grade_counts.get(grade.value, 0) + 1
            
            # Defect frequency
            defect_counts = {}
            total_defects = 0
            for product in [p for p in self.quality_history if hasattr(p, 'defects')]:
                if isinstance(product, GlassProduct):
                    for defect_type in product.defects:
                        defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
                        total_defects += 1
            
            # Average quality (simplified)
            quality_scores = []
            for grade in self.quality_history:
                if grade == QualityGrade.EXCELLENT:
                    quality_scores.append(0.975)
                elif grade == QualityGrade.GOOD:
                    quality_scores.append(0.90)
                elif grade == QualityGrade.ACCEPTABLE:
                    quality_scores.append(0.75)
                else:
                    quality_scores.append(0.3)
            
            return {
                'grade_distribution': grade_counts,
                'average_quality_score': np.mean(quality_scores) if quality_scores else 0.0,
                'defect_frequency': defect_counts,
                'total_defects': total_defects
            }
    
    def reset_production(self):
        """
        Reset production statistics
        """
        with self.lock:
            self.production_count = 0
            self.total_products = 0
            self.rejected_products = 0
            self.quality_history.clear()
            self.production_rate_history.clear()
            self.current_product = None
        logger.info("Reset production statistics")


def create_production_simulator(**kwargs) -> ProductionSimulator:
    """
    Factory function to create a ProductionSimulator instance
    
    Args:
        **kwargs: Parameters for ProductionSimulator
        
    Returns:
        ProductionSimulator instance
    """
    simulator = ProductionSimulator(**kwargs)
    logger.info("Created ProductionSimulator")
    return simulator


if __name__ == "__main__":
    # Example usage
    print("Testing ProductionSimulator...")
    
    # Create simulator
    simulator = create_production_simulator(
        product_type=ProductType.FLAT_GLASS,
        production_params=ProductionParameters(
            forming_speed=180.0,
            production_rate=80.0,
            quality_threshold=0.75
        )
    )
    
    # Start production
    simulator.start_production()
    
    # Monitor production for 10 seconds
    print("Starting production simulation...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < 10.0:
            status = simulator.get_production_status()
            print(f"Status: {status['production_count']} products, "
                  f"Quality: {status['quality_rate']:.2f}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping simulation...")
    
    # Stop production
    simulator.stop_production()
    
    # Get final statistics
    status = simulator.get_production_status()
    quality_stats = simulator.get_quality_statistics()
    
    print(f"\nFinal Production Statistics:")
    print(f"  Total products: {status['total_products']}")
    print(f"  Rejected products: {status['rejected_products']}")
    print(f"  Quality rate: {status['quality_rate']:.2f}")
    print(f"  Grade distribution: {quality_stats['grade_distribution']}")
    print(f"  Defect frequency: {quality_stats['defect_frequency']}")
    
    print("\nProductionSimulator testing completed!")