"""
Stress Analysis for Glass Manufacturing
Thermal stress, residual stress, and fracture mechanics analysis
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressAnalyzer:
    """
    Advanced stress analyzer for glass with capabilities for:
    - Thermal stress analysis
    - Residual stress computation
    - Fracture mechanics
    - Stress intensity factors
    - Failure probability estimation
    """
    
    def __init__(
        self,
        material_properties: Optional[Dict] = None,
        geometry: Optional[Dict] = None
    ):
        """
        Args:
            material_properties: Dictionary with material properties
            geometry: Dictionary with geometric parameters
        """
        # Default material properties for soda-lime glass
        self.material = material_properties or {
            'youngs_modulus': 70e9,  # Pa
            'poissons_ratio': 0.22,
            'thermal_expansion': 9e-6,  # 1/K
            'thermal_conductivity': 1.4,  # W/(m·K)
            'tensile_strength': 50e6,  # Pa
            'fracture_toughness': 0.75,  # MPa·√m
            'density': 2500.0,  # kg/m³
        }
        
        # Geometric properties
        self.geometry = geometry or {
            'thickness': 0.01,  # m
            'width': 1.0,  # m
            'length': 2.0,  # m
        }
        
        # Stress analysis parameters
        self.analysis_parameters = {
            'safety_factor': 2.0,
            'surface_flaw_size': 1e-6,  # m (Griffith flaw size)
            'weibull_modulus': 10.0,  # For failure probability
            'characteristic_strength': 100e6,  # Pa
        }
        
        logger.info("Initialized StressAnalyzer")
    
    def compute_thermal_stress(
        self,
        temperature_field: np.ndarray,
        temperature_reference: float = 300.0,
        boundary_conditions: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute thermal stress from temperature field
        
        Args:
            temperature_field: 2D or 3D temperature distribution (K)
            temperature_reference: Reference temperature (K)
            boundary_conditions: Stress boundary conditions
            
        Returns:
            Dictionary with stress components
        """
        # Material properties
        E = self.material['youngs_modulus']
        nu = self.material['poissons_ratio']
        alpha = self.material['thermal_expansion']
        
        # Temperature difference from reference
        delta_T = temperature_field - temperature_reference
        
        # Simplified 2D thermal stress analysis (plane stress assumption)
        if len(delta_T.shape) == 2:
            nx, ny = delta_T.shape
            
            # Thermal strain
            epsilon_th = alpha * delta_T
            
            # Simplified stress calculation (assuming constrained in one direction)
            # This is a very simplified approach - real analysis would require FEM
            sigma_x = E / (1 - nu) * (epsilon_th - nu * epsilon_th)  # Simplified
            sigma_y = sigma_x  # Assuming isotropic
            sigma_xy = np.zeros_like(sigma_x)
            
            # Maximum principal stress
            sigma_max = (sigma_x + sigma_y) / 2 + np.sqrt(((sigma_x - sigma_y) / 2)**2 + sigma_xy**2)
            sigma_min = (sigma_x + sigma_y) / 2 - np.sqrt(((sigma_x - sigma_y) / 2)**2 + sigma_xy**2)
            
        else:  # 3D case
            nx, ny, nz = delta_T.shape
            
            # Thermal strain (3D)
            epsilon_th = alpha * delta_T
            
            # Simplified 3D stress (this is highly simplified)
            sigma_x = sigma_y = sigma_z = E * epsilon_th / (1 - 2*nu)
            sigma_xy = sigma_yz = sigma_xz = np.zeros_like(sigma_x)
            
            # Maximum principal stress (simplified)
            sigma_max = sigma_x
            sigma_min = sigma_x
    
        return {
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'sigma_z': sigma_z if 'sigma_z' in locals() else np.zeros_like(sigma_x),
            'sigma_xy': sigma_xy,
            'sigma_yz': sigma_yz if 'sigma_yz' in locals() else np.zeros_like(sigma_x),
            'sigma_xz': sigma_xz if 'sigma_xz' in locals() else np.zeros_like(sigma_x),
            'sigma_max': sigma_max,
            'sigma_min': sigma_min,
            'temperature_field': temperature_field
        }
    
    def compute_residual_stress(
        self,
        cooling_rate_field: np.ndarray,
        thickness: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute residual stress from cooling process
        
        Args:
            cooling_rate_field: Cooling rate distribution (K/s)
            thickness: Glass thickness (m), uses default if None
            
        Returns:
            Dictionary with residual stress components
        """
        thickness = thickness or self.geometry['thickness']
        E = self.material['youngs_modulus']
        alpha = self.material['thermal_expansion']
        
        # Simplified residual stress model
        # σ_res = E * α * ΔT_quench
        # Where ΔT_quench is estimated from cooling rate
        
        # Estimate quench temperature difference from cooling rate
        # This is a very simplified relationship
        delta_T_quench = cooling_rate_field * thickness / (2 * self.material['thermal_conductivity'])
        
        # Residual stress
        sigma_res = E * alpha * delta_T_quench
        
        # Surface stress (typically higher due to faster cooling)
        sigma_surface = sigma_res * 1.5  # Empirical factor
        
        # Through-thickness stress distribution (simplified)
        if len(sigma_res.shape) == 2:
            nz = 20  # Discretization in thickness direction
            z_coords = np.linspace(-thickness/2, thickness/2, nz)
            
            # Parabolic distribution
            stress_profile = np.zeros((*sigma_res.shape, nz))
            for i in range(nz):
                z_norm = z_coords[i] / (thickness/2)
                stress_profile[:, :, i] = sigma_res * (1 - z_norm**2)
        else:
            stress_profile = sigma_res
        
        return {
            'residual_stress': sigma_res,
            'surface_stress': sigma_surface,
            'through_thickness_profile': stress_profile,
            'cooling_rate_field': cooling_rate_field
        }
    
    def compute_stress_intensity_factor(
        self,
        stress_field: np.ndarray,
        crack_length: float,
        crack_orientation: str = 'edge'
    ) -> float:
        """
        Compute stress intensity factor for fracture mechanics
        
        Args:
            stress_field: Stress field where crack is located
            crack_length: Crack length (m)
            crack_orientation: 'edge', 'surface', or 'embedded'
            
        Returns:
            Stress intensity factor K_I (Pa·√m)
        """
        # Maximum stress in the field
        sigma_max = np.max(stress_field)
        
        # Geometry factor based on crack type
        if crack_orientation == 'edge':
            beta = 1.12  # Edge crack
        elif crack_orientation == 'surface':
            beta = 1.0  # Surface crack
        else:  # embedded
            beta = 1.12  # Internal crack
        
        # Stress intensity factor
        K_I = beta * sigma_max * np.sqrt(np.pi * crack_length)
        
        return K_I
    
    def check_fracture_criterion(
        self,
        stress_intensity_factor: float,
        fracture_toughness: Optional[float] = None
    ) -> Dict[str, Union[float, bool]]:
        """
        Check if fracture criterion is met
        
        Args:
            stress_intensity_factor: K_I value
            fracture_toughness: Material fracture toughness, uses default if None
            
        Returns:
            Dictionary with fracture analysis results
        """
        K_IC = fracture_toughness or self.material['fracture_toughness'] * 1e6  # Convert to Pa·√m
        
        # Fracture criterion: K_I >= K_IC
        will_fracture = stress_intensity_factor >= K_IC
        
        # Safety margin
        safety_margin = K_IC / stress_intensity_factor if stress_intensity_factor > 0 else float('inf')
        
        return {
            'stress_intensity_factor': stress_intensity_factor,
            'fracture_toughness': K_IC,
            'will_fracture': will_fracture,
            'safety_margin': safety_margin,
            'fracture_probability': 1.0 if will_fracture else 0.0
        }
    
    def estimate_failure_probability(
        self,
        stress_field: np.ndarray,
        volume: Optional[float] = None
    ) -> float:
        """
        Estimate probability of failure using Weibull statistics
        
        Args:
            stress_field: Stress distribution
            volume: Component volume (m³), calculated if None
            
        Returns:
            Probability of failure
        """
        # Weibull parameters
        m = self.analysis_parameters['weibull_modulus']
        sigma_0 = self.analysis_parameters['characteristic_strength']
        
        # Volume calculation if not provided
        if volume is None:
            thickness = self.geometry['thickness']
            area = self.geometry['width'] * self.geometry['length']
            volume = area * thickness
        
        # Maximum stress
        sigma_max = np.max(stress_field)
        
        # Weibull probability of failure
        P_f = 1 - np.exp(-(sigma_max / sigma_0)**m * volume)
        
        return P_f
    
    def analyze_stress_concentration(
        self,
        geometry_discontinuities: Dict[str, float],
        applied_stress: float
    ) -> Dict[str, float]:
        """
        Analyze stress concentration around geometric discontinuities
        
        Args:
            geometry_discontinuities: Dictionary with discontinuity types and sizes
            applied_stress: Applied stress (Pa)
            
        Returns:
            Dictionary with stress concentration factors
        """
        stress_concentrations = {}
        
        for discontinuity, size in geometry_discontinuities.items():
            if discontinuity == 'hole':
                # Stress concentration factor for circular hole in plate
                # K_t = 3 for through-thickness hole
                K_t = 3.0
            elif discontinuity == 'notch':
                # Approximate stress concentration for sharp notch
                # K_t ≈ 2.0 + 2.0 * sqrt(a/ρ) where a is crack length, ρ is notch radius
                # Simplified approximation
                K_t = 2.0 + 2.0 * np.sqrt(size / 1e-6)  # Assuming very small notch radius
                K_t = min(K_t, 10.0)  # Cap at reasonable value
            elif discontinuity == 'fillet':
                # Stress concentration for fillet
                # Decreases with increasing fillet radius
                K_t = max(1.5, 3.0 - size / 1e-3)  # Simplified relationship
            else:
                K_t = 1.0  # No concentration
            
            stress_concentrations[discontinuity] = {
                'stress_concentration_factor': K_t,
                'maximum_stress': K_t * applied_stress,
                'is_critical': K_t * applied_stress > self.material['tensile_strength']
            }
        
        return stress_concentrations
    
    def get_stress_analysis_report(
        self,
        temperature_field: np.ndarray,
        cooling_rate_field: Optional[np.ndarray] = None
    ) -> Dict[str, Union[float, Dict]]:
        """
        Generate comprehensive stress analysis report
        
        Args:
            temperature_field: Temperature distribution
            cooling_rate_field: Cooling rate distribution (optional)
            
        Returns:
            Dictionary with comprehensive stress analysis
        """
        # Thermal stress analysis
        thermal_stress = self.compute_thermal_stress(temperature_field)
        max_thermal_stress = np.max(thermal_stress['sigma_max'])
        
        # Residual stress analysis (if cooling rate provided)
        if cooling_rate_field is not None:
            residual_stress = self.compute_residual_stress(cooling_rate_field)
            max_residual_stress = np.max(residual_stress['residual_stress'])
        else:
            residual_stress = None
            max_residual_stress = 0.0
        
        # Combined stress
        combined_stress = max_thermal_stress + max_residual_stress
        
        # Safety factors
        tensile_strength = self.material['tensile_strength']
        thermal_safety_factor = tensile_strength / max_thermal_stress if max_thermal_stress > 0 else float('inf')
        combined_safety_factor = tensile_strength / combined_stress if combined_stress > 0 else float('inf')
        
        # Failure probability
        failure_probability = self.estimate_failure_probability(thermal_stress['sigma_max'])
        
        # Stress concentration analysis (example with a hole)
        stress_concentration = self.analyze_stress_concentration(
            {'hole': 5e-3},  # 5mm hole
            combined_stress
        )
        
        return {
            'thermal_stress': {
                'max_thermal_stress': max_thermal_stress,
                'thermal_stress_field': thermal_stress['sigma_max'],
                'safety_factor': thermal_safety_factor
            },
            'residual_stress': {
                'max_residual_stress': max_residual_stress,
                'residual_stress_field': residual_stress['residual_stress'] if residual_stress else None
            },
            'combined_analysis': {
                'combined_stress': combined_stress,
                'combined_safety_factor': combined_safety_factor,
                'failure_probability': failure_probability
            },
            'stress_concentration': stress_concentration,
            'material_properties': self.material
        }


def create_stress_analyzer(**kwargs) -> StressAnalyzer:
    """
    Factory function to create a StressAnalyzer instance
    
    Args:
        **kwargs: Parameters for StressAnalyzer
        
    Returns:
        StressAnalyzer instance
    """
    analyzer = StressAnalyzer(**kwargs)
    logger.info("Created StressAnalyzer")
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("Testing StressAnalyzer...")
    
    # Create analyzer
    analyzer = create_stress_analyzer()
    
    # Generate test temperature field (simplified)
    nx, ny = 50, 20
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 0.5, ny)
    X, Y = np.meshgrid(x, y)
    
    # Non-uniform temperature field (hot center, cool edges)
    temperature_field = 300 + 1200 * np.exp(-((X - 0.5)**2 + (Y - 0.25)**2) / 0.1)
    
    # Cooling rate field
    cooling_rate_field = 100 * np.ones_like(temperature_field)  # 100 K/s uniform cooling
    
    # Perform stress analysis
    print("Performing stress analysis...")
    report = analyzer.get_stress_analysis_report(temperature_field, cooling_rate_field)
    
    # Print results
    print("\nStress Analysis Report:")
    print(f"  Max Thermal Stress: {report['thermal_stress']['max_thermal_stress']:.2e} Pa")
    print(f"  Thermal Safety Factor: {report['thermal_stress']['safety_factor']:.2f}")
    print(f"  Max Residual Stress: {report['residual_stress']['max_residual_stress']:.2e} Pa")
    print(f"  Combined Stress: {report['combined_analysis']['combined_stress']:.2e} Pa")
    print(f"  Combined Safety Factor: {report['combined_analysis']['combined_safety_factor']:.2f}")
    print(f"  Failure Probability: {report['combined_analysis']['failure_probability']:.2e}")
    
    # Stress concentration analysis
    print("\nStress Concentration Analysis:")
    for discontinuity, data in report['stress_concentration'].items():
        if isinstance(data, dict):
            print(f"  {discontinuity}:")
            print(f"    K_t = {data['stress_concentration_factor']:.2f}")
            print(f"    Max Stress = {data['maximum_stress']:.2e} Pa")
            print(f"    Critical = {data['is_critical']}")
    
    print("\nStressAnalyzer testing completed!")