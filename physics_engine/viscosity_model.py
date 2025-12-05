"""
Advanced Viscosity Model for Glass Melt
Temperature-dependent viscosity modeling with multiple empirical relationships
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViscosityModel:
    """
    Advanced viscosity model for glass melt with multiple modeling approaches:
    - Arrhenius model
    - Vogel-Fulcher-Tammann (VFT) model
    - WLF model
    - Empirical polynomial fits
    - Composition-dependent models
    """
    
    def __init__(
        self,
        model_type: str = "vft",
        reference_parameters: Optional[Dict] = None,
        composition_dependent: bool = False
    ):
        """
        Args:
            model_type: Type of viscosity model ('arrhenius', 'vft', 'wlf', 'empirical')
            reference_parameters: Model-specific parameters
            composition_dependent: Whether to consider glass composition
        """
        self.model_type = model_type
        self.composition_dependent = composition_dependent
        
        # Default parameters for different glass types
        self.default_parameters = {
            'soda_lime': {
                'arrhenius': {
                    'A': 1e-2,  # Pre-exponential factor (Pa·s)
                    'E_a': 300000.0,  # Activation energy (J/mol)
                },
                'vft': {
                    'A': 1e-2,  # Pre-exponential factor
                    'B': 5000.0,  # VFT parameter (K)
                    'T_0': 300.0,  # VFT reference temperature (K)
                },
                'wlf': {
                    'C_1': 17.4,  # WLF parameter
                    'C_2': 51.6,  # WLF parameter (K)
                    'T_g': 900.0,  # Glass transition temperature (K)
                }
            },
            'borosilicate': {
                'vft': {
                    'A': 5e-3,
                    'B': 6000.0,
                    'T_0': 350.0,
                }
            }
        }
        
        # Initialize parameters
        self.parameters = reference_parameters or self.default_parameters['soda_lime'][model_type]
        
        # Composition effects (if enabled)
        if composition_dependent:
            self.composition_effects = {
                'SiO2': {'coefficient': -0.1, 'reference': 70.0},  # %
                'Na2O': {'coefficient': 0.2, 'reference': 15.0},   # %
                'CaO': {'coefficient': -0.05, 'reference': 10.0},  # %
                'Al2O3': {'coefficient': 0.15, 'reference': 2.0},  # %
            }
        else:
            self.composition_effects = {}
        
        # Temperature range for validity
        self.valid_temperature_range = (800.0, 2000.0)  # K
        
        # Viscosity bounds
        self.viscosity_bounds = (1e-2, 1e12)  # Pa·s
        
        logger.info(f"Initialized ViscosityModel: {model_type}")
    
    def _arrhenius_viscosity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Arrhenius model: η = A * exp(E_a / (R * T))
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Viscosity in Pa·s
        """
        A = self.parameters['A']
        E_a = self.parameters['E_a']
        R = 8.314  # Gas constant J/(mol·K)
        
        viscosity = A * np.exp(E_a / (R * T))
        return np.clip(viscosity, self.viscosity_bounds[0], self.viscosity_bounds[1])
    
    def _vft_viscosity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Vogel-Fulcher-Tammann model: η = A * exp(B / (T - T_0))
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Viscosity in Pa·s
        """
        A = self.parameters['A']
        B = self.parameters['B']
        T_0 = self.parameters['T_0']
        
        # Avoid division by zero
        T_safe = np.maximum(T, T_0 + 10.0)
        viscosity = A * np.exp(B / (T_safe - T_0))
        return np.clip(viscosity, self.viscosity_bounds[0], self.viscosity_bounds[1])
    
    def _wlf_viscosity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Williams-Landel-Ferry model: log(η/η_g) = -C_1 * (T - T_g) / (C_2 + T - T_g)
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Viscosity in Pa·s
        """
        C_1 = self.parameters['C_1']
        C_2 = self.parameters['C_2']
        T_g = self.parameters['T_g']
        eta_g = 1e12  # Reference viscosity at T_g (Pa·s)
        
        # Avoid division by zero
        denominator = C_2 + T - T_g
        denominator_safe = np.maximum(np.abs(denominator), 1e-6) * np.sign(denominator)
        
        log_eta_ratio = -C_1 * (T - T_g) / denominator_safe
        viscosity = eta_g * np.exp(log_eta_ratio)
        return np.clip(viscosity, self.viscosity_bounds[0], self.viscosity_bounds[1])
    
    def _empirical_viscosity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Empirical polynomial fit: log10(η) = a + b/T + c/T^2 + ...
        Coefficients are stored in parameters as 'coefficients'
        
        Args:
            T: Temperature in Kelvin
            
        Returns:
            Viscosity in Pa·s
        """
        coeffs = self.parameters['coefficients']
        
        # Polynomial evaluation: log10(η) = sum(coeff[i] * T^(-i))
        log_eta = np.zeros_like(T, dtype=float)
        for i, coeff in enumerate(coeffs):
            log_eta += coeff * np.power(T, -i)
        
        viscosity = np.power(10.0, log_eta)
        return np.clip(viscosity, self.viscosity_bounds[0], self.viscosity_bounds[1])
    
    def compute_viscosity(
        self, 
        temperature: Union[float, np.ndarray], 
        composition: Optional[Dict[str, float]] = None
    ) -> Union[float, np.ndarray]:
        """
        Compute viscosity at given temperature(s) with optional composition correction
        
        Args:
            temperature: Temperature in Kelvin (scalar or array)
            composition: Glass composition as weight percentages
            
        Returns:
            Viscosity in Pa·s (scalar or array)
        """
        # Convert to numpy array for consistent handling
        T = np.asarray(temperature)
        
        # Check temperature bounds
        T_clipped = np.clip(T, self.valid_temperature_range[0], self.valid_temperature_range[1])
        if not np.array_equal(T, T_clipped):
            logger.warning("Temperature outside valid range, clipping to bounds")
        
        # Select model
        if self.model_type == "arrhenius":
            viscosity = self._arrhenius_viscosity(T_clipped)
        elif self.model_type == "vft":
            viscosity = self._vft_viscosity(T_clipped)
        elif self.model_type == "wlf":
            viscosity = self._wlf_viscosity(T_clipped)
        elif self.model_type == "empirical":
            viscosity = self._empirical_viscosity(T_clipped)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Apply composition corrections if provided
        if composition and self.composition_dependent:
            viscosity = self._apply_composition_correction(viscosity, composition)
        
        return viscosity
    
    def _apply_composition_correction(
        self, 
        viscosity: Union[float, np.ndarray], 
        composition: Dict[str, float]
    ) -> Union[float, np.ndarray]:
        """
        Apply composition-dependent corrections to viscosity
        
        Args:
            viscosity: Base viscosity value(s)
            composition: Glass composition as weight percentages
            
        Returns:
            Corrected viscosity
        """
        correction_factor = 1.0
        
        for oxide, effect in self.composition_effects.items():
            if oxide in composition:
                actual_content = composition[oxide]
                reference_content = effect['reference']
                coefficient = effect['coefficient']
                
                # Linear correction
                deviation = actual_content - reference_content
                correction_factor *= np.exp(coefficient * deviation / 100.0)
        
        return viscosity * correction_factor
    
    def fit_model(
        self, 
        temperature_data: np.ndarray, 
        viscosity_data: np.ndarray,
        model_type: Optional[str] = None
    ) -> Dict:
        """
        Fit model parameters to experimental data
        
        Args:
            temperature_data: Temperature values in Kelvin
            viscosity_data: Viscosity values in Pa·s
            model_type: Model type to fit (uses current if None)
            
        Returns:
            Dictionary with fitted parameters and statistics
        """
        model_to_fit = model_type or self.model_type
        
        # Convert to log scale for better fitting
        log_viscosity = np.log(viscosity_data)
        
        if model_to_fit == "arrhenius":
            # Linear fit: ln(η) = ln(A) + E_a/(R*T)
            x = 1.0 / temperature_data
            coeffs = np.polyfit(x, log_viscosity, 1)
            E_a = coeffs[0] * 8.314  # Convert back to J/mol
            A = np.exp(coeffs[1])
            
            fitted_params = {'A': A, 'E_a': E_a}
            
        elif model_to_fit == "vft":
            # Nonlinear fit for VFT model
            def vft_model(T, A, B, T_0):
                return np.log(A) + B / (T - T_0)
            
            try:
                popt, pcov = curve_fit(
                    vft_model, 
                    temperature_data, 
                    log_viscosity,
                    p0=[self.parameters['A'], self.parameters['B'], self.parameters['T_0']],
                    maxfev=10000
                )
                fitted_params = {'A': popt[0], 'B': popt[1], 'T_0': popt[2]}
            except Exception as e:
                logger.warning(f"VFT fitting failed: {e}")
                fitted_params = self.parameters
        
        else:
            logger.warning(f"Fitting not implemented for model type: {model_to_fit}")
            fitted_params = self.parameters
        
        # Update model parameters
        self.parameters = fitted_params
        self.model_type = model_to_fit
        
        return {
            'parameters': fitted_params,
            'model_type': model_to_fit,
            'success': True
        }
    
    def get_flow_properties(self, temperature: float) -> Dict[str, float]:
        """
        Get comprehensive flow properties at a given temperature
        
        Args:
            temperature: Temperature in Kelvin
            
        Returns:
            Dictionary with flow properties
        """
        viscosity = self.compute_viscosity(temperature)
        
        # Derived properties
        properties = {
            'viscosity': viscosity,  # Pa·s
            'kinematic_viscosity': viscosity / 2500.0,  # m²/s (assuming density = 2500 kg/m³)
            'relaxation_time': viscosity / 1e9,  # s (assuming shear modulus = 1 GPa)
            'deborah_number': 1e-3 / (viscosity / 1e9),  # Characteristic time = 1ms
        }
        
        # Flow regime classification
        if viscosity < 10:
            properties['flow_regime'] = 'liquid'
        elif viscosity < 1000:
            properties['flow_regime'] = 'viscous'
        elif viscosity < 1e6:
            properties['flow_regime'] = 'glassy'
        else:
            properties['flow_regime'] = 'solid'
        
        return properties


class TemperatureDependentViscosityField:
    """
    Spatial field of viscosity values based on temperature distribution
    """
    
    def __init__(
        self,
        viscosity_model: ViscosityModel,
        temperature_field: np.ndarray,
        spatial_coordinates: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    ):
        """
        Args:
            viscosity_model: Viscosity model instance
            temperature_field: 3D temperature field in Kelvin
            spatial_coordinates: Spatial coordinates (x, y, z) arrays
        """
        self.viscosity_model = viscosity_model
        self.temperature_field = temperature_field
        self.spatial_coordinates = spatial_coordinates
        
        # Compute viscosity field
        self.viscosity_field = self.viscosity_model.compute_viscosity(temperature_field)
        
        logger.info(f"Created viscosity field: {temperature_field.shape}")
    
    def get_viscosity_at_point(self, point_index: Tuple[int, int, int]) -> float:
        """
        Get viscosity at a specific point
        
        Args:
            point_index: (i, j, k) indices
            
        Returns:
            Viscosity at point in Pa·s
        """
        i, j, k = point_index
        return self.viscosity_field[i, j, k]
    
    def get_average_viscosity(self) -> float:
        """
        Get average viscosity across the field
        
        Returns:
            Average viscosity in Pa·s
        """
        return np.mean(self.viscosity_field)
    
    def get_viscosity_gradient(self) -> np.ndarray:
        """
        Compute viscosity gradient field
        
        Returns:
            4D array with viscosity gradients (dη/dx, dη/dy, dη/dz)
        """
        if self.spatial_coordinates is None:
            logger.warning("Spatial coordinates not provided, using uniform spacing")
            dx = dy = dz = 1.0
        else:
            x, y, z = self.spatial_coordinates
            dx = np.diff(x)[0] if len(x) > 1 else 1.0
            dy = np.diff(y)[0] if len(y) > 1 else 1.0
            dz = np.diff(z)[0] if len(z) > 1 else 1.0
        
        dη_dx = np.gradient(self.viscosity_field, dx, axis=0)
        dη_dy = np.gradient(self.viscosity_field, dy, axis=1)
        dη_dz = np.gradient(self.viscosity_field, dz, axis=2)
        
        return np.stack([dη_dx, dη_dy, dη_dz], axis=-1)


def create_viscosity_model(**kwargs) -> ViscosityModel:
    """
    Factory function to create a ViscosityModel instance
    
    Args:
        **kwargs: Parameters for ViscosityModel
        
    Returns:
        ViscosityModel instance
    """
    model = ViscosityModel(**kwargs)
    logger.info("Created ViscosityModel")
    return model


if __name__ == "__main__":
    # Example usage
    print("Testing ViscosityModel...")
    
    # Create model
    model = create_viscosity_model(model_type="vft")
    
    # Test viscosity computation
    temperatures = np.array([1000, 1200, 1400, 1600, 1800])
    viscosities = model.compute_viscosity(temperatures)
    
    print("Temperature (K) | Viscosity (Pa·s)")
    print("-" * 35)
    for T, eta in zip(temperatures, viscosities):
        print(f"{T:12.1f} | {eta:12.2e}")
    
    # Test flow properties
    print("\nFlow properties at 1500K:")
    properties = model.get_flow_properties(1500.0)
    for key, value in properties.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    
    # Test fitting
    print("\nTesting model fitting...")
    # Generate synthetic data
    true_temps = np.linspace(1000, 1800, 20)
    true_viscosities = model.compute_viscosity(true_temps)
    
    # Add some noise
    noisy_viscosities = true_viscosities * np.exp(np.random.normal(0, 0.1, len(true_viscosities)))
    
    # Fit new model
    new_model = create_viscosity_model(model_type="arrhenius")
    fit_result = new_model.fit_model(true_temps, noisy_viscosities)
    print(f"Fitting result: {fit_result}")
    
    print("ViscosityModel testing completed!")