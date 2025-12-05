"""
Enhanced Thermal Dynamics Simulation
Advanced heat transfer modeling with conduction, convection, and radiation
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThermalDynamicsSimulator:
    """
    Advanced thermal dynamics simulator with:
    - 3D transient heat conduction
    - Natural and forced convection modeling
    - Radiative heat transfer with view factors
    - Phase change modeling (melting/solidification)
    - Anisotropic thermal properties
    """
    
    def __init__(
        self,
        dimensions: Tuple[float, float, float] = (50.0, 5.0, 3.0),  # length, width, height (m)
        spatial_resolution: Tuple[int, int, int] = (50, 10, 10),
        time_step: float = 0.1,  # seconds
        material_properties: Optional[Dict] = None
    ):
        """
        Args:
            dimensions: Furnace dimensions (length, width, height) in meters
            spatial_resolution: Grid resolution (nx, ny, nz)
            time_step: Time step for simulation in seconds
            material_properties: Dictionary with material properties
        """
        self.length, self.width, self.height = dimensions
        self.nx, self.ny, self.nz = spatial_resolution
        self.dt = time_step
        
        # Spatial discretization
        self.dx = self.length / (self.nx - 1)
        self.dy = self.width / (self.ny - 1)
        self.dz = self.height / (self.nz - 1)
        
        # Material properties with defaults
        self.material = material_properties or {
            'density': 2500.0,  # kg/m³
            'specific_heat': 840.0,  # J/(kg·K)
            'thermal_conductivity': {
                'x': 1.4,  # W/(m·K) in x-direction
                'y': 1.4,  # W/(m·K) in y-direction
                'z': 1.2,  # W/(m·K) in z-direction
            },
            'emissivity': 0.9,
            'thermal_expansion': 9e-6,  # 1/K
        }
        
        # Initialize temperature field (K)
        self.temperature = np.full((self.nx, self.ny, self.nz), 300.0)  # Ambient temperature
        
        # Initialize boundary conditions
        self.boundary_conditions = {
            'west': {'type': 'dirichlet', 'value': 300.0},    # Inlet
            'east': {'type': 'neumann', 'value': 0.0},       # Outlet
            'south': {'type': 'dirichlet', 'value': 300.0},  # Side walls
            'north': {'type': 'dirichlet', 'value': 300.0},
            'bottom': {'type': 'dirichlet', 'value': 300.0}, # Floor
            'top': {'type': 'convection', 'h': 25.0, 'T_inf': 300.0},  # Ceiling
        }
        
        # Heat sources (burners, electrical heating, etc.)
        self.heat_sources = np.zeros((self.nx, self.ny, self.nz))
        
        # Phase change properties
        self.phase_change = {
            'melting_temp': 1400.0,  # K
            'latent_heat': 250000.0,  # J/kg
            'mushy_zone': 50.0,  # K temperature range for phase transition
        }
        
        # Convection parameters
        self.convection = {
            'natural': True,
            'forced': False,
            'velocity_field': np.zeros((self.nx, self.ny, self.nz, 3)),  # vx, vy, vz
        }
        
        # Radiation parameters
        self.radiation = {
            'enabled': True,
            'stefan_boltzmann': 5.67e-8,  # W/(m²·K⁴)
            'view_factors': self._compute_view_factors(),
        }
        
        # Time tracking
        self.current_time = 0.0
        
        logger.info(f"Initialized ThermalDynamicsSimulator: {self.nx}x{self.ny}x{self.nz} grid")
    
    def _compute_view_factors(self) -> np.ndarray:
        """
        Compute view factors between surfaces for radiation calculation
        Simplified model assuming uniform view factors
        """
        # For a rectangular enclosure, view factors can be computed analytically
        # This is a simplified approximation
        view_factors = np.ones((6, 6)) * 0.2  # 6 surfaces: west, east, south, north, bottom, top
        
        # Diagonal is 0 (surface doesn't see itself)
        np.fill_diagonal(view_factors, 0.0)
        
        # Normalize rows to sum to 1
        view_factors = view_factors / np.sum(view_factors, axis=1, keepdims=True)
        
        return view_factors
    
    def set_boundary_condition(self, surface: str, condition_type: str, value: float, **kwargs):
        """
        Set boundary condition for a surface
        
        Args:
            surface: Surface name ('west', 'east', 'south', 'north', 'bottom', 'top')
            condition_type: 'dirichlet', 'neumann', 'convection'
            value: Boundary value
            **kwargs: Additional parameters (e.g., 'h' for convection coefficient)
        """
        self.boundary_conditions[surface] = {
            'type': condition_type,
            'value': value,
            **kwargs
        }
        logger.info(f"Set {condition_type} BC on {surface}: {value}")
    
    def add_heat_source(self, position: Tuple[int, int, int], power: float, radius: int = 1):
        """
        Add a heat source at a specific position
        
        Args:
            position: Grid position (i, j, k)
            power: Heat power in Watts
            radius: Radius of heat source influence
        """
        i, j, k = position
        
        # Add heat source with Gaussian distribution
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                for dk in range(-radius, radius+1):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < self.nx and 0 <= jj < self.ny and 0 <= kk < self.nz:
                        distance = np.sqrt(di**2 + dj**2 + dk**2)
                        if distance <= radius:
                            # Gaussian decay
                            weight = np.exp(-distance**2 / (2 * (radius/2)**2))
                            self.heat_sources[ii, jj, kk] += power * weight
    
    def _compute_conduction(self) -> np.ndarray:
        """
        Compute heat conduction using finite difference method
        
        Returns:
            dT/dt array for conduction
        """
        kx = self.material['thermal_conductivity']['x']
        ky = self.material['thermal_conductivity']['y']
        kz = self.material['thermal_conductivity']['z']
        rho = self.material['density']
        cp = self.material['specific_heat']
        
        # Thermal diffusivity components
        alpha_x = kx / (rho * cp)
        alpha_y = ky / (rho * cp)
        alpha_z = kz / (rho * cp)
        
        # Initialize conduction term
        dT_cond = np.zeros_like(self.temperature)
        
        # Interior points
        dT_cond[1:-1, 1:-1, 1:-1] = (
            alpha_x * (self.temperature[2:, 1:-1, 1:-1] - 2*self.temperature[1:-1, 1:-1, 1:-1] + self.temperature[:-2, 1:-1, 1:-1]) / self.dx**2 +
            alpha_y * (self.temperature[1:-1, 2:, 1:-1] - 2*self.temperature[1:-1, 1:-1, 1:-1] + self.temperature[1:-1, :-2, 1:-1]) / self.dy**2 +
            alpha_z * (self.temperature[1:-1, 1:-1, 2:] - 2*self.temperature[1:-1, 1:-1, 1:-1] + self.temperature[1:-1, 1:-1, :-2]) / self.dz**2
        )
        
        return dT_cond
    
    def _compute_convection(self) -> np.ndarray:
        """
        Compute convective heat transfer
        
        Returns:
            dT/dt array for convection
        """
        if not (self.convection['natural'] or self.convection['forced']):
            return np.zeros_like(self.temperature)
        
        # Simplified convection model
        # In reality, this would involve solving Navier-Stokes equations
        velocity = self.convection['velocity_field']
        
        # Upwind scheme for convective terms
        dT_conv = np.zeros_like(self.temperature)
        
        # X-direction convection
        vx = velocity[:, :, :, 0]
        dT_dx = np.gradient(self.temperature, self.dx, axis=0)
        dT_conv -= vx * dT_dx
        
        # Y-direction convection
        vy = velocity[:, :, :, 1]
        dT_dy = np.gradient(self.temperature, self.dy, axis=1)
        dT_conv -= vy * dT_dy
        
        # Z-direction convection
        vz = velocity[:, :, :, 2]
        dT_dz = np.gradient(self.temperature, self.dz, axis=2)
        dT_conv -= vz * dT_dz
        
        return dT_conv
    
    def _compute_radiation(self) -> np.ndarray:
        """
        Compute radiative heat transfer
        
        Returns:
            dT/dt array for radiation
        """
        if not self.radiation['enabled']:
            return np.zeros_like(self.temperature)
        
        sigma = self.radiation['stefan_boltzmann']
        emissivity = self.material['emissivity']
        T_inf = 300.0  # Ambient temperature
        
        # Net radiative heat transfer
        # Simplified model: assuming radiation to ambient
        q_rad = emissivity * sigma * (self.temperature**4 - T_inf**4)
        
        # Convert to temperature rate of change
        rho = self.material['density']
        cp = self.material['specific_heat']
        dT_rad = -q_rad / (rho * cp)
        
        return dT_rad
    
    def _compute_phase_change(self) -> np.ndarray:
        """
        Compute latent heat effects during phase change
        
        Returns:
            Additional dT/dt term for phase change
        """
        T_melt = self.phase_change['melting_temp']
        L = self.phase_change['latent_heat']
        mushy_zone = self.phase_change['mushy_zone']
        
        rho = self.material['density']
        
        # Liquid fraction calculation
        # 0 = solid, 1 = liquid
        liquid_fraction = np.clip((self.temperature - (T_melt - mushy_zone/2)) / mushy_zone, 0.0, 1.0)
        
        # Time derivative of liquid fraction
        dT = np.gradient(self.temperature, self.dt, axis=0)  # Simplified
        d_liquid_fraction_dt = np.clip(dT / mushy_zone, -1.0, 1.0)
        
        # Latent heat source term
        dT_phase = -L * d_liquid_fraction_dt / (rho * self.material['specific_heat'])
        
        return dT_phase
    
    def _apply_boundary_conditions(self):
        """Apply boundary conditions to temperature field"""
        # West boundary (inlet)
        bc = self.boundary_conditions['west']
        if bc['type'] == 'dirichlet':
            self.temperature[0, :, :] = bc['value']
        elif bc['type'] == 'neumann':
            self.temperature[0, :, :] = self.temperature[1, :, :] - bc['value'] * self.dx
        
        # East boundary (outlet)
        bc = self.boundary_conditions['east']
        if bc['type'] == 'dirichlet':
            self.temperature[-1, :, :] = bc['value']
        elif bc['type'] == 'neumann':
            self.temperature[-1, :, :] = self.temperature[-2, :, :] + bc['value'] * self.dx
        
        # South boundary
        bc = self.boundary_conditions['south']
        if bc['type'] == 'dirichlet':
            self.temperature[:, 0, :] = bc['value']
        
        # North boundary
        bc = self.boundary_conditions['north']
        if bc['type'] == 'dirichlet':
            self.temperature[:, -1, :] = bc['value']
        
        # Bottom boundary
        bc = self.boundary_conditions['bottom']
        if bc['type'] == 'dirichlet':
            self.temperature[:, :, 0] = bc['value']
        
        # Top boundary
        bc = self.boundary_conditions['top']
        if bc['type'] == 'dirichlet':
            self.temperature[:, :, -1] = bc['value']
        elif bc['type'] == 'convection':
            h = bc['h']
            T_inf = bc['T_inf']
            rho = self.material['density']
            cp = self.material['specific_heat']
            # Newton's law of cooling
            q_conv = h * (self.temperature[:, :, -1] - T_inf)
            dT_conv = -q_conv / (rho * cp)
            # This would be applied in the time stepping, not directly to temperature
    
    def step(self, external_heat_sources: Optional[np.ndarray] = None) -> Dict:
        """
        Perform one time step of the thermal simulation
        
        Args:
            external_heat_sources: Additional heat sources to apply
            
        Returns:
            Dictionary with simulation state
        """
        # Apply external heat sources if provided
        if external_heat_sources is not None:
            total_heat_sources = self.heat_sources + external_heat_sources
        else:
            total_heat_sources = self.heat_sources
        
        # Compute all heat transfer mechanisms
        dT_cond = self._compute_conduction()
        dT_conv = self._compute_convection()
        dT_rad = self._compute_radiation()
        dT_phase = self._compute_phase_change()
        
        # Total heat source contribution
        rho = self.material['density']
        cp = self.material['specific_heat']
        dT_source = total_heat_sources / (rho * cp)
        
        # Total temperature rate of change
        dT_total = dT_cond + dT_conv + dT_rad + dT_phase + dT_source
        
        # Update temperature field
        self.temperature += dT_total * self.dt
        
        # Apply boundary conditions
        self._apply_boundary_conditions()
        
        # Update time
        self.current_time += self.dt
        
        # Compute derived quantities
        avg_temp = np.mean(self.temperature)
        max_temp = np.max(self.temperature)
        min_temp = np.min(self.temperature)
        
        return {
            'temperature_field': self.temperature.copy(),
            'time': self.current_time,
            'avg_temperature': avg_temp,
            'max_temperature': max_temp,
            'min_temperature': min_temp,
            'heat_sources': total_heat_sources.copy(),
        }
    
    def get_thermal_stress(self) -> np.ndarray:
        """
        Estimate thermal stress based on temperature gradients
        
        Returns:
            Thermal stress field
        """
        alpha = self.material['thermal_expansion']
        E = 70e9  # Young's modulus for glass, Pa
        nu = 0.22  # Poisson's ratio
        
        # Temperature gradient
        dT_dx = np.gradient(self.temperature, self.dx, axis=0)
        dT_dy = np.gradient(self.temperature, self.dy, axis=1)
        dT_dz = np.gradient(self.temperature, self.dz, axis=2)
        
        # Simplified thermal stress calculation
        # In reality, this would involve solving elasticity equations
        thermal_stress = E * alpha * np.sqrt(dT_dx**2 + dT_dy**2 + dT_dz**2)
        
        return thermal_stress


def create_thermal_dynamics_simulator(**kwargs) -> ThermalDynamicsSimulator:
    """
    Factory function to create a ThermalDynamicsSimulator instance
    
    Args:
        **kwargs: Parameters for ThermalDynamicsSimulator
        
    Returns:
        ThermalDynamicsSimulator instance
    """
    simulator = ThermalDynamicsSimulator(**kwargs)
    logger.info("Created ThermalDynamicsSimulator")
    return simulator


if __name__ == "__main__":
    # Example usage
    simulator = create_thermal_dynamics_simulator()
    
    # Add some heat sources
    simulator.add_heat_source((25, 5, 5), 1e6, radius=3)  # Main burner
    simulator.add_heat_source((10, 2, 2), 5e5, radius=2)   # Preheating zone
    
    # Set boundary conditions
    simulator.set_boundary_condition('west', 'dirichlet', 1600.0)  # Hot inlet
    simulator.set_boundary_condition('east', 'neumann', 0.0)      # Insulated outlet
    
    # Run simulation
    print("Running thermal dynamics simulation...")
    for i in range(10):
        state = simulator.step()
        if i % 5 == 0:
            print(f"Step {i}: Avg T = {state['avg_temperature']:.1f} K, "
                  f"Max T = {state['max_temperature']:.1f} K")
    
    print("Simulation completed!")