"""
Enhanced Physics Simulation with adaptive time stepping, GPU acceleration, and calibration
КРИТИЧЕСКИЕ УЛУЧШЕНИЯ:
- Адаптивный шаг времени для стабильности
- GPU ускорение вычислений (CuPy)
- Калибровка по реальным данным
- Параллелизация на CPU (numba)
- Более точные модели теплообмена
- Модель конвекции расплава
- Реалистичная вязкость
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import warnings

# GPU acceleration (если доступно)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

# JIT compilation для ускорения
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback декоратор
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    NUMBA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU acceleration (если доступно)
if GPU_AVAILABLE:
    logger.info("✅ CuPy доступен, используется GPU ускорение")
else:
    logger.info("⚠️ CuPy недоступен, используется CPU")


@dataclass
class GlassProperties:
    """Расширенные физические свойства стекла"""
    density: float = 2500.0  # kg/m³
    specific_heat: float = 840.0  # J/(kg·K)
    thermal_conductivity: float = 1.4  # W/(m·K)
    
    # Arrhenius viscosity model
    viscosity_pre_exponential: float = 1e-2  # Pa·s
    viscosity_activation_energy: float = 300000.0  # J/mol
    glass_transition_temp: float = 900.0  # K
    
    # Optical properties
    emissivity: float = 0.9  # для излучения
    absorption_coef: float = 0.5  # для поглощения света
    
    # Mechanical properties
    youngs_modulus: float = 70e9  # Pa
    poisson_ratio: float = 0.22
    thermal_expansion: float = 9e-6  # 1/K


@dataclass
class FurnaceParameters:
    """Расширенные параметры печи"""
    length: float = 50.0  # m
    width: float = 5.0  # m
    height: float = 3.0  # m
    wall_thickness: float = 0.5  # m
    wall_thermal_conductivity: float = 1.5  # W/(m·K)
    ambient_temp: float = 300.0  # K
    
    # Burner configuration
    num_burners: int = 12
    burner_power_max: float = 2e6  # W
    burner_efficiency: float = 0.85
    
    # Convection parameters
    convection_coef: float = 25.0  # W/(m²·K)
    radiation_view_factor: float = 0.8


class AdaptiveTimeStepper:
    """Адаптивный выбор шага времени"""
    
    def __init__(self, dt_min: float = 0.01, dt_max: float = 10.0,
                 tolerance: float = 1e-3):
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tolerance = tolerance
        self.dt_current = 1.0
    
    def compute_next_dt(self, error_estimate: float) -> float:
        """Вычисление следующего шага времени на основе ошибки"""
        if error_estimate < self.tolerance / 10:
            # Ошибка очень мала, можно увеличить шаг
            new_dt = min(self.dt_current * 1.5, self.dt_max)
        elif error_estimate > self.tolerance:
            # Ошибка велика, уменьшаем шаг
            new_dt = max(self.dt_current * 0.5, self.dt_min)
        else:
            # Ошибка в пределах нормы
            new_dt = self.dt_current
        
        self.dt_current = new_dt
        return new_dt


class GlassFurnaceSimulator:
    """Улучшенный симулятор с адаптивным шагом и GPU"""
    
    def __init__(self, furnace_params: FurnaceParameters = None,
                 glass_props: GlassProperties = None,
                 use_gpu: bool = True):
        self.furnace = furnace_params or FurnaceParameters()
        self.glass = glass_props or GlassProperties()
        self.R = 8.314  # J/(mol·K)
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        
        # Используем GPU если доступно
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        # Spatial discretization
        self.nx = 100
        self.ny = 20
        self.x_points = np.linspace(0, self.furnace.length, self.nx)
        self.y_points = np.linspace(0, self.furnace.height, self.ny)
        
        # Adaptive time stepping
        self.time_stepper = AdaptiveTimeStepper()
        self.current_time = 0.0
        
        # State variables (2D temperature field)
        self.temperature_field = self.xp.full((self.nx, self.ny), 
                                             self.furnace.ambient_temp)
        self.velocity_field = self.xp.zeros((self.nx, self.ny, 2))  # vx, vy
        self.viscosity_field = self.xp.zeros((self.nx, self.ny))
        self.melt_level = 0.0
        
        # Calibration parameters
        self.calibration_factors = {
            'heat_transfer_coef': 1.0,
            'convection_strength': 1.0,
            'radiation_factor': 1.0
        }
        
        logger.info(f"Enhanced Furnace Simulator инициализирован "
                   f"({'GPU' if self.use_gpu else 'CPU'})")
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_heat_conduction_numba(T: np.ndarray, dx: float, dy: float,
                                       k: float, rho: float, cp: float) -> np.ndarray:
        """JIT-compiled теплопроводность (2D)"""
        nx, ny = T.shape
        dT_dt = np.zeros_like(T)
        alpha = k / (rho * cp)
        
        for i in prange(1, nx - 1):
            for j in range(1, ny - 1):
                d2T_dx2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / (dx**2)
                d2T_dy2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / (dy**2)
                dT_dt[i, j] = alpha * (d2T_dx2 + d2T_dy2)
        
        return dT_dt
    
    def _compute_heat_conduction_gpu(self, T: cp.ndarray) -> cp.ndarray:
        """GPU ускоренная теплопроводность"""
        dx = self.furnace.length / (self.nx - 1)
        dy = self.furnace.height / (self.ny - 1)
        
        k = self.glass.thermal_conductivity * self.calibration_factors['heat_transfer_coef']
        rho = self.glass.density
        cp_val = self.glass.specific_heat
        alpha = k / (rho * cp_val)
        
        # Laplacian operator
        dT_dt = cp.zeros_like(T)
        
        # Interior points
        dT_dt[1:-1, 1:-1] = alpha * (
            (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / (dx**2) +
            (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / (dy**2)
        )
        
        return dT_dt
    
    def _compute_convection(self, T: cp.ndarray) -> cp.ndarray:
        """Расчет конвекции расплава"""
        # Упрощенная модель конвекции на основе buoyancy
        g = 9.81  # m/s²
        beta = self.glass.thermal_expansion
        
        # Температурный градиент вызывает buoyancy-driven flow
        dT_dy = self.xp.gradient(T, axis=1)
        
        # Вертикальная скорость пропорциональна температурному градиенту
        convection_strength = self.calibration_factors['convection_strength']
        vy = convection_strength * g * beta * dT_dy * 0.1
        
        # Конвективный перенос тепла
        dT_conv = -vy * self.xp.gradient(T, axis=1)
        
        return dT_conv
    
    def _compute_radiation(self, T: cp.ndarray) -> cp.ndarray:
        """Расчет радиационных потерь"""
        emissivity = self.glass.emissivity
        sigma = self.stefan_boltzmann
        T_ambient = self.furnace.ambient_temp
        view_factor = self.furnace.radiation_view_factor
        
        radiation_factor = self.calibration_factors['radiation_factor']
        
        # Stefan-Boltzmann law
        q_rad = emissivity * sigma * view_factor * (T**4 - T_ambient**4)
        
        # Конверсия в изменение температуры
        dT_rad = -q_rad / (self.glass.density * self.glass.specific_heat) * radiation_factor
        
        return dT_rad
    
    def _compute_viscosity(self, T: cp.ndarray) -> cp.ndarray:
        """Расчет вязкости по модели Arrhenius"""
        A = self.glass.viscosity_pre_exponential
        E_a = self.glass.viscosity_activation_energy
        
        # Viscosity = A * exp(E_a / (R * T))
        viscosity = A * self.xp.exp(E_a / (self.R * T))
        
        return viscosity
    
    def update(self, heat_input_profile: Optional[np.ndarray] = None,
               fuel_flow_rate: float = 1.0,
               burner_zones: Optional[List[float]] = None) -> Dict:
        """
        Обновление симуляции с адаптивным шагом
        
        Args:
            heat_input_profile: 2D профиль подвода тепла
            fuel_flow_rate: Общий расход топлива (0-1)
            burner_zones: Мощность по зонам горелок
        """
        if heat_input_profile is None:
            # Default heat profile
            heat_input_profile = self.xp.ones((self.nx, self.ny)) * 1e6
            # Выше в зоне плавления
            heat_input_profile[self.nx//3:2*self.nx//3, :] *= 2.0
        else:
            heat_input_profile = self.xp.array(heat_input_profile)
        
        # Текущий шаг времени
        dt = self.time_stepper.dt_current
        
        # Вычисление компонентов изменения температуры
        if self.use_gpu:
            dT_cond = self._compute_heat_conduction_gpu(self.temperature_field)
        else:
            # CPU version с numba
            if GPU_AVAILABLE and isinstance(self.temperature_field, cp.ndarray):
                T_cpu = cp.asnumpy(self.temperature_field)
            else:
                T_cpu = self.temperature_field
            dx = self.furnace.length / (self.nx - 1)
            dy = self.furnace.height / (self.ny - 1)
            dT_cond = self._compute_heat_conduction_numba(
                T_cpu, dx, dy,
                self.glass.thermal_conductivity,
                self.glass.density,
                self.glass.specific_heat
            )
            dT_cond = self.xp.array(dT_cond)
        
        dT_conv = self._compute_convection(self.temperature_field)
        dT_rad = self._compute_radiation(self.temperature_field)
        
        # Подвод тепла от горелок
        heat_source = heat_input_profile * fuel_flow_rate * self.furnace.burner_efficiency
        dT_source = heat_source / (self.glass.density * self.glass.specific_heat)
        
        # Полное изменение температуры
        dT_total = dT_cond + dT_conv + dT_rad + dT_source
        
        # Оценка ошибки для адаптивного шага
        error_estimate = float(self.xp.max(self.xp.abs(dT_total * dt)))
        
        # Обновление температурного поля
        self.temperature_field += dT_total * dt
        
        # Ограничение температуры
        self.temperature_field = self.xp.clip(self.temperature_field, 300.0, 2500.0)
        
        # Обновление вязкости
        self.viscosity_field = self._compute_viscosity(self.temperature_field)
        
        # Обновление уровня расплава
        melt_temp = self.glass.glass_transition_temp + 200
        melted_fraction = float(self.xp.mean(self.temperature_field > melt_temp))
        self.melt_level = np.clip(melted_fraction * self.furnace.height, 0.0, 
                                  self.furnace.height)
        
        # Обновление времени
        self.current_time += dt
        
        # Адаптация шага времени
        new_dt = self.time_stepper.compute_next_dt(error_estimate)
        
        # Конвертация в CPU для вывода
        if self.use_gpu and GPU_AVAILABLE:
            temp_field_cpu = cp.asnumpy(self.temperature_field)
            viscosity_field_cpu = cp.asnumpy(self.viscosity_field)
        else:
            temp_field_cpu = self.temperature_field
            viscosity_field_cpu = self.viscosity_field
        
        return {
            'temperature_field': temp_field_cpu.copy(),
            'viscosity_field': viscosity_field_cpu.copy(),
            'melt_level': self.melt_level,
            'time': self.current_time,
            'dt': dt,
            'error_estimate': error_estimate
        }
    
    def calibrate(self, real_measurements: List[Dict]):
        """
        Калибровка модели по реальным измерениям
        
        Args:
            real_measurements: Список измерений с полями:
                - temperature: измеренная температура
                - position: (x, y) позиция измерения
                - timestamp: время измерения
        """
        logger.info("🔧 Калибровка модели по реальным данным...")
        
        def objective(params):
            """Функция ошибки для оптимизации"""
            self.calibration_factors = {
                'heat_transfer_coef': params[0],
                'convection_strength': params[1],
                'radiation_factor': params[2]
            }
            
            total_error = 0.0
            
            # Симуляция и сравнение с реальными данными
            for measurement in real_measurements[:10]:  # Первые 10 для скорости
                # Получаем предсказание модели
                state = self.update()
                
                # Извлекаем температуру в точке измерения
                x_idx = int(measurement['position'][0] / self.furnace.length * self.nx)
                y_idx = int(measurement['position'][1] / self.furnace.height * self.ny)
                x_idx = np.clip(x_idx, 0, self.nx - 1)
                y_idx = np.clip(y_idx, 0, self.ny - 1)
                
                predicted_temp = state['temperature_field'][x_idx, y_idx]
                actual_temp = measurement['temperature']
                
                error = (predicted_temp - actual_temp) ** 2
                total_error += error
            
            return total_error
        
        # Оптимизация параметров
        initial_params = [1.0, 1.0, 1.0]
        bounds = [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5)]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        self.calibration_factors = {
            'heat_transfer_coef': result.x[0],
            'convection_strength': result.x[1],
            'radiation_factor': result.x[2]
        }
        
        logger.info(f"✅ Калибровка завершена: {self.calibration_factors}")
    
    def get_defect_likelihood_enhanced(self) -> Dict[str, float]:
        """Улучшенная оценка вероятности дефектов"""
        if self.use_gpu and GPU_AVAILABLE:
            temp_field = cp.asnumpy(self.temperature_field)
            visc_field = cp.asnumpy(self.viscosity_field)
        else:
            temp_field = self.temperature_field
            visc_field = self.viscosity_field
        
        temp_avg = np.mean(temp_field)
        temp_std = np.std(temp_field)
        temp_gradient = np.max(np.gradient(temp_field))
        visc_avg = np.mean(visc_field)
        
        defects = {}
        
        # Трещины - высокие термические градиенты
        defects['crack'] = np.clip(temp_gradient / 50.0, 0.0, 1.0)
        
        # Пузыри - слишком высокая температура + низкая вязкость
        bubble_factor = (temp_avg > 1800) * (visc_avg < 100)
        defects['bubble'] = np.clip(bubble_factor * 0.8, 0.0, 1.0)
        
        # Сколы - неравномерная вязкость
        visc_std = np.std(visc_field)
        defects['chip'] = np.clip(visc_std / 1000.0, 0.0, 1.0)
        
        # Мутность - колебания температуры
        defects['cloudiness'] = np.clip(temp_std / 50.0, 0.0, 1.0)
        
        # Деформация - слишком низкая вязкость
        defects['deformation'] = np.clip(1.0 - np.tanh(visc_avg / 1000.0), 0.0, 1.0)
        
        # Напряжения - быстрое охлаждение
        cooling_rate = np.max(np.abs(np.gradient(temp_field, axis=0)))
        defects['stress'] = np.clip(cooling_rate / 100.0, 0.0, 1.0)
        
        return defects


def create_glass_furnace_simulator(
    use_gpu: bool = True,
    furnace_params: FurnaceParameters = None,
    glass_props: GlassProperties = None
) -> GlassFurnaceSimulator:
    """Factory function to create a GlassFurnaceSimulator instance"""
    simulator = GlassFurnaceSimulator(
        furnace_params=furnace_params,
        glass_props=glass_props,
        use_gpu=use_gpu
    )
    
    logger.info(f"✅ Glass Furnace Simulator created: GPU={'enabled' if use_gpu else 'disabled'}")
    
    return simulator


class DigitalTwin:
    """Digital Twin wrapper that provides the expected interface for Phase 3 integration"""
    
    def __init__(self):
        """Initialize the Digital Twin with a Glass Furnace Simulator"""
        self.simulator = GlassFurnaceSimulator()
        self.forming_state = {
            'belt_speed': 150.0,
            'mold_temp': 320.0,
            'quality_score': 0.85
        }
        # Store real sensor data when available
        self.real_sensor_data = None
        logger.info("✅ Digital Twin initialized")
    
    def update_with_real_data(self, sensor_data: Dict):
        """Update digital twin with real sensor data
        
        Args:
            sensor_data: Dict with real sensor measurements
        """
        self.real_sensor_data = sensor_data
        
        # Update forming state with real data if available
        if 'sensors' in sensor_data and 'forming' in sensor_data['sensors']:
            forming_data = sensor_data['sensors']['forming']
            if 'belt_speed' in forming_data:
                self.forming_state['belt_speed'] = forming_data['belt_speed']
            if 'mold_temperature' in forming_data:
                self.forming_state['mold_temp'] = forming_data['mold_temperature']
    
    def step(self, furnace_controls: Dict, forming_controls: Dict) -> Dict:
        """Execute one step of the digital twin simulation
        
        Args:
            furnace_controls: Dict with heat_input_profile and fuel_flow_rate
            forming_controls: Dict with belt_speed and mold_temp
            
        Returns:
            Dict with furnace, forming, and defects state
        """
        # Update forming state with controls
        if 'belt_speed' in forming_controls:
            self.forming_state['belt_speed'] = forming_controls['belt_speed']
        if 'mold_temp' in forming_controls:
            self.forming_state['mold_temp'] = forming_controls['mold_temp']
        
        # Run furnace simulation
        furnace_state = self.simulator.update(
            heat_input_profile=furnace_controls.get('heat_input_profile'),
            fuel_flow_rate=furnace_controls.get('fuel_flow_rate', 1.0)
        )
        
        # Calculate defects based on current state
        defects = self.simulator.get_defect_likelihood_enhanced()
        
        # Estimate quality score based on defects and temperature
        avg_temp = float(np.mean(furnace_state['temperature_field']))
        defect_penalty = sum(defects.values()) / len(defects)
        quality_score = max(0.0, min(1.0, 0.9 - defect_penalty * 0.5 + (abs(avg_temp - 1550) < 50) * 0.1))
        self.forming_state['quality_score'] = quality_score
        
        return {
            'furnace': {
                'temperature_profile': furnace_state['temperature_field'],
                'melt_level': furnace_state['melt_level'],
                'time': furnace_state['time'],
                'viscosity': furnace_state['viscosity_field']
            },
            'forming': self.forming_state.copy(),
            'defects': defects
        }
    
    def get_current_state(self) -> Dict:
        """Get current state of the digital twin, using real data when available"""
        # If we have real sensor data, use it to enhance the simulation
        if self.real_sensor_data and 'sensors' in self.real_sensor_data:
            sensors = self.real_sensor_data['sensors']
            
            # Extract furnace data
            furnace_data = sensors.get('furnace', {})
            forming_data = sensors.get('forming', {})
            
            # Get individual sensor values with proper defaults based on task requirements
            furnace_temp = furnace_data.get('temperature', 1500.0)  # °C
            furnace_pressure = furnace_data.get('pressure', 15.0)    # кПа
            melt_level = furnace_data.get('melt_level', 2500.0)      # мм
            
            belt_speed = forming_data.get('belt_speed', 150.0)       # м/мин
            mold_temp = forming_data.get('mold_temperature', 320.0)   # °C
            forming_pressure = forming_data.get('pressure', 50.0)    # МПа
            
            # Create a more realistic state based on real data
            return {
                'furnace': {
                    'temperature_profile': np.full((100, 20), furnace_temp),
                    'melt_level': melt_level,
                    'time': 0.0,
                    'viscosity': np.full((100, 20), self._calculate_viscosity(furnace_temp))
                },
                'forming': {
                    'belt_speed': belt_speed,
                    'mold_temp': mold_temp,
                    'quality_score': self._calculate_quality_score(furnace_temp, melt_level, belt_speed, mold_temp)
                },
                'defects': self._calculate_defect_probabilities(furnace_temp, melt_level, belt_speed, mold_temp, forming_pressure)
            }
        
        # Otherwise, run a simulation step with default parameters
        furnace_controls = {
            'heat_input_profile': np.ones((100, 20)) * 1e6,
            'fuel_flow_rate': 1.0
        }
        
        forming_controls = {
            'belt_speed': 150.0,
            'mold_temp': 320.0
        }
        
        return self.step(furnace_controls, forming_controls)
    
    def _calculate_viscosity(self, temperature: float) -> float:
        """Calculate glass viscosity based on temperature"""
        # Simplified Arrhenius equation for glass viscosity
        # Viscosity decreases exponentially with temperature
        A = 1e12  # Pre-exponential factor
        Ea = 300000  # Activation energy (J/mol)
        R = 8.314  # Gas constant
        
        # Convert Celsius to Kelvin
        T_K = temperature + 273.15
        
        # Calculate viscosity in Pa·s
        viscosity = A * np.exp(Ea / (R * T_K))
        
        # Clamp to reasonable range for glass production
        return np.clip(viscosity, 100.0, 10000.0)
    
    def _calculate_quality_score(self, furnace_temp: float, melt_level: float, 
                               belt_speed: float, mold_temp: float) -> float:
        """Calculate quality score based on process parameters"""
        # Base quality score
        quality = 0.95
        
        # Penalize deviations from optimal ranges
        # Furnace temperature optimal range: 1450-1550°C
        if not (1450 <= furnace_temp <= 1550):
            temp_deviation = min(abs(furnace_temp - 1450), abs(furnace_temp - 1550))
            quality -= temp_deviation * 0.0001
        
        # Melt level optimal range: 2300-2700 mm
        if not (2300 <= melt_level <= 2700):
            level_deviation = min(abs(melt_level - 2300), abs(melt_level - 2700))
            quality -= level_deviation * 0.00005
        
        # Belt speed optimal range: 130-170 m/min
        if not (130 <= belt_speed <= 170):
            speed_deviation = min(abs(belt_speed - 130), abs(belt_speed - 170))
            quality -= speed_deviation * 0.001
        
        # Mold temperature optimal range: 300-340°C
        if not (300 <= mold_temp <= 340):
            mold_deviation = min(abs(mold_temp - 300), abs(mold_temp - 340))
            quality -= mold_deviation * 0.0005
        
        # Ensure quality is between 0 and 1
        return np.clip(quality, 0.0, 1.0)
    
    def _calculate_defect_probabilities(self, furnace_temp: float, melt_level: float,
                                      belt_speed: float, mold_temp: float, 
                                      forming_pressure: float) -> Dict[str, float]:
        """Calculate defect probabilities based on process parameters"""
        defects = {}
        
        # Crack probability - high temperature gradients
        temp_gradient = abs(furnace_temp - 1500)  # Deviation from optimal
        defects['crack'] = np.clip(temp_gradient * 0.005, 0.0, 0.3)
        
        # Bubble probability - too high temperature or unstable melt level
        bubble_temp_factor = max(0, furnace_temp - 1550) * 0.01
        bubble_level_factor = abs(melt_level - 2500) * 0.0001
        defects['bubble'] = np.clip(bubble_temp_factor + bubble_level_factor, 0.0, 0.25)
        
        # Chip probability - high belt speed or mold temperature issues
        speed_factor = max(0, belt_speed - 170) * 0.01
        mold_factor = abs(mold_temp - 320) * 0.005
        defects['chip'] = np.clip(speed_factor + mold_factor, 0.0, 0.2)
        
        # Cloudiness probability - forming pressure issues
        pressure_deviation = abs(forming_pressure - 50)
        defects['cloudiness'] = np.clip(pressure_deviation * 0.02, 0.0, 0.15)
        
        # Deformation probability - combination of multiple factors
        deformation_factor = (
            abs(furnace_temp - 1500) * 0.001 +
            abs(belt_speed - 150) * 0.005 +
            abs(mold_temp - 320) * 0.002
        )
        defects['deformation'] = np.clip(deformation_factor, 0.0, 0.25)
        
        # Stress probability - rapid cooling or temperature changes
        stress_factor = abs(belt_speed - 150) * 0.01
        defects['stress'] = np.clip(stress_factor, 0.0, 0.15)
        
        return defects


# Пример использования
if __name__ == "__main__":
    sim = GlassFurnaceSimulator(use_gpu=GPU_AVAILABLE)
    
    print(f"🔬 Enhanced Physics Simulator ({'GPU' if GPU_AVAILABLE else 'CPU'})")
    
    # Симуляция
    for i in range(10):
        state = sim.update(fuel_flow_rate=0.9)
        
        if i % 2 == 0:
            print(f"\nШаг {i}:")
            print(f"  Время: {state['time']:.2f}s")
            print(f"  dt: {state['dt']:.4f}s")
            print(f"  Средняя T: {np.mean(state['temperature_field']):.1f}K")
            print(f"  Уровень расплава: {state['melt_level']:.2f}m")
            print(f"  Ошибка: {state['error_estimate']:.6f}")
            
            defects = sim.get_defect_likelihood_enhanced()
            print(f"  Дефекты: crack={defects['crack']:.3f}, "
                  f"bubble={defects['bubble']:.3f}")
    
    print("\n✅ Симуляция завершена!")