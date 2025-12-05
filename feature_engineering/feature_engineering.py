"""
Feature Engineering для предиктивной аналитики производства стекла
Извлечение статистических, временных и доменных признаков
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeFeatureExtractor:
    """Извлечение признаков в реальном времени из потока данных"""
    
    def __init__(
        self,
        window_size: int = 60,
        window_step: int = 1
    ):
        self.window_size = window_size
        self.window_step = window_step
        
        # Буферы для временных окон
        self.sensor_buffers: Dict[str, deque] = {}
        self.timestamp_buffer = deque(maxlen=window_size)
        
        # Инициализация буферов для каждого датчика
        self._initialize_buffers()
    
    def _initialize_buffers(self):
        """Инициализация буферов для всех датчиков"""
        sensor_keys = [
            "furnace_temperature", "furnace_pressure", "furnace_melt_level",
            "furnace_o2", "furnace_co2",
            "forming_mold_temperature", "forming_pressure", "forming_belt_speed",
            "annealing_temperature", "batch_flow"
        ]
        
        for key in sensor_keys:
            self.sensor_buffers[key] = deque(maxlen=self.window_size)
    
    def update(self, sensor_data: Dict[str, Any]) -> bool:
        """Обновление буферов новыми данными"""
        try:
            timestamp = datetime.fromisoformat(sensor_data["timestamp"])
            self.timestamp_buffer.append(timestamp)
            
            sensors = sensor_data.get("sensors", {})
            
            # Обновление буферов печи
            furnace = sensors.get("furnace", {})
            self.sensor_buffers["furnace_temperature"].append(
                furnace.get("temperature", np.nan)
            )
            self.sensor_buffers["furnace_pressure"].append(
                furnace.get("pressure", np.nan)
            )
            self.sensor_buffers["furnace_melt_level"].append(
                furnace.get("melt_level", np.nan)
            )
            self.sensor_buffers["furnace_o2"].append(
                furnace.get("o2_percent", np.nan)
            )
            self.sensor_buffers["furnace_co2"].append(
                furnace.get("co2_percent", np.nan)
            )
            
            # Обновление буферов формования
            forming = sensors.get("forming", {})
            self.sensor_buffers["forming_mold_temperature"].append(
                forming.get("mold_temperature", np.nan)
            )
            self.sensor_buffers["forming_pressure"].append(
                forming.get("pressure", np.nan)
            )
            self.sensor_buffers["forming_belt_speed"].append(
                forming.get("belt_speed", np.nan)
            )
            
            # Обновление буферов отжига
            annealing = sensors.get("annealing", {})
            self.sensor_buffers["annealing_temperature"].append(
                annealing.get("temperature", np.nan)
            )
            
            # Обновление процесса
            process = sensors.get("process", {})
            self.sensor_buffers["batch_flow"].append(
                process.get("batch_flow", np.nan)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления буферов: {e}")
            return False
    
    def extract_features(self) -> Dict[str, float]:
        """Извлечение всех признаков из текущих буферов"""
        if len(self.timestamp_buffer) < 10:
            logger.warning("⚠️ Недостаточно данных для извлечения признаков")
            return {}
        
        features = {}
        
        # Извлечение признаков для каждого датчика
        for sensor_name, buffer in self.sensor_buffers.items():
            if len(buffer) < 10:
                continue
            
            values = np.array(list(buffer))
            values = values[~np.isnan(values)]
            
            if len(values) == 0:
                continue
            
            # Статистические признаки
            features.update(self._statistical_features(sensor_name, values))
            
            # Временные признаки
            features.update(self._temporal_features(sensor_name, values))
            
            # Тренд признаки
            features.update(self._trend_features(sensor_name, values))
        
        # Доменные признаки (специфичные для производства стекла)
        features.update(self._domain_features())
        
        # Кросс-сенсорные признаки
        features.update(self._cross_sensor_features())
        
        return features
    
    def _statistical_features(
        self,
        sensor_name: str,
        values: np.ndarray
    ) -> Dict[str, float]:
        """Статистические признаки"""
        prefix = f"{sensor_name}_"
        
        return {
            f"{prefix}mean": float(np.mean(values)),
            f"{prefix}std": float(np.std(values)),
            f"{prefix}min": float(np.min(values)),
            f"{prefix}max": float(np.max(values)),
            f"{prefix}median": float(np.median(values)),
            f"{prefix}q25": float(np.percentile(values, 25)),
            f"{prefix}q75": float(np.percentile(values, 75)),
            f"{prefix}range": float(np.ptp(values)),
            f"{prefix}cv": float(np.std(values) / (np.mean(values) + 1e-8))
        }
    
    def _temporal_features(
        self,
        sensor_name: str,
        values: np.ndarray
    ) -> Dict[str, float]:
        """Временные признаки (скользящие окна)"""
        prefix = f"{sensor_name}_"
        
        features = {}
        
        # Скользящие средние
        if len(values) >= 5:
            features[f"{prefix}ma_5"] = float(np.mean(values[-5:]))
        if len(values) >= 15:
            features[f"{prefix}ma_15"] = float(np.mean(values[-15:]))
        if len(values) >= 30:
            features[f"{prefix}ma_30"] = float(np.mean(values[-30:]))
        
        # Скользящие стандартные отклонения
        if len(values) >= 10:
            features[f"{prefix}rolling_std_10"] = float(np.std(values[-10:]))
        
        # Разница между последним и средним
        if len(values) > 0:
            features[f"{prefix}diff_from_mean"] = float(values[-1] - np.mean(values))
        
        return features
    
    def _trend_features(
        self,
        sensor_name: str,
        values: np.ndarray
    ) -> Dict[str, float]:
        """Признаки тренда"""
        prefix = f"{sensor_name}_"
        
        features = {}
        
        if len(values) < 3:
            return features
        
        # Линейная регрессия для определения тренда
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        
        features[f"{prefix}trend_slope"] = float(coeffs[0])
        features[f"{prefix}trend_intercept"] = float(coeffs[1])
        
        # Изменение за последние N точек
        if len(values) >= 5:
            features[f"{prefix}change_5"] = float(values[-1] - values[-5])
        if len(values) >= 10:
            features[f"{prefix}change_10"] = float(values[-1] - values[-10])
        
        # Скорость изменения (первая производная)
        if len(values) >= 2:
            derivatives = np.diff(values)
            features[f"{prefix}velocity_mean"] = float(np.mean(derivatives))
            features[f"{prefix}velocity_std"] = float(np.std(derivatives))
        
        # Ускорение (вторая производная)
        if len(values) >= 3:
            second_derivatives = np.diff(np.diff(values))
            features[f"{prefix}acceleration_mean"] = float(np.mean(second_derivatives))
        
        return features
    
    def _domain_features(self) -> Dict[str, float]:
        """Доменные признаки специфичные для производства стекла"""
        features = {}
        
        # Градиент температуры (печь - формование)
        if (len(self.sensor_buffers["furnace_temperature"]) > 0 and
            len(self.sensor_buffers["forming_mold_temperature"]) > 0):
            
            furnace_temp = list(self.sensor_buffers["furnace_temperature"])[-1]
            forming_temp = list(self.sensor_buffers["forming_mold_temperature"])[-1]
            
            if not (np.isnan(furnace_temp) or np.isnan(forming_temp)):
                features["temperature_gradient"] = float(furnace_temp - forming_temp)
        
        # Скорость охлаждения (приблизительная)
        if len(self.sensor_buffers["furnace_temperature"]) >= 2:
            temps = np.array(list(self.sensor_buffers["furnace_temperature"]))
            temps = temps[~np.isnan(temps)]
            
            if len(temps) >= 2:
                cooling_rate = (temps[-1] - temps[-2]) / (1/60)  # °C/мин
                features["cooling_rate"] = float(cooling_rate)
        
        # Индекс вязкости (упрощенная формула)
        if len(self.sensor_buffers["furnace_temperature"]) > 0:
            temp = list(self.sensor_buffers["furnace_temperature"])[-1]
            if not np.isnan(temp) and temp > 0:
                # Упрощенная модель вязкости Аррениуса
                viscosity_index = 1000 / (temp + 273.15)
                features["viscosity_index"] = float(viscosity_index)
        
        # Индекс стабильности процесса (комбинированный показатель)
        stability_scores = []
        for buffer in self.sensor_buffers.values():
            if len(buffer) >= 10:
                values = np.array(list(buffer))
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    cv = np.std(values) / (np.mean(values) + 1e-8)
                    stability_scores.append(1 / (1 + cv))
        
        if stability_scores:
            features["process_stability_index"] = float(np.mean(stability_scores))
        
        # Индекс риска дефектов (эвристика)
        risk_factors = []
        
        # Высокая температура печи
        if len(self.sensor_buffers["furnace_temperature"]) > 0:
            temp = list(self.sensor_buffers["furnace_temperature"])[-1]
            if not np.isnan(temp):
                if temp > 1600:
                    risk_factors.append(0.8)
                elif temp < 1400:
                    risk_factors.append(0.6)
        
        # Высокая скорость формования
        if len(self.sensor_buffers["forming_belt_speed"]) > 0:
            speed = list(self.sensor_buffers["forming_belt_speed"])[-1]
            if not np.isnan(speed):
                if speed > 180:
                    risk_factors.append(0.7)
        
        if risk_factors:
            features["defect_risk_index"] = float(np.mean(risk_factors))
        else:
            features["defect_risk_index"] = 0.3
        
        return features
    
    def _cross_sensor_features(self) -> Dict[str, float]:
        """Кросс-сенсорные признаки (корреляции между датчиками)"""
        features = {}
        
        # Корреляция температура печи - давление
        if (len(self.sensor_buffers["furnace_temperature"]) >= 10 and
            len(self.sensor_buffers["furnace_pressure"]) >= 10):
            
            temp = np.array(list(self.sensor_buffers["furnace_temperature"]))
            pressure = np.array(list(self.sensor_buffers["furnace_pressure"]))
            
            valid_mask = ~(np.isnan(temp) | np.isnan(pressure))
            if valid_mask.sum() >= 10:
                corr = np.corrcoef(temp[valid_mask], pressure[valid_mask])[0, 1]
                features["furnace_temp_pressure_corr"] = float(corr)
        
        # Соотношение скорости формования и температуры
        if (len(self.sensor_buffers["forming_belt_speed"]) > 0 and
            len(self.sensor_buffers["forming_mold_temperature"]) > 0):
            
            speed = list(self.sensor_buffers["forming_belt_speed"])[-1]
            temp = list(self.sensor_buffers["forming_mold_temperature"])[-1]
            
            if not (np.isnan(speed) or np.isnan(temp)) and temp > 0:
                features["speed_temp_ratio"] = float(speed / temp)
        
        return features
    
    def get_feature_vector(self) -> Optional[np.ndarray]:
        """Получение вектора признаков для ML модели"""
        features = self.extract_features()
        
        if not features:
            return None
        
        # Сортировка ключей для консистентности
        sorted_keys = sorted(features.keys())
        vector = np.array([features[k] for k in sorted_keys])
        
        return vector
    
    def get_feature_names(self) -> List[str]:
        """Получение имен признаков"""
        features = self.extract_features()
        return sorted(features.keys())


class BatchFeatureExtractor:
    """Batch извлечение признаков из DataFrame"""
    
    @staticmethod
    def extract_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение признаков из DataFrame с историческими данными"""
        
        # Сортировка по времени
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        features_list = []
        extractor = RealTimeFeatureExtractor(window_size=60)
        
        for idx, row in df.iterrows():
            sensor_data = {
                "timestamp": row["timestamp"],
                "sensors": {
                    "furnace": {
                        "temperature": row.get("furnace_temperature"),
                        "pressure": row.get("furnace_pressure")
                    },
                    "forming": {
                        "belt_speed": row.get("forming_belt_speed"),
                        "mold_temperature": row.get("forming_mold_temperature")
                    }
                }
            }
            
            extractor.update(sensor_data)
            
            if idx >= 30:  # Начинаем извлекать признаки после накопления данных
                features = extractor.extract_features()
                features["timestamp"] = row["timestamp"]
                features_list.append(features)
        
        return pd.DataFrame(features_list)


def main_example():
    """Пример использования"""
    import random
    
    extractor = RealTimeFeatureExtractor(window_size=60)
    
    # Симуляция потока данных
    for i in range(100):
        sensor_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "sensors": {
                "furnace": {
                    "temperature": 1500 + random.uniform(-50, 50),
                    "pressure": 15 + random.uniform(-2, 2),
                    "melt_level": 2500 + random.uniform(-100, 100)
                },
                "forming": {
                    "belt_speed": 150 + random.uniform(-10, 10),
                    "mold_temperature": 320 + random.uniform(-20, 20)
                }
            }
        }
        
        extractor.update(sensor_data)
        
        if i >= 30 and i % 10 == 0:
            features = extractor.extract_features()
            logger.info(f"📊 Извлечено {len(features)} признаков")
            
            # Показываем некоторые признаки
            for key in list(features.keys())[:5]:
                logger.info(f"  {key}: {features[key]:.2f}")


if __name__ == "__main__":
    main_example()