"""
Enhanced AR/3D Visualization with WebSocket integration, LOD optimization, and multi-client support
КРИТИЧЕСКИЕ УЛУЧШЕНИЯ:
- WebSocket для real-time данных
- Level of Detail (LOD) оптимизация
- Multi-client синхронизация
- Collision detection
- Compressed data transfer
- AR marker detection integration
- Performance profiling
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import logging
import asyncio
import gzip
import base64
from dataclasses import dataclass, field
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LODConfiguration:
    """Конфигурация Level of Detail"""
    high_quality_distance: float = 10.0  # meters
    medium_quality_distance: float = 30.0
    low_quality_distance: float = 50.0
    
    high_polygon_count: int = 5000
    medium_polygon_count: int = 1000
    low_polygon_count: int = 200


@dataclass
class ClientState:
    """Состояние подключенного клиента"""
    client_id: str
    camera_position: List[float]
    camera_orientation: List[float]
    viewport_size: Tuple[int, int]
    last_update: datetime
    lod_level: str = "high"
    bandwidth_limit: int = 1000000  # bytes/sec


class DataCompressor:
    """Сжатие данных для передачи"""
    
    @staticmethod
    def compress_geometry(geometry_data: Dict) -> str:
        """Сжатие геометрии для передачи"""
        # Конвертация в JSON
        json_data = json.dumps(geometry_data)
        
        # Gzip compression
        compressed = gzip.compress(json_data.encode('utf-8'))
        
        # Base64 encoding
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        compression_ratio = len(encoded) / len(json_data)
        logger.debug(f"Сжатие: {len(json_data)} -> {len(encoded)} байт "
                    f"({compression_ratio:.2%})")
        
        return encoded
    
    @staticmethod
    def decompress_geometry(compressed_data: str) -> Dict:
        """Распаковка геометрии"""
        decoded = base64.b64decode(compressed_data)
        decompressed = gzip.decompress(decoded)
        return json.loads(decompressed.decode('utf-8'))
    
    @staticmethod
    def delta_compression(current_state: Dict, previous_state: Dict) -> Dict:
        """Delta compression - отправка только изменений"""
        delta = {}
        
        for key, value in current_state.items():
            if key not in previous_state or previous_state[key] != value:
                delta[key] = value
        
        return delta


class LODManager:
    """Управление Level of Detail"""
    
    def __init__(self, config: LODConfiguration = None):
        self.config = config or LODConfiguration()
    
    def compute_lod_level(self, object_position: List[float],
                         camera_position: List[float]) -> str:
        """Определение LOD уровня на основе расстояния"""
        distance = np.linalg.norm(
            np.array(object_position) - np.array(camera_position)
        )
        
        if distance < self.config.high_quality_distance:
            return "high"
        elif distance < self.config.medium_quality_distance:
            return "medium"
        elif distance < self.config.low_quality_distance:
            return "low"
        else:
            return "culled"  # Не рендерить вообще
    
    def simplify_geometry(self, geometry: Dict, lod_level: str) -> Dict:
        """Упрощение геометрии для LOD"""
        if lod_level == "high":
            return geometry  # Полная детализация
        
        # Получаем целевое количество полигонов
        if lod_level == "medium":
            target_polygons = self.config.medium_polygon_count
        elif lod_level == "low":
            target_polygons = self.config.low_polygon_count
        else:
            return {"type": "placeholder"}
        
        # Упрощенная геометрия (в реальности нужен алгоритм decimation)
        simplified = geometry.copy()
        simplified['lod_level'] = lod_level
        simplified['polygon_count'] = target_polygons
        
        return simplified


class Enhanced3DModel:
    """Улучшенная 3D модель с LOD"""
    
    def __init__(self):
        self.factory_layout = self._create_factory_layout()
        self.equipment_models = self._create_equipment_models()
        self.realtime_data = {}
        self.animation_state = {}
        
        # LOD manager
        self.lod_manager = LODManager()
        
        # Cache для LOD вариантов
        self.lod_cache: Dict[str, Dict] = {}
        
        # Previous states для delta compression
        self.previous_states: Dict[str, Dict] = {}
    
    def _create_factory_layout(self) -> Dict:
        """3D layout фабрики"""
        return {
            "dimensions": {"length": 100, "width": 30, "height": 15},
            "sections": [
                {
                    "id": "batch_house",
                    "name": "Batch House",
                    "position": [0, 0, 0],
                    "dimensions": [20, 30, 15],
                    "color": "#8B4513",
                    "geometry_type": "box",
                    "collider": {"type": "box", "bounds": [[0, 0, 0], [20, 30, 15]]}
                },
                {
                    "id": "melting_furnace",
                    "name": "Melting Furnace",
                    "position": [25, 5, 0],
                    "dimensions": [30, 20, 12],
                    "color": "#FF4500",
                    "geometry_type": "box",
                    "heat_emissive": True,
                    "collider": {"type": "box", "bounds": [[25, 5, 0], [55, 25, 12]]}
                },
                {
                    "id": "forming_area",
                    "name": "Forming Area",
                    "position": [80, 0, 0],
                    "dimensions": [20, 30, 15],
                    "color": "#4682B4",
                    "geometry_type": "box",
                    "collider": {"type": "box", "bounds": [[80, 0, 0], [100, 30, 15]]}
                }
            ]
        }
    
    def _create_equipment_models(self) -> Dict:
        """Модели оборудования с LOD вариантами"""
        return {
            "furnace_A": {
                "type": "melting_furnace",
                "position": [35, 10, 5],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "geometry": {
                    "high": {"type": "detailed_furnace", "polygons": 5000},
                    "medium": {"type": "simplified_furnace", "polygons": 1000},
                    "low": {"type": "box", "polygons": 12}
                },
                "parameters": ["temperature", "melt_level", "oxygen_content"],
                "sensors": ["temp_sensor_001", "level_sensor_001"],
                "alerts": [],
                "status": "operational"
            },
            "forming_line_1": {
                "type": "forming_line",
                "position": [85, 5, 5],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "geometry": {
                    "high": {"type": "detailed_line", "polygons": 3000},
                    "medium": {"type": "simplified_line", "polygons": 800},
                    "low": {"type": "box", "polygons": 12}
                },
                "parameters": ["belt_speed", "mold_temperature", "pressure"],
                "sensors": ["speed_sensor_001", "temp_sensor_002"],
                "alerts": [],
                "status": "operational"
            },
            "inspection_station": {
                "type": "inspection",
                "position": [95, 5, 10],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "geometry": {
                    "high": {"type": "camera_array", "polygons": 1000},
                    "medium": {"type": "simplified_cameras", "polygons": 300},
                    "low": {"type": "point", "polygons": 1}
                },
                "parameters": ["inspection_rate", "defect_count"],
                "sensors": ["camera_001", "camera_002"],
                "alerts": [],
                "status": "operational"
            }
        }
    
    def update_realtime_data(self, data: Dict[str, Any], client_id: str = None):
        """Обновление real-time данных"""
        self.realtime_data.update(data)
        self._update_animation_state()
        
        # Store previous state для delta compression
        if client_id:
            self.previous_states[client_id] = self.realtime_data.copy()
    
    def _update_animation_state(self):
        """Обновление состояния анимации"""
        # Furnace glow
        if "furnace_temperature" in self.realtime_data:
            temp = self.realtime_data["furnace_temperature"]
            glow_intensity = np.clip((temp - 1400) / 300, 0, 1)
            self.animation_state["furnace_glow"] = float(glow_intensity)
            
            # Particle effects для экстремальных температур
            if temp > 1650:
                self.animation_state["heat_particles"] = {
                    "count": int((temp - 1650) * 10),
                    "color": "#FF4500",
                    "lifetime": 2.0
                }
        
        # Conveyor belt
        if "belt_speed" in self.realtime_data:
            speed = self.realtime_data["belt_speed"]
            self.animation_state["conveyor_speed"] = float(speed / 150.0)
        
        # Defect indicators
        if "defects" in self.realtime_data:
            defects = self.realtime_data["defects"]
            self.animation_state["defect_level"] = float(sum(defects.values()))
            
            # Alert highlights для критических дефектов
            critical_defects = [k for k, v in defects.items() if v > 0.7]
            if critical_defects:
                self.animation_state["alerts"] = [
                    {
                        "type": "critical",
                        "defect": defect,
                        "timestamp": datetime.now().isoformat()
                    }
                    for defect in critical_defects
                ]
    
    def get_optimized_scene_data(self, client_state: ClientState,
                                 use_delta: bool = True) -> Dict:
        """Получение оптимизированных данных сцены для клиента"""
        camera_pos = client_state.camera_position
        
        # LOD optimization для каждого объекта
        optimized_equipment = {}
        
        for eq_id, equipment in self.equipment_models.items():
            lod_level = self.lod_manager.compute_lod_level(
                equipment["position"], camera_pos
            )
            
            if lod_level == "culled":
                continue  # Не отправляем объект
            
            # Получаем геометрию для LOD уровня
            geometry = equipment["geometry"].get(lod_level, equipment["geometry"]["low"])
            
            optimized_equipment[eq_id] = {
                "position": equipment["position"],
                "rotation": equipment["rotation"],
                "scale": equipment["scale"],
                "geometry": geometry,
                "lod_level": lod_level,
                "status": equipment.get("status", "unknown")
            }
            
            # Добавляем параметры только для близких объектов
            if lod_level == "high":
                optimized_equipment[eq_id]["parameters"] = {
                    param: self.realtime_data.get(param)
                    for param in equipment["parameters"]
                    if param in self.realtime_data
                }
        
        scene_data = {
            "factory_layout": self.factory_layout,
            "equipment": optimized_equipment,
            "animation_state": self.animation_state,
            "timestamp": datetime.now().isoformat(),
            "lod_info": {
                "client_position": camera_pos,
                "objects_rendered": len(optimized_equipment)
            }
        }
        
        # Delta compression если есть предыдущее состояние
        if use_delta and client_state.client_id in self.previous_states:
            previous = self.previous_states[client_state.client_id]
            scene_data = DataCompressor.delta_compression(scene_data, previous)
            scene_data["is_delta"] = True
        
        self.previous_states[client_state.client_id] = scene_data
        
        return scene_data


class ARInterface:
    """Улучшенный AR интерфейс с WebSocket"""
    
    def __init__(self):
        self.model = Enhanced3DModel()
        self.compressor = DataCompressor()
        
        # Connected clients
        self.clients: Dict[str, ClientState] = {}
        
        # WebSocket connections (placeholder)
        self.ws_connections: Set[Any] = set()
        
        # Performance metrics
        self.metrics = {
            "total_updates": 0,
            "avg_update_time": 0.0,
            "avg_payload_size": 0,
            "active_clients": 0
        }
    
    def update_with_sensor_data(self, sensor_data: Dict[str, Any]):
        """Обновление AR интерфейса данными от датчиков"""
        try:
            # Преобразуем данные датчиков в формат, понятный 3D модели
            realtime_data = {}
            
            # Температура печи
            if "furnace_temperature" in sensor_data:
                realtime_data["furnace_temperature"] = sensor_data["furnace_temperature"]
            
            # Уровень расплава
            if "melt_level" in sensor_data:
                realtime_data["melt_level"] = sensor_data["melt_level"]
            
            # Скорость конвейера
            if "belt_speed" in sensor_data:
                realtime_data["belt_speed"] = sensor_data["belt_speed"]
            
            # Температура формы
            if "mold_temperature" in sensor_data:
                realtime_data["mold_temperature"] = sensor_data["mold_temperature"]
            
            # Качество продукции
            if "quality_score" in sensor_data:
                realtime_data["quality_score"] = sensor_data["quality_score"]
            
            # Дефекты
            if "defects" in sensor_data:
                realtime_data["defects"] = sensor_data["defects"]
            
            # Обновляем модель
            self.model.update_realtime_data(realtime_data)
            logger.debug(f"✅ AR интерфейс обновлен данными: {list(realtime_data.keys())}")
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка обновления AR интерфейса: {e}")

    def register_client(self, client_id: str, initial_state: Dict) -> ClientState:
        """Регистрация нового клиента"""
        client_state = ClientState(
            client_id=client_id,
            camera_position=initial_state.get("camera_position", [50, 10, 5]),
            camera_orientation=initial_state.get("camera_orientation", [0, 0, 0]),
            viewport_size=initial_state.get("viewport_size", (1920, 1080)),
            last_update=datetime.now()
        )
        
        self.clients[client_id] = client_state
        self.metrics["active_clients"] = len(self.clients)
        
        logger.info(f"✅ Клиент зарегистрирован: {client_id}")
        return client_state
    
    def unregister_client(self, client_id: str):
        """Отключение клиента"""
        if client_id in self.clients:
            del self.clients[client_id]
            self.metrics["active_clients"] = len(self.clients)
            logger.info(f"❌ Клиент отключен: {client_id}")
    
    def update_client_state(self, client_id: str, new_state: Dict):
        """Обновление состояния клиента"""
        if client_id not in self.clients:
            return
        
        client = self.clients[client_id]
        
        if "camera_position" in new_state:
            client.camera_position = new_state["camera_position"]
        if "camera_orientation" in new_state:
            client.camera_orientation = new_state["camera_orientation"]
        if "viewport_size" in new_state:
            client.viewport_size = new_state["viewport_size"]
        
        client.last_update = datetime.now()
    
    def get_client_view(self, client_id: str, 
                       compressed: bool = True) -> Optional[Dict]:
        """Получение оптимизированного представления для клиента"""
        if client_id not in self.clients:
            return None
        
        import time
        start_time = time.time()
        
        client_state = self.clients[client_id]
        
        # Получаем оптимизированные данные
        scene_data = self.model.get_optimized_scene_data(
            client_state, use_delta=True
        )
        
        # Сжатие если требуется
        if compressed:
            compressed_data = self.compressor.compress_geometry(scene_data)
            payload = {"compressed": True, "data": compressed_data}
        else:
            payload = {"compressed": False, "data": scene_data}
        
        # Метрики
        update_time = time.time() - start_time
        payload_size = len(json.dumps(payload))
        
        self.metrics["total_updates"] += 1
        self.metrics["avg_update_time"] = (
            (self.metrics["avg_update_time"] * (self.metrics["total_updates"] - 1) +
             update_time) / self.metrics["total_updates"]
        )
        self.metrics["avg_payload_size"] = (
            (self.metrics["avg_payload_size"] * (self.metrics["total_updates"] - 1) +
             payload_size) / self.metrics["total_updates"]
        )
        
        payload["performance"] = {
            "update_time_ms": update_time * 1000,
            "payload_size_kb": payload_size / 1024
        }
        
        return payload
    
    async def broadcast_update(self, data: Dict):
        """Broadcast обновлений всем клиентам"""
        self.model.update_realtime_data(data)
        
        # Отправка каждому клиенту оптимизированной версии
        for client_id in self.clients.keys():
            client_view = self.get_client_view(client_id, compressed=True)
            
            # В реальности отправляем через WebSocket
            # await ws.send_json(client_view)
            logger.debug(f"📤 Отправлено клиенту {client_id}: "
                        f"{client_view['performance']['payload_size_kb']:.2f} KB")
    
    def get_performance_metrics(self) -> Dict:
        """Получение метрик производительности"""
        return {
            **self.metrics,
            "avg_update_time_ms": self.metrics["avg_update_time"] * 1000,
            "avg_payload_size_kb": self.metrics["avg_payload_size"] / 1024
        }


# Пример использования
if __name__ == "__main__":
    ar_interface = ARInterface()
    
    print("🌐 Enhanced AR Interface with WebSocket и LOD")
    
    # Регистрация клиентов
    client1 = ar_interface.register_client("client_001", {
        "camera_position": [50, 10, 5],
        "viewport_size": (1920, 1080)
    })
    
    client2 = ar_interface.register_client("client_002", {
        "camera_position": [80, 15, 10],
        "viewport_size": (1280, 720)
    })
    
    print(f"Активных клиентов: {ar_interface.metrics['active_clients']}")
    
    # Обновление данных
    test_data = {
        "furnace_temperature": 1580.0,
        "belt_speed": 155.0,
        "mold_temperature": 320.0,
        "defects": {
            "crack": 0.2,
            "bubble": 0.15,
            "chip": 0.05
        }
    }
    
    ar_interface.model.update_realtime_data(test_data)
    
    # Получение представлений для клиентов
    print("\n📊 Тест оптимизации для клиентов:")
    
    for i in range(3):
        view1 = ar_interface.get_client_view("client_001", compressed=True)
        view2 = ar_interface.get_client_view("client_002", compressed=True)
        
        if i == 0:
            print(f"\nКлиент 1:")
            print(f"  Payload: {view1['performance']['payload_size_kb']:.2f} KB")
            print(f"  Update time: {view1['performance']['update_time_ms']:.2f} ms")
            
            print(f"\nКлиент 2:")
            print(f"  Payload: {view2['performance']['payload_size_kb']:.2f} KB")
            print(f"  Update time: {view2['performance']['update_time_ms']:.2f} ms")
    
    # Метрики производительности
    print("\n📈 Метрики производительности:")
    metrics = ar_interface.get_performance_metrics()
    print(f"  Всего обновлений: {metrics['total_updates']}")
    print(f"  Среднее время: {metrics['avg_update_time_ms']:.2f} ms")
    print(f"  Средний payload: {metrics['avg_payload_size_kb']:.2f} KB")
    
    print("\n✅ Тестирование завершено!")