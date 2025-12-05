"""
OPC UA клиент для сбора данных с промышленного оборудования
Поддерживает асинхронное чтение, подписки и обработку событий
"""

import asyncio
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime
from asyncua import Client, Node, ua
from asyncua.common.subscription import DataChangeNotif
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OPCUAClient:
    """Асинхронный OPC UA клиент для производственных датчиков"""
    
    def __init__(
        self, 
        server_url: str = "opc.tcp://localhost:4840",
        namespace: str = "http://glass.factory/UA/",
        callback: Optional[Callable] = None
    ):
        self.server_url = server_url
        self.namespace = namespace
        self.client: Optional[Client] = None
        self.subscription = None
        self.callback = callback
        self.nodes: Dict[str, Node] = {}
        self.running = False
        
        # Определение тегов датчиков согласно документации
        self.sensor_tags = {
            # Печь
            "furnace_temp": "MIK1.Furnace.Temperature",
            "furnace_pressure": "MIK1.Furnace.Pressure",
            "furnace_level": "MIK1.Furnace.MeltLevel",
            "furnace_o2": "MIK1.Furnace.O2_Percent",
            "furnace_co2": "MIK1.Furnace.CO2_Percent",
            
            # Формирование
            "forming_temp": "MIK1.Forming.MoldTemperature",
            "forming_pressure": "MIK1.Forming.Pressure",
            "forming_speed": "MIK1.Forming.BeltSpeed",
            
            # Отжиг
            "annealing_temp": "MIK1.Annealing.Temperature",
            
            # Процесс
            "batch_flow": "MIK1.Process.BatchFlow",
            
            # Качество (МИК-1)
            "defect_count": "MIK1.Quality.DefectCount",
            "defect_types": "MIK1.Quality.DefectTypes"
        }
    
    async def connect(self) -> bool:
        """Подключение к OPC UA серверу"""
        try:
            self.client = Client(url=self.server_url)
            await self.client.connect()
            
            # Verify connection by accessing namespace array
            try:
                ns_array = await self.client.get_namespace_array()
                logger.info(f"✅ Подключено к OPC UA серверу: {self.server_url}")
                logger.info(f"📋 Namespace array: {ns_array}")
                
                # Try to get namespace index, but handle if it doesn't exist
                try:
                    nsidx = await self.client.get_namespace_index(self.namespace)
                    logger.info(f"📋 Namespace index: {nsidx}")
                except Exception:
                    logger.warning(f"⚠️ Namespace '{self.namespace}' not found, using default namespace")
                    nsidx = 1  # Use default namespace index
                
                # Инициализация узлов
                await self._initialize_nodes(nsidx)
                
                return True
            except Exception as verify_error:
                logger.error(f"❌ Ошибка проверки подключения к OPC UA: {verify_error}")
                await self.client.disconnect()
                return False
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к OPC UA: {e}")
            return False
    
    async def _initialize_nodes(self, nsidx: int):
        """Инициализация узлов датчиков"""
        root = self.client.nodes.root
        
        for sensor_id, tag_path in self.sensor_tags.items():
            try:
                # Парсинг пути тега
                path_parts = tag_path.split('.')
                node = root
                
                # Навигация по дереву OPC UA
                for part in path_parts:
                    children = await node.get_children()
                    found = False
                    for child in children:
                        browse_name = await child.read_browse_name()
                        if browse_name.Name == part:
                            node = child
                            found = True
                            break
                    
                    if not found:
                        logger.warning(f"⚠️ Узел не найден: {part} в {tag_path}")
                        break
                
                if found:
                    self.nodes[sensor_id] = node
                    logger.info(f"✅ Узел инициализирован: {sensor_id} -> {tag_path}")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка инициализации узла {sensor_id}: {e}")
    
    async def read_sensor_data(self) -> Dict:
        """Синхронное чтение всех датчиков"""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "production_line": "Line_A",
            "sensors": {}
        }
        
        for sensor_id, node in self.nodes.items():
            try:
                value = await node.read_value()
                data["sensors"][sensor_id] = {
                    "value": float(value) if isinstance(value, (int, float)) else value,
                    "status": "OK"
                }
            except Exception as e:
                logger.error(f"❌ Ошибка чтения {sensor_id}: {e}")
                data["sensors"][sensor_id] = {
                    "value": None,
                    "status": "ERROR"
                }
        
        return data
    
    async def subscribe_to_changes(self, interval: int = 1000):
        """Подписка на изменения данных (interval в мс)"""
        if not self.client:
            logger.error("❌ Клиент не подключен")
            return
        
        # Check if client is actually connected by trying to access a property
        try:
            # Try to access a basic property to check if client is connected
            _ = await self.client.get_namespace_array()
        except Exception as e:
            logger.error(f"❌ Клиент не подключен к серверу: {e}")
            return
        
        try:
            # Создание подписки
            self.subscription = await self.client.create_subscription(
                period=interval,
                handler=DataChangeHandler(self.callback)
            )
            
            # Подписка на все узлы
            nodes_to_subscribe = list(self.nodes.values())
            if nodes_to_subscribe:
                await self.subscription.subscribe_data_change(nodes_to_subscribe)
                logger.info(f"✅ Подписка создана на {len(nodes_to_subscribe)} узлов")
            else:
                logger.warning("⚠️ Нет узлов для подписки")
            
            self.running = True
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания подписки: {e}")
            # Continue with polling as fallback
    
    async def start_polling(self, interval_seconds: int = 60):
        """Запуск циклического опроса датчиков"""
        self.running = True
        logger.info(f"🔄 Запуск опроса каждые {interval_seconds}с")
        
        while self.running:
            try:
                # Check if client is still valid before reading
                if self.client:
                    try:
                        _ = await self.client.get_namespace_array()
                        data = await self.read_sensor_data()
                        
                        if self.callback:
                            await self.callback(data)
                    except Exception as e:
                        logger.error(f"❌ Ошибка чтения данных: {e}")
                        # Try to reconnect
                        await self.connect()
                else:
                    logger.warning("⚠️ OPC UA клиент не инициализирован")
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("⏹️ Опрос остановлен")
                break
            except Exception as e:
                logger.error(f"❌ Ошибка в цикле опроса: {e}")
                await asyncio.sleep(5)
    
    async def disconnect(self):
        """Отключение от сервера"""
        self.running = False
        
        if self.subscription:
            try:
                await self.subscription.delete()
                logger.info("✅ Подписка удалена")
            except Exception as e:
                logger.error(f"❌ Ошибка удаления подписки: {e}")
        
        if self.client:
            try:
                await self.client.disconnect()
                logger.info("✅ Отключено от OPC UA сервера")
            except Exception as e:
                logger.error(f"❌ Ошибка отключения от OPC UA сервера: {e}")


class DataChangeHandler:
    """Обработчик изменений данных OPC UA"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
    
    def datachange_notification(self, node: Node, val, data: DataChangeNotif):
        """Вызывается при изменении данных"""
        try:
            change_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "node_id": str(node),
                "value": val,
                "status": str(data.monitored_item.Value.StatusCode)
            }
            
            logger.debug(f"📊 Изменение: {change_data}")
            
            if self.callback:
                asyncio.create_task(self.callback(change_data))
                
        except Exception as e:
            logger.error(f"❌ Ошибка обработки изменения: {e}")


async def simulate_opc_ua_server():
    """Симулятор OPC UA сервера для тестирования"""
    from asyncua import Server
    import random
    
    server = Server()
    await server.init()
    server.set_endpoint("opc.tcp://0.0.0.0:4840")
    
    # Регистрация namespace
    uri = "http://glass.factory/UA/"
    nsidx = await server.register_namespace(uri)
    
    # Создание структуры узлов
    objects = server.nodes.objects
    
    # Печь
    furnace = await objects.add_folder(nsidx, "Furnace")
    furnace_temp = await furnace.add_variable(nsidx, "Temperature", 1500.0)
    furnace_pressure = await furnace.add_variable(nsidx, "Pressure", 15.0)
    furnace_level = await furnace.add_variable(nsidx, "MeltLevel", 2500.0)
    
    # Формирование
    forming = await objects.add_folder(nsidx, "Forming")
    forming_temp = await forming.add_variable(nsidx, "MoldTemperature", 320.0)
    forming_speed = await forming.add_variable(nsidx, "BeltSpeed", 150.0)
    
    # Качество
    quality = await objects.add_folder(nsidx, "Quality")
    defect_count = await quality.add_variable(nsidx, "DefectCount", 0)
    
    # Разрешение записи
    await furnace_temp.set_writable()
    await furnace_pressure.set_writable()
    await forming_speed.set_writable()
    
    logger.info("🚀 Симулятор OPC UA сервера запущен на opc.tcp://0.0.0.0:4840")
    
    async with server:
        # Симуляция изменения значений
        while True:
            try:
                # Симуляция колебаний температуры
                temp = 1500 + random.uniform(-50, 50)
                await furnace_temp.write_value(temp)
                
                # Симуляция давления
                pressure = 15 + random.uniform(-2, 2)
                await furnace_pressure.write_value(pressure)
                
                # Симуляция дефектов
                defects = random.randint(0, 5)
                await defect_count.write_value(defects)
                
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break


async def main_example():
    """Пример использования клиента"""
    
    async def data_callback(data):
        """Callback для обработки данных"""
        print(f"📡 Получены данные: {json.dumps(data, indent=2)}")
    
    # Создание клиента
    client = OPCUAClient(
        server_url="opc.tcp://localhost:4840",
        callback=data_callback
    )
    
    # Подключение
    if await client.connect():
        try:
            # Запуск подписки или опроса
            # await client.subscribe_to_changes(interval=1000)
            await client.start_polling(interval_seconds=10)
            
        except KeyboardInterrupt:
            logger.info("⏹️ Остановка по Ctrl+C")
        finally:
            await client.disconnect()


if __name__ == "__main__":
    # Для тестирования можно запустить симулятор сервера:
    asyncio.run(simulate_opc_ua_server())
    
    # Или клиента:
    #asyncio.run(main_example())