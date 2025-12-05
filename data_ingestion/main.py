"""
Main Entry Point for Data Ingestion System
Orchestrates data collection, routing, and processing for glass production
"""

import asyncio
import signal
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import system components
from data_ingestion.data_collector import DataCollector
from data_ingestion.data_router import DataRouter, DataBuffer
from data_ingestion.setup import DataIngestionSetup
from storage.influxdb_client import GlassInfluxDBClient

# Configure logging with UTF-8 encoding support for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataIngestionSystem:
    """Основная система сбора данных"""
    
    def __init__(self, config_file: str = "data_ingestion_config.json"):
        self.config_file = config_file
        self.setup = DataIngestionSetup(config_file=config_file)
        self.config = self.setup.load_config()
        self.collector: Optional[DataCollector] = None
        self.data_router: Optional[DataRouter] = None
        self.influxdb_client: Optional[GlassInfluxDBClient] = None
        self.running = False
        self.collection_task = None
        self.router_task = None
        
    async def initialize_system(self):
        """Инициализация системы сбора данных"""
        try:
            logger.info("🔧 Инициализация системы сбора данных...")
            
            # Load configuration
            self.config = self.setup.load_config()
            
            # Validate configuration
            if not self.setup.validate_config():
                raise Exception("Ошибка валидации конфигурации")
            
            # Setup environment and logging
            self.setup.setup_environment()
            self.setup.setup_logging()
            
            # Initialize components
            await self._initialize_components()
            
            logger.info("✅ Система сбора данных инициализирована")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации системы: {e}")
            raise

    async def _initialize_components(self):
        """Инициализация всех компонентов системы"""
        try:
            # Initialize data router first
            self.data_router = DataRouter()
            await self.data_router.initialize_destinations()
            await self.data_router.start_destinations()
            
            # Initialize data collector with configuration
            self.collector = DataCollector(
                collection_interval=self.config.get("collector", {}).get("collection_interval", 1.0),
                data_callback=self._data_collection_callback,
                config=self.config
            )
            await self.collector.initialize_connectors()
            
            # Initialize InfluxDB client
            self.influxdb_client = GlassInfluxDBClient(
                url=self.config.get("influxdb", {}).get("url"),
                token=self.config.get("influxdb", {}).get("token"),
                org=self.config.get("influxdb", {}).get("org"),
                bucket=self.config.get("influxdb", {}).get("bucket")
            )
            await self.influxdb_client.connect()
            
            logger.info("🔧 Компоненты системы инициализированы")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации компонентов: {e}")
            raise
    
    async def start_system(self):
        """Запуск системы сбора данных"""
        try:
            logger.info("🚀 Запуск системы сбора данных...")
            
            # Connect to data sources
            connection_results = await self.collector.connect_sources()
            connected_sources = sum(1 for result in connection_results.values() if result)
            
            if connected_sources == 0:
                logger.error("❌ Не удалось подключиться ни к одному источнику данных")
                return False
            
            logger.info(f"🔗 Подключено к {connected_sources}/{len(connection_results)} источникам данных")
            
            # Start data collection
            self.collection_task = asyncio.create_task(self.collector.start_collection())
            
            # Start data routing - create a background task for routing
            # Note: route_data() is called with data in the callback, not as a standalone task
            self.router_task = None  # Routing happens in _data_collection_callback            
            self.running = True
            logger.info("✅ Система сбора данных запущена успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска системы: {e}")
            return False
    
    async def _data_collection_callback(self, collected_data: Dict[str, Any]):
        """Обработка собранных данных"""
        try:
            if not self.running or not self.data_router:
                return
            
            # Route data from each source
            for source_name, source_data in collected_data.get("sources", {}).items():
                if source_data:
                    # Determine data type based on source and content
                    data_type = self._determine_data_type(source_name, source_data)
                    
                    # Route the data
                    await self.data_router.route_data(source_data, data_type)
            
            # Log collection summary
            data_points = collected_data.get("data_points", 0)
            sources = list(collected_data.get("sources", {}).keys())
            logger.debug(f"📊 Рутировано {data_points} данных от {sources}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка в коллбэке сбора данных: {e}")
    
    def _determine_data_type(self, source_name: str, data: Dict[str, Any]) -> str:
        """Определение типа данных на основе источника и содержимого"""
        # Source-based determination
        if source_name == "mik1_camera":
            return "image_data"
        elif source_name == "mqtt":
            # MQTT topic-based determination
            topic = data.get("topic", "")
            if "defects" in topic:
                return "defect_data"
            elif "alarms" in topic:
                return "alarm_data"
            elif "quality" in topic:
                return "quality_data"
            elif "control" in topic:
                return "control_data"
            else:
                return "sensor_data"
        else:
            # Content-based determination
            if "defects" in data or "defect_count" in data:
                return "defect_data"
            elif "quality_score" in data:
                return "quality_data"
            elif "frame_id" in data:
                return "image_data"
            else:
                return "sensor_data"
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы"""
        try:
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "running": self.running,
                "components": {}
            }
            
            # Collector status
            if self.collector:
                status["components"]["collector"] = {
                    "sources": await self.collector.get_source_status(),
                    "stats": await self.collector.get_collection_stats()
                }
            
            # Router status
            if self.data_router:
                status["components"]["router"] = {
                    "stats": await self.data_router.get_routing_stats()
                }
            
            # Buffer status
            if self.data_router and self.data_router.destinations["buffer"]:
                status["components"]["buffer"] = (
                    self.data_router.destinations["buffer"].get_buffer_stats()
                )
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения статуса системы: {e}")
            return {"error": str(e)}
    
    async def stop_system(self):
        """Остановка системы сбора данных"""
        try:
            logger.info("🔴 Остановка системы сбора данных...")
            
            self.running = False
            
            # Stop data collection
            if self.collector:
                await self.collector.stop_collection()
            
            # Stop routing destinations
            if self.data_router:
                await self.data_router.stop_destinations()
            
            # Wait for tasks to complete
            if self.collection_task and not self.collection_task.done():
                self.collection_task.cancel()
                await self.collection_task
            
            if self.router_task and not self.router_task.done():
                self.router_task.cancel()
                await self.router_task

            logger.info("✅ Система сбора данных остановлена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка остановки системы: {e}")
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Проверка состояния системы"""
        try:
            health_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "healthy": True,
                "issues": []
            }
            
            # Check collector health
            if self.collector:
                source_status = await self.collector.get_source_status()
                disconnected_sources = [
                    source for source, status in source_status.items() 
                    if not status["connected"]
                ]
                
                if disconnected_sources:
                    health_status["healthy"] = False
                    health_status["issues"].append({
                        "component": "collector",
                        "problem": f"Отключенные источники: {disconnected_sources}",
                        "severity": "HIGH"
                    })
            
            # Check router health
            if self.data_router:
                routing_stats = await self.data_router.get_routing_stats()
                high_error_routes = [
                    route for route, stats in routing_stats.items()
                    if stats["errors"] > 10
                ]
                
                if high_error_routes:
                    health_status["healthy"] = False
                    health_status["issues"].append({
                        "component": "router",
                        "problem": f"Маршруты с ошибками: {high_error_routes}",
                        "severity": "MEDIUM"
                    })
            
            # Check buffer health
            if (self.data_router and 
                self.data_router.destinations["buffer"]):
                buffer_stats = self.data_router.destinations["buffer"].get_buffer_stats()
                if buffer_stats["utilization"] > 0.8:
                    health_status["healthy"] = False
                    health_status["issues"].append({
                        "component": "buffer",
                        "problem": f"Буфер заполнен на {buffer_stats['utilization']:.1%}",
                        "severity": "HIGH"
                    })
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Ошибка проверки состояния системы: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "healthy": False,
                "issues": [{"component": "system", "problem": str(e), "severity": "CRITICAL"}]
            }


async def main():
    """Основная точка входа в систему сбора данных"""
    logger.info("🔬 Система сбора данных для производства стекла")
    logger.info("=" * 50)
    
    # Create system instance
    system = DataIngestionSystem()
    
    # Handle shutdown signals
    def signal_handler(signum, frame):
        logger.info("🛑 Получен сигнал остановки")
        # This will be handled in the main loop
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize system
        await system.initialize_system()
        
        # Start system
        if not await system.start_system():
            logger.error("❌ Не удалось запустить систему")
            return 1
        
        logger.info("🔄 Система сбора данных работает. Нажмите Ctrl+C для остановки.")
        
        # Main loop
        while system.running:
            try:
                # Periodic health checks
                if system.running:
                    health = await system.run_health_check()
                    if not health["healthy"]:
                        logger.warning(f"⚠️ Проблемы с работой системы: {health['issues']}")
                
                # Periodic status reports
                if system.running:
                    status = await system.get_system_status()
                    collector_stats = status.get("components", {}).get("collector", {}).get("stats", {})
                    success_rate = collector_stats.get("success_rate", 0)
                    logger.info(f"📈 Скорость успешной сборки: {success_rate:.1%}")
                
                # Wait before next check
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                logger.info("⏹️ Основной цикл отменен")
                break
            except Exception as e:
                logger.error(f"❌ Ошибка в основном цикле: {e}")
                await asyncio.sleep(5)
    
    except KeyboardInterrupt:
        logger.info("🛑 Система остановлена пользователем")
    except Exception as e:
        logger.error(f"💥 Неожиданная ошибка: {e}")
        return 1
    finally:
        # Graceful shutdown
        await system.stop_system()
    
    logger.info("🏁 Система сбора данных остановлена")
    return 0


# Example usage function
async def run_example():
    """Запуск упрощенного примера для тестирования"""
    logger.info("🧪 Запуск примера системы сбора данных...")
    
    system = DataIngestionSystem()
    
    try:
        # Initialize
        await system.initialize_system()
        
        # Show system is ready
        logger.info("✅ Система инициализирована и готова")
        logger.info("💡 В реальном развертывании это будет подключено к реальным промышленным устройствам")
        logger.info("💡 Для тестирования можно имитировать данные или подключиться к демонстрационным серверам")
        
        # Show system status
        status = await system.get_system_status()
        logger.info(f"📊 Состояние системы: {status.get('running', False)}")
        
        # Run health check
        health = await system.run_health_check()
        logger.info(f"🏥 Здоровье системы: {health.get('healthy', False)}")
        
        # Wait a moment
        await asyncio.sleep(2)
        
    except Exception as e:
        logger.error(f"❌ Ошибка в примере: {e}")
    finally:
        await system.stop_system()


if __name__ == "__main__":
    # Check if running in example mode
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        asyncio.run(run_example())
    else:
        # Run main system
        exit_code = asyncio.run(main())
        sys.exit(exit_code)