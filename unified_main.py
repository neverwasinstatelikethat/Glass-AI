"""
System Integrator for Glass Production Predictive Analytics
Integrates all system components including Phase 3 modules
"""

import os
import sys
import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Add parent directory and models directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(parent_dir, "models")
sys.path.insert(0, parent_dir)
sys.path.insert(0, models_dir)

# Import all system components
from data_ingestion.main import DataIngestionSystem
from models.lstm_predictor.attention_lstm import EnhancedAttentionLSTM as AttentionLSTM
from models.vision_transformer.defect_detector import MultiTaskViT as VisionTransformerDefectDetector
from models.gnn_sensor_network.gnn_model import EnhancedGATSensorGNN as GNNSensorNetwork
from models.ensemble.meta_learner import DiversityRegularizedEnsemble as MetaLearningEnsemble
from knowledge_graph.causal_graph import EnhancedGlassProductionKnowledgeGraph as CausalKnowledgeGraph
from digital_twin.physics_simulation import DigitalTwin as DigitalTwinSimulator
from reinforcement_learning.ppo_optimizer import GlassProductionPPO as RLOptimizer
from explainability.feature_attribution import EnhancedGlassProductionExplainer as Explainer
from visualization.ar_interface import ARInterface as ARVisualizationInterface

# Import inference components
from inference.edge_inference import EdgeModelManager, MultiModelEnsembleInference
# Import database clients
from storage.influxdb_client import GlassInfluxDBClient
from storage.postgres_client import GlassPostgresClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedGlassProductionSystem:
    """Main unified system that integrates data ingestion and AI models"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.data_system: Optional[DataIngestionSystem] = None
        self.model_manager: Optional[EdgeModelManager] = None
        self.ensemble_inference: Optional[MultiModelEnsembleInference] = None
        self.system_integrator: Optional[SystemIntegrator] = None
        self.is_running = False
        self.predictions_cache = {}
        
        # Initialize system components
        self._initialize_system(config_file)
    
    def _detect_environment(self) -> str:
        """Detect runtime environment"""
        env = os.getenv("ENVIRONMENT", "development")
        
        # Check for common environment indicators
        if os.getenv("DOCKER_CONTAINER") or os.getenv("HOSTNAME"):
            env = "docker"
        elif os.getenv("KUBERNETES_SERVICE_HOST"):
            env = "kubernetes"
        elif os.getenv("PRODUCTION", "").lower() in ["true", "1", "yes"]:
            env = "production"
        
        return env
    
    def _initialize_system(self, config_file: Optional[str] = None):
        """Initialize all system components"""
        logger.info("🚀 Initializing Unified Glass Production System...")
        
        # Auto-detect environment and select appropriate config
        if config_file is None:
            environment = self._detect_environment()
            if environment == "docker":
                # Check if Docker-specific config exists
                docker_config_path = "data_ingestion_config_docker.json"
                if os.path.exists(docker_config_path):
                    config_file = docker_config_path
                    logger.info("🐳 Detected Docker environment, using Docker-specific configuration")
                else:
                    config_file = "data_ingestion_config.json"
                    logger.warning("🐳 Detected Docker environment but no Docker config found, using default")
            else:
                config_file = "data_ingestion_config.json"
                logger.info(f"🔧 Using {environment} environment configuration")
        
        # Initialize Data Ingestion System (Phase 1) with correct config
        self.data_system = DataIngestionSystem(config_file=config_file)
        logger.info("✅ Data Ingestion System initialized")
        
        # Initialize AI Models (Phase 2)
        model_configs = {
            "lstm": {
                "path": "models/lstm_predictor/lstm_model.onnx",
                "type": "onnx"
            },
            "vit": {
                "path": "models/vision_transformer/vit_model.onnx",
                "type": "onnx"
            },
            "gnn": {
                "path": "models/gnn_sensor_network/gnn_model.onnx",
                "type": "onnx"
            }
        }
        
        try:
            self.model_manager = EdgeModelManager(model_configs)
            self.ensemble_inference = MultiModelEnsembleInference(self.model_manager)
            logger.info("✅ AI Models and Inference System initialized")
        except Exception as e:
            logger.warning(f"⚠️ AI Models initialization failed: {e}")
            self.model_manager = None
            self.ensemble_inference = None
        
        # Initialize SystemIntegrator
        try:
            self.system_integrator = SystemIntegrator()
            # Set reference to parent system for accessing model_manager
            self.system_integrator._parent_system = self
            logger.info("✅ System Integrator initialized")
        except Exception as e:
            logger.warning(f"⚠️ System Integrator initialization failed: {e}")
            self.system_integrator = None
        
        logger.info("🟢 Unified System initialization complete")
    
    async def initialize_system(self):
        """Initialize the complete system with all components"""
        logger.info("🔄 Initializing complete system...")
        
        try:
            # Initialize data system
            await self.data_system.initialize_system()
            logger.info("✅ Data system initialized")
            
            # Initialize system integrator if available
            if self.system_integrator:
                await self.system_integrator.initialize_system()
                logger.info("✅ System integrator initialized")
            
            return True
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def start_system(self):
        """Start the unified system"""
        logger.info("🟢 Starting Unified Glass Production System...")
        
        try:
            self.is_running = True
            
            # Start data ingestion system
            success = await self.data_system.start_system()
            if not success:
                logger.error("❌ Failed to start data ingestion system")
                return False
            
            logger.info("✅ Unified system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting unified system: {e}")
            await self.shutdown_system()
            return False
    
    async def stop_system(self):
        """Stop the unified system gracefully"""
        logger.info("🔴 Stopping Unified Glass Production System...")
        
        self.is_running = False
        
        # Stop data system
        if self.data_system:
            await self.data_system.stop_system()
        
        logger.info("✅ Unified system stopped")
    
    async def shutdown_system(self):
        """Shutdown the unified system"""
        await self.stop_system()
        logger.info("⏹️ Unified system shutdown complete")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "running": self.is_running,
                "components": {}
            }
            
            # Data system status
            if self.data_system:
                status["components"]["data_system"] = await self.data_system.get_system_status()
            
            # Model system status
            if self.model_manager:
                status["components"]["model_system"] = self.model_manager.get_system_health()
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {"error": str(e)}
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run system health check"""
        try:
            health_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "healthy": True,
                "issues": []
            }
            
            # Check data system health
            if self.data_system:
                data_health = await self.data_system.run_health_check()
                if not data_health.get("healthy", True):
                    health_status["healthy"] = False
                    health_status["issues"].extend(data_health.get("issues", []))
            
            # Check model system health
            if self.model_manager:
                model_health = self.model_manager.get_system_health()
                if model_health.get("system_status") != "healthy":
                    health_status["healthy"] = False
                    health_status["issues"].append({
                        "component": "model_system",
                        "problem": f"Model system status: {model_health.get('system_status')}",
                        "severity": "HIGH"
                    })
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Error running health check: {e}")
            return {"error": str(e)}
    
    async def get_model_predictions(self) -> Dict[str, Any]:
        """Get predictions from all AI models"""
        try:
            if not self.model_manager or not self.ensemble_inference:
                return {"error": "AI models not available"}
            
            # Simulate some input data for demonstration
            # In a real system, this would come from the data ingestion system
            lstm_input = {"input": np.random.randn(1, 30, 20).astype(np.float32)}
            vit_input = {"input": np.random.randn(1, 3, 32, 32).astype(np.float32)}
            
            # Prepare model inputs
            model_inputs = {
                "lstm": lstm_input,
                "vit": vit_input
                # Note: GNN requires graph structure, which is more complex to simulate
            }
            
            # Get individual predictions
            predictions = {}
            for model_name, inputs in model_inputs.items():
                try:
                    output = self.model_manager.predict(model_name, inputs)
                    # Extract the main output (first value in dict or first element in list)
                    if isinstance(output, dict):
                        main_output = list(output.values())[0]
                    else:
                        main_output = output[0] if isinstance(output, list) else output
                    
                    predictions[model_name] = main_output.tolist()
                except Exception as e:
                    logger.warning(f"⚠️ Prediction failed for {model_name}: {e}")
                    predictions[model_name] = None
            
            # Get ensemble prediction
            try:
                ensemble_output, individual_outputs = self.ensemble_inference.predict_with_ensemble(model_inputs)
                predictions["ensemble"] = ensemble_output.tolist()
            except Exception as e:
                logger.warning(f"⚠️ Ensemble prediction failed: {e}")
                predictions["ensemble"] = None
            
            # Cache predictions
            self.predictions_cache = predictions
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "defects": {
                    "crack": float(np.random.random()),
                    "bubble": float(np.random.random()),
                    "chip": float(np.random.random()),
                    "deformation": float(np.random.random()),
                    "cloudiness": float(np.random.random()),
                    "stain": float(np.random.random())
                },
                "confidence": 0.85 + 0.1 * np.random.random(),
                "predictions": predictions,
                "recommendations": [
                    "Monitor furnace temperature",
                    "Check forming belt speed",
                    "Review quality control parameters"
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting model predictions: {e}")
            return {"error": str(e)}
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            if not self.model_manager:
                return {"error": "Model manager not available"}
            
            performance = self.model_manager.get_all_latency_stats()
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "performance": performance
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting model performance: {e}")
            return {"error": str(e)}
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            if not self.model_manager:
                return {"error": "Model manager not available"}
            
            info = self.model_manager.get_model_info()
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "models": info
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting model info: {e}")
            return {"error": str(e)}
    
    # Add missing methods that delegate to system_integrator
    async def predict_defects(self, horizon_hours: int = 1, production_line: str = "Line_A", 
                            include_confidence: bool = True) -> Dict[str, Any]:
        """Прогнозирование дефектов с использованием всех моделей"""
        try:
            if not self.system_integrator:
                raise Exception("Система не инициализирована")
            
            return await self.system_integrator.predict_defects(horizon_hours, production_line, include_confidence)
        except Exception as e:
            logger.error(f"❌ Ошибка прогнозирования: {e}")
            return {"error": str(e)}
    
    async def get_digital_twin_state(self) -> Dict[str, Any]:
        """Get current state from digital twin"""
        try:
            if not self.system_integrator:
                return {"error": "System integrator not available"}
            
            return await self.system_integrator.get_digital_twin_state()
        except Exception as e:
            logger.error(f"❌ Error getting digital twin state: {e}")
            return {"error": str(e)}
    
    async def get_rl_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations from RL agent"""
        try:
            if not self.system_integrator:
                return {"error": "System integrator not available"}
            
            return await self.system_integrator.get_rl_recommendations()
        except Exception as e:
            logger.error(f"❌ Error getting RL recommendations: {e}")
            return {"error": str(e)}
    
    async def get_model_explanations(self) -> Dict[str, Any]:
        """Get explanations for model predictions"""
        try:
            if not self.system_integrator:
                return {"error": "System integrator not available"}
            
            return await self.system_integrator.get_model_explanations()
        except Exception as e:
            logger.error(f"❌ Error getting model explanations: {e}")
            return {"error": str(e)}
    
    async def get_intervention_recommendations(self, defect_type: str, 
                                             parameter_values: Dict[str, float]) -> Dict[str, Any]:
        """Get recommendations for addressing a defect from Knowledge Graph"""
        try:
            if not self.system_integrator:
                return {"error": "System integrator not available"}
            
            return await self.system_integrator.get_intervention_recommendations(defect_type, parameter_values)
        except Exception as e:
            logger.error(f"❌ Error getting intervention recommendations: {e}")
            return {"error": str(e)}
    
    async def get_knowledge_graph_subgraph(self, defect_type: str, max_depth: int = 2,
                                 include_recommendations: bool = True,
                                 include_human_decisions: bool = True) -> Dict[str, Any]:
        """Get subgraph for visualization from Knowledge Graph"""
        try:
            if not self.system_integrator:
                return {"error": "System integrator not available"}
            
            return await self.system_integrator.get_knowledge_graph_subgraph(
                defect_type, max_depth, include_recommendations, include_human_decisions
            )
        except Exception as e:
            logger.error(f"❌ Error getting knowledge graph subgraph: {e}")
            return {"error": str(e)}    
    async def get_causes_of_defect(self, defect_type: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        """Get causes of a defect from Knowledge Graph"""
        try:
            if not self.system_integrator:
                return {"error": "System integrator not available"}
            
            return await self.system_integrator.get_causes_of_defect(defect_type, min_confidence)
        except Exception as e:
            logger.error(f"❌ Error getting causes of defect: {e}")
            return {"error": str(e)}

class SystemIntegrator:
    """Интегратор всех компонентов системы"""
    
    def __init__(self):
        self.initialized = False
        self.data_system: Optional[DataIngestionSystem] = None
        self.ml_models: Dict[str, Any] = {}
        self.knowledge_graph: Optional[CausalKnowledgeGraph] = None
        self.digital_twin: Optional[DigitalTwinSimulator] = None
        self.rl_agent: Optional[RLOptimizer] = None
        self.explainer: Optional[Explainer] = None
        self.ar_interface: Optional[ARVisualizationInterface] = None
        self._latest_sensor_data: Optional[Dict[str, Any]] = None
        self._parent_system: Optional['UnifiedGlassProductionSystem'] = None
        
        # Database clients
        self.influxdb_client: Optional[GlassInfluxDBClient] = None
        self.postgres_client: Optional[GlassPostgresClient] = None
        
        # System state
        self.system_state = {
            "last_prediction_time": None,
            "last_defect_detection": None,
            "current_quality_score": 0.0,
            "active_alerts": [],
            "pending_recommendations": []
        }
    
    async def initialize_system(self):
        """Инициализация всей системы"""
        try:
            logger.info("🔧 Инициализация интегрированной системы...")
            
            # Initialize database connections
            await self._initialize_databases()
            
            # Initialize data ingestion system
            await self._initialize_data_system()
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Initialize Knowledge Graph
            await self._initialize_knowledge_graph()
            
            # Initialize Digital Twin
            await self._initialize_digital_twin()
            
            # Initialize RL Agent
            await self._initialize_rl_agent()
            
            # Initialize Explainer
            await self._initialize_explainer()
            
            # Initialize AR Interface
            await self._initialize_ar_interface()
            
            self.initialized = True
            logger.info("✅ Система полностью инициализирована")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации системы: {e}")
            raise

    async def _initialize_databases(self):
        """Initialize database connections"""
        try:
            # Initialize InfluxDB client
            self.influxdb_client = GlassInfluxDBClient()
            await self.influxdb_client.connect()
            logger.info("✅ InfluxDB client initialized")
            
            # Initialize PostgreSQL client with connection URL from environment
            postgres_url = os.getenv("POSTGRES_URL")
            if postgres_url:
                self.postgres_client = GlassPostgresClient(connection_url=postgres_url)
            else:
                self.postgres_client = GlassPostgresClient()
            await self.postgres_client.connect()
            logger.info("✅ PostgreSQL client initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            # Continue without databases if initialization fails
            pass

    async def _initialize_data_system(self):
        """Initialize data ingestion system"""
        try:
            self.data_system = DataIngestionSystem()
            await self.data_system.initialize_system()
            logger.info("✅ Data ingestion system initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing data ingestion system: {e}")
            raise

    async def _initialize_ml_models(self):
        """Initialize ML models"""
        try:
            # Initialize LSTM model with proper parameters
            lstm_model = AttentionLSTM(
                input_size=20,
                hidden_size=64,
                num_layers=2,
                output_size=6,
                dropout=0.2
            )
            self.ml_models["lstm"] = lstm_model
            logger.info("✅ LSTM model initialized")
            
            # Initialize Vision Transformer model with proper parameters
            vit_model = VisionTransformerDefectDetector(
                img_size=224,
                patch_size=16,
                in_channels=3,
                n_classes=7,
                embed_dim=384,
                depth=6
            )
            self.ml_models["vit"] = vit_model
            logger.info("✅ Vision Transformer model initialized")
            
            # Initialize GNN model with proper parameters
            gnn_model = GNNSensorNetwork(
                num_sensors=20,
                input_dim=1,
                hidden_dim=64,
                output_dim=32,
                edge_dim=3,
                num_layers=3
            )
            self.ml_models["gnn"] = gnn_model
            logger.info("✅ GNN model initialized")
            
            # Initialize ensemble model using factory function with correct parameters
            # The ensemble expects output dimensions of each model, not the models themselves
            from models.ensemble.meta_learner import create_diversity_ensemble
            ensemble_model = create_diversity_ensemble(
                model_outputs=[6, 7, 32],  # Output dimensions of LSTM, ViT, and GNN models
                n_classes=7,  # Number of defect classes
                diversity_weight=0.1
            )
            self.ml_models["ensemble"] = ensemble_model
            logger.info("✅ Ensemble model initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing ML models: {e}")
            raise

    async def _initialize_knowledge_graph(self):
        """Initialize Knowledge Graph"""
        try:
            # Initialize with environment variables for Docker compatibility
            import os
            
            # Check if we're running in Docker
            is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
            
            if is_docker:
                # In Docker, use Docker-specific defaults
                neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
                neo4j_user = os.getenv("NEO4J_USER", "neo4j")
                neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
                redis_host = os.getenv("REDIS_HOST", "redis")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
            else:
                # Outside Docker, use localhost defaults
                neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                neo4j_user = os.getenv("NEO4J_USER", "neo4j")
                neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
                redis_host = os.getenv("REDIS_HOST", "localhost")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
            
            self.knowledge_graph = CausalKnowledgeGraph(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                redis_host=redis_host,
                redis_port=redis_port
            )
            # Initialize knowledge base (this handles connection failures gracefully)
            self.knowledge_graph.initialize_knowledge_base()
            logger.info("✅ Knowledge Graph initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Knowledge Graph initialization failed: {e}")
            # Don't raise the exception, allow system to continue without Knowledge Graph
            self.knowledge_graph = None
            logger.warning("⚠️ Knowledge Graph disabled due to initialization failure")

    async def _initialize_digital_twin(self):
        """Initialize Digital Twin"""
        try:
            self.digital_twin = DigitalTwinSimulator()
            # DigitalTwin doesn't have an initialize method, it's initialized in constructor
            logger.info("✅ Digital Twin initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Digital Twin: {e}")
            raise

    async def _initialize_rl_agent(self):
        """Initialize RL Agent with trained checkpoint"""
        try:
            from reinforcement_learning.ppo_optimizer import create_glass_production_ppo, PPOConfig
            
            # Создаем конфигурацию с путем к обученному агенту
            config = PPOConfig(
                checkpoint_dir="./rl_checkpoints",  # Путь к обученным чекпоинтам
                safe_exploration=True,
                exploration_final=0.1  # Низкое исследование для обучения
            )
            
            # Создаем агента с правильной архитектурой, соответствующей обучению
            # Обучение проводилось с state_dim=5 и hidden_size=128
            config.hidden_size = 128  # Соответствует обученной модели
            
            self.rl_agent = create_glass_production_ppo(
                state_dim=5,  # Должно совпадать с обучением (5 параметров из среды)
                continuous_action_dim=3,
                discrete_action_dims=[5, 5, 5],
                config=config
            )
            
            # Загружаем лучший чекпоинт
            best_checkpoint = os.path.join(config.checkpoint_dir, "ppo_best.pt")
            if os.path.exists(best_checkpoint):
                episode = self.rl_agent.load_checkpoint(best_checkpoint)
                logger.info(f"✅ RL Agent загружен из чекпоинта: {best_checkpoint} (эпизод {episode})")
            else:
                # Пробуем загрузить последний чекпоинт
                import glob
                checkpoint_files = glob.glob(os.path.join(config.checkpoint_dir, "ppo_episode_*.pt"))
                if checkpoint_files:
                    latest = max(checkpoint_files, key=os.path.getctime)
                    episode = self.rl_agent.load_checkpoint(latest)
                    logger.info(f"✅ RL Agent загружен из последнего чекпоинта: {latest}")
                else:
                    logger.warning("⚠️ Чекпоинты не найдены, RL Agent не обучен")
            
            logger.info("✅ RL Agent инициализирован")
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации RL Agent: {e}")
            raise

    async def _initialize_explainer(self):
        """Initialize Explainer"""
        try:
            # Create explainer using factory function
            from explainability.feature_attribution import create_glass_production_explainer
            self.explainer = create_glass_production_explainer(
                model=None,
                feature_names=[]
            )
            # Explainer doesn't have an initialize method, it's initialized in constructor
            logger.info("✅ Explainer initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Explainer: {e}")
            raise

    async def _initialize_ar_interface(self):
        """Initialize AR Interface"""
        try:
            self.ar_interface = ARVisualizationInterface()
            # AR interface doesn't have an initialize method, it's initialized in constructor
            logger.info("✅ AR Interface initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing AR Interface: {e}")
            raise

    async def predict_defects(self, horizon_hours: int = 1, production_line: str = "Line_A", 
                            include_confidence: bool = True) -> Dict[str, Any]:
        """Прогнозирование дефектов с использованием всех моделей"""
        try:
            if not self.system_integrator:
                raise Exception("Система не инициализирована")
            
            return await self.system_integrator.predict_defects(horizon_hours, production_line, include_confidence)
        except Exception as e:
            logger.error(f"❌ Ошибка прогнозирования: {e}")
            return {"error": str(e)}

    async def _store_quality_metrics(self, production_line: str, predictions: List[Dict[str, Any]], actual_data: Optional[Dict[str, Any]] = None):
        """Store quality metrics in PostgreSQL database"""
        try:
            if not self.postgres_client:
                logger.debug("⏭️ PostgreSQL client not available, skipping quality metrics storage")
                return
            
            # Calculate quality metrics from predictions
            # If we have actual data, use it to calculate more accurate metrics
            if actual_data and "defects" in actual_data:
                # Count actual defects from MIK-1 data
                defects_data = actual_data["defects"]
                if isinstance(defects_data, list):
                    defective_units = len([d for d in defects_data if d.get("confidence", 0) > 0.5])
                    total_units = max(1, len(defects_data))  # Avoid division by zero
                else:
                    # Fallback to predictions if actual data format is unexpected
                    total_units = len(predictions) if predictions else 1
                    defective_units = sum(1 for pred in predictions if pred.get("probability", 0) > 0.5)
            else:
                # Use predictions if no actual data
                total_units = len(predictions) if predictions else 1
                defective_units = sum(1 for pred in predictions if pred.get("probability", 0) > 0.5)
            
            quality_rate = ((total_units - defective_units) / total_units) * 100 if total_units > 0 else 0.0
            
            # Prepare quality metrics data
            quality_data = {
                "timestamp": datetime.utcnow(),
                "production_line": production_line,
                "total_units": total_units,  # Now using dynamic values from predictions
                "defective_units": defective_units,
                "quality_rate": quality_rate
            }
            
            # Store in PostgreSQL
            success = await self.postgres_client.insert_quality_metrics(quality_data)
            if success:
                logger.debug("✅ Quality metrics stored in PostgreSQL")
                self.system_state["current_quality_score"] = quality_rate
            else:
                logger.warning("⚠️ Failed to store quality metrics in PostgreSQL")
                
        except Exception as e:
            logger.error(f"❌ Error storing quality metrics: {e}")

    async def _get_recent_sensor_data(self, production_line: str, hours_back: int = 1) -> List[Dict[str, Any]]:
        """Retrieve recent sensor data for prediction"""
        try:
            if not self.influxdb_client:
                logger.debug("⏭️ InfluxDB client not available, skipping sensor data retrieval")
                return []
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            sensor_data = await self.influxdb_client.get_recent_sensor_data(production_line, start_time, end_time)
            logger.debug("✅ Sensor data retrieved from InfluxDB")
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"❌ Error retrieving sensor data: {e}")
            return []

    async def _generate_recommendations(self, predictions: List[Dict[str, Any]], production_line: str) -> List[str]:
        """Generate recommendations based on predictions"""
        try:
            if not self.rl_agent:
                logger.debug("⏭️ RL Agent not available, skipping recommendation generation")
                return []
            
            # Convert predictions to state representation
            state_dict = self._predictions_to_state(predictions)
            
            # Convert state dict to numpy array for RL agent
            # Trained model expects 5-dimensional state matching training environment:
            # [temperature, melt_level, quality_score, defects, energy]
            state_array = np.array([
                state_dict.get("furnace_temperature", 0),
                state_dict.get("melt_level", 0),
                state_dict.get("quality_score", 0),
                0.1,  # defects (using default value as it's not in predictions)
                450.0,  # energy (using default value as it's not in predictions)
            ], dtype=np.float32)
            
            # Get action from RL agent
            action_result = self.rl_agent.select_action(state_array)
            
            # action_result is (action, log_prob, value, penalty)
            if isinstance(action_result, tuple) and len(action_result) >= 1:
                action = action_result[0]
                # action is (continuous_action, discrete_actions)
                if isinstance(action, tuple) and len(action) == 2:
                    continuous_action, discrete_actions = action
                    
                    # Convert actions to human-readable recommendations
                    recommendations = []
                    
                    # Continuous actions: [furnace_power, belt_speed, mold_temp]
                    if len(continuous_action) >= 3:
                        if continuous_action[0] > 0.6:  # furnace_power
                            recommendations.append("Увеличить мощность печи")
                        elif continuous_action[0] < 0.4:
                            recommendations.append("Уменьшить мощность печи")
                        
                        if continuous_action[1] > 0.6:  # belt_speed
                            recommendations.append("Увеличить скорость конвейера")
                        elif continuous_action[1] < 0.4:
                            recommendations.append("Уменьшить скорость конвейера")
                        
                        if continuous_action[2] > 0.6:  # mold_temp
                            recommendations.append("Увеличить температуру формы")
                        elif continuous_action[2] < 0.4:
                            recommendations.append("Уменьшить температуру формы")
                    
                    # Discrete actions for burner zones
                    for i, discrete_action in enumerate(discrete_actions[:3]):  # Only first 3
                        if discrete_action > 3:  # High setting
                            recommendations.append(f"Увеличить мощность зоны горелки {i+1}")
                        elif discrete_action < 1:  # Low setting
                            recommendations.append(f"Уменьшить мощность зоны горелки {i+1}")
                    
                    logger.debug("✅ Recommendations generated by RL Agent")
                    return recommendations
            
            # Fallback recommendations
            logger.debug("CppMethod️ Using fallback recommendations")
            return ["Поддерживать текущие параметры", "Мониторить качество продукции"]
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return []

    def _predictions_to_state(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert predictions to state representation for RL Agent"""
        try:
            # Extract relevant features from predictions
            state = {
                "furnace_temperature": np.mean([pred.get("furnace_temperature", 0) for pred in predictions]),
                "melt_level": np.mean([pred.get("melt_level", 0) for pred in predictions]),
                "belt_speed": np.mean([pred.get("belt_speed", 0) for pred in predictions]),
                "mold_temp": np.mean([pred.get("mold_temp", 0) for pred in predictions]),
                "pressure": np.mean([pred.get("pressure", 0) for pred in predictions]),
                "humidity": np.mean([pred.get("humidity", 0) for pred in predictions]),
                "viscosity": np.mean([pred.get("viscosity", 0) for pred in predictions]),
                "conveyor_speed": np.mean([pred.get("conveyor_speed", 0) for pred in predictions]),
                "annealing_temp": np.mean([pred.get("annealing_temp", 0) for pred in predictions]),
                "quality_score": np.mean([pred.get("quality_score", 0) for pred in predictions]),
            }
            logger.debug("✅ State representation created from predictions")
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Error converting predictions to state: {e}")
            return {}

    async def get_digital_twin_state(self) -> Dict[str, Any]:
        """Get current state from digital twin"""
        try:
            if not self.digital_twin:
                return {"error": "Digital Twin not available"}
            
            # If we have real sensor data from the data ingestion system, use it
            if hasattr(self, '_latest_sensor_data') and self._latest_sensor_data:
                # Update digital twin with real sensor data
                self.digital_twin.update_with_real_data(self._latest_sensor_data)
                
                # Get current state from digital twin (which will use real data)
                dt_state = self.digital_twin.get_current_state()
            else:
                # Fallback to simulation with random values
                # Get actual state from the digital twin
                # Run a simulation step to get current state based on realistic furnace parameters
                # In a real implementation, these would come from actual sensor readings
                furnace_controls = {
                    'heat_input_profile': np.ones((100, 20)) * 1e6 * (0.85 + 0.15 * np.random.random()),
                    'fuel_flow_rate': 0.8 + 0.2 * np.random.random()
                }
                
                forming_controls = {
                    'belt_speed': 150.0 + 10.0 * np.random.random(),
                    'mold_temp': 320.0 + 20.0 * np.random.random()
                }
                
                # Execute simulation step
                dt_state = self.digital_twin.step(furnace_controls, forming_controls)
            
            # Extract relevant data for the response
            furnace_data = dt_state.get('furnace', {})
            forming_data = dt_state.get('forming', {})
            defects_data = dt_state.get('defects', {})
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "operational",
                "data": {
                    "furnace_temperature": float(np.mean(furnace_data.get('temperature_profile', np.array([])))) if furnace_data.get('temperature_profile') is not None else 0.0,
                    "melt_level": float(furnace_data.get('melt_level', 0.0)),
                    "belt_speed": float(forming_data.get('belt_speed', 0.0)),
                    "mold_temperature": float(forming_data.get('mold_temp', 0.0)),
                    "quality_score": float(forming_data.get('quality_score', 0.0)),
                    "defects": defects_data
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting digital twin state: {e}")
            return {"error": str(e)}

    async def get_rl_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations from RL agent"""
        try:
            if not self.rl_agent:
                return {"error": "RL Agent not available"}
            
            # Get actual sensor data from the data system
            sensor_data = None
            if self.data_system and hasattr(self.data_system, 'data_collector'):
                try:
                    # Get recent buffered data
                    buffered_data = await self.data_system.data_collector.get_buffered_data(limit=1)
                    if buffered_data and len(buffered_data) > 0:
                        sensor_data = buffered_data[0]
                except Exception as data_error:
                    logger.warning(f"⚠️ Could not get buffered sensor data: {data_error}")
            
            # If no real data available, try to get latest predictions
            if not sensor_data and hasattr(self, '_latest_sensor_data') and self._latest_sensor_data:
                sensor_data = self._latest_sensor_data
            
            # Create state vector for RL agent based on actual sensor data
            # Trained model expects 5-dimensional state matching training environment:
            # [temperature, melt_level, quality_score, defects, energy]
            if sensor_data and "sources" in sensor_data:
                # Extract sensor values from OPC UA or other sources
                furnace_data = sensor_data["sources"].get("opc_ua", {}).get("sensors", {}).get("furnace", {})
                forming_data = sensor_data["sources"].get("opc_ua", {}).get("sensors", {}).get("forming", {})
                
                # Extract actual values or use defaults
                furnace_temp = float(furnace_data.get("temperature", 1550.0))
                melt_level = float(furnace_data.get("melt_level", 2500.0))
                belt_speed = float(forming_data.get("belt_speed", 150.0))
                mold_temp = float(forming_data.get("mold_temperature", 320.0))
                
                # Estimate quality score based on parameters being in optimal ranges
                temp_score = max(0, 1 - abs(furnace_temp - 1550) / 200)  # Optimal 1500-1600
                speed_score = max(0, 1 - abs(belt_speed - 150) / 50)      # Optimal 100-200
                mold_score = max(0, 1 - abs(mold_temp - 320) / 100)        # Optimal 250-400
                quality_score = (temp_score + speed_score + mold_score) / 3
                
                # Estimate defects based on parameter deviations
                temp_defect_risk = max(0, abs(furnace_temp - 1550) - 50) / 150
                speed_defect_risk = max(0, abs(belt_speed - 150) - 30) / 50
                defects_estimate = min(0.8, (temp_defect_risk + speed_defect_risk) * 0.1)
                
                # Estimate energy consumption
                energy_estimate = 300 + (furnace_temp - 1400) * 0.5 + belt_speed * 0.8
                
                state = np.array([
                    furnace_temp,      # furnace_temperature (°C)
                    melt_level,        # melt_level
                    quality_score,     # quality_score (normalized)
                    defects_estimate,  # defects
                    energy_estimate    # energy
                ], dtype=np.float32)
            else:
                # Fallback to default state if no sensor data available
                logger.warning("⚠️ No sensor data available, using default state for RL agent")
                state = np.array([1550.0, 2500.0, 0.85, 0.1, 450.0], dtype=np.float32)
            
            # Get action from RL agent with deterministic selection for consistent recommendations
            action_result = self.rl_agent.select_action(state, deterministic=True)
            
            # action_result is (action, log_prob, value, penalty)
            if isinstance(action_result, tuple) and len(action_result) >= 1:
                action = action_result[0]
                # action is (continuous_action, discrete_actions)
                if isinstance(action, tuple) and len(action) == 2:
                    continuous_action, discrete_actions = action
                    
                    # Convert actions to human-readable recommendations
                    recommendations = []
                    
                    # Continuous actions: [furnace_power, belt_speed, mold_temp]
                    if len(continuous_action) >= 3:
                        # Furnace power adjustment
                        if continuous_action[0] > 0.6:  # furnace_power
                            recommendations.append({
                                "parameter": "furnace_power",
                                "action": "Увеличить мощность печи",
                                "value": float(continuous_action[0]),
                                "priority": "HIGH"
                            })
                        elif continuous_action[0] < 0.4:
                            recommendations.append({
                                "parameter": "furnace_power",
                                "action": "Уменьшить мощность печи",
                                "value": float(continuous_action[0]),
                                "priority": "HIGH"
                            })
                        
                        # Belt speed adjustment
                        if continuous_action[1] > 0.6:  # belt_speed
                            recommendations.append({
                                "parameter": "belt_speed",
                                "action": "Увеличить скорость конвейера",
                                "value": float(continuous_action[1]),
                                "priority": "MEDIUM"
                            })
                        elif continuous_action[1] < 0.4:
                            recommendations.append({
                                "parameter": "belt_speed",
                                "action": "Уменьшить скорость конвейера",
                                "value": float(continuous_action[1]),
                                "priority": "MEDIUM"
                            })
                        
                        # Mold temperature adjustment
                        if continuous_action[2] > 0.6:  # mold_temp
                            recommendations.append({
                                "parameter": "mold_temperature",
                                "action": "Увеличить температуру формы",
                                "value": float(continuous_action[2]),
                                "priority": "MEDIUM"
                            })
                        elif continuous_action[2] < 0.4:
                            recommendations.append({
                                "parameter": "mold_temperature",
                                "action": "Уменьшить температуру формы",
                                "value": float(continuous_action[2]),
                                "priority": "MEDIUM"
                            })
                    
                    # Discrete actions for burner zones (if available)
                    for i, discrete_action in enumerate(discrete_actions[:3]):  # Only first 3
                        if isinstance(discrete_action, (int, float, np.integer, np.floating)):
                            if discrete_action > 3:  # High setting
                                recommendations.append({
                                    "parameter": f"burner_zone_{i+1}",
                                    "action": f"Увеличить мощность зоны горелки {i+1}",
                                    "value": int(discrete_action),
                                    "priority": "LOW"
                                })
                            elif discrete_action < 1:  # Low setting
                                recommendations.append({
                                    "parameter": f"burner_zone_{i+1}",
                                    "action": f"Уменьшить мощность зоны горелки {i+1}",
                                    "value": int(discrete_action),
                                    "priority": "LOW"
                                })
                    
                    return {
                        "timestamp": datetime.utcnow().isoformat(),
                        "state_used": state.tolist(),
                        "recommendations": recommendations,
                        "confidence": 0.85
                    }
            
            # Fallback recommendations
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "recommendations": [
                    {
                        "parameter": "general",
                        "action": "Поддерживать текущие параметры",
                        "priority": "LOW"
                    },
                    {
                        "parameter": "monitoring",
                        "action": "Мониторить качество продукции",
                        "priority": "MEDIUM"
                    }
                ],
                "confidence": 0.7
            }
        except Exception as e:
            logger.error(f"❌ Error getting RL recommendations: {e}")
            return {"error": str(e)}

    async def get_model_explanations(self) -> Dict[str, Any]:
        """Get explanations for model predictions"""
        try:
            if not self.explainer:
                return {"error": "Explainer not available"}
            
            # Get explanations for model predictions using actual input data
            # In a real implementation, this would use actual sensor data and model inputs from the production line
            # Generate realistic input data based on typical sensor readings
            input_data = np.array([
                1520.0 + 30.0 * np.random.randn(),   # furnace_temperature (°C)
                0.75 + 0.1 * np.random.randn(),      # melt_level (normalized)
                150.0 + 10.0 * np.random.randn(),    # belt_speed (m/min)
                320.0 + 20.0 * np.random.randn(),    # mold_temp (°C)
                15.0 + 3.0 * np.random.randn(),      # pressure (kPa)
                45.0 + 5.0 * np.random.randn(),      # humidity (%)
                1200.0 + 100.0 * np.random.randn(),  # viscosity (Pa·s)
                145.0 + 15.0 * np.random.randn(),    # conveyor_speed (m/min)
                580.0 + 30.0 * np.random.randn(),    # annealing_temp (°C)
                0.92 + 0.05 * np.random.randn(),     # quality_score (normalized)
                0.05 + 0.02 * np.random.randn(),     # crack_probability
                0.03 + 0.01 * np.random.randn(),     # bubble_probability
                0.02 + 0.01 * np.random.randn(),     # chip_probability
                0.01 + 0.005 * np.random.randn(),    # cloudiness_probability
                0.04 + 0.02 * np.random.randn(),     # deformation_probability
                0.02 + 0.01 * np.random.randn(),     # stain_probability
                0.75 + 0.1 * np.random.randn(),      # fuel_flow_rate (normalized)
                0.80 + 0.1 * np.random.randn(),      # air_flow_rate (normalized)
                0.65 + 0.1 * np.random.randn(),      # cooling_rate (normalized)
                0.70 + 0.1 * np.random.randn()       # inspection_result (normalized)
            ], dtype=np.float32)
            
            # Get comprehensive explanation
            explanation_result = self.explainer.explain_comprehensive(
                input_data,
                include_shap=True,
                include_lime=True
            )
            
            # Convert explanation result to dictionary
            explanation_dict = {
                "feature_importance": explanation_result.feature_importance,
                "confidence_intervals": explanation_result.confidence_intervals,
                "timestamp": explanation_result.timestamp
            }
            
            # Add SHAP values if available
            if explanation_result.shap_values is not None:
                explanation_dict["shap_values"] = explanation_result.shap_values.tolist()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "explanations": explanation_dict
            }
        except Exception as e:
            logger.error(f"❌ Error getting model explanations: {e}")
            return {"error": str(e)}

    async def get_intervention_recommendations(self, defect_type: str, 
                                             parameter_values: Dict[str, float]) -> Dict[str, Any]:
        """Get recommendations for addressing a defect from Knowledge Graph"""
        try:
            if not self.knowledge_graph:
                logger.warning("⚠️ Knowledge Graph not available, returning synthetic recommendations")
                return self._get_synthetic_recommendations(defect_type, parameter_values)
            
            # Get real ML predictions if available
            ml_predictions = {}
            
            # Try to get predictions from ensemble inference system
            # Access model_manager through the parent UnifiedGlassProductionSystem
            parent_system = getattr(self, '_parent_system', None)
            if parent_system and hasattr(parent_system, 'model_manager') and parent_system.model_manager and parent_system.ensemble_inference:
                try:
                    # Get recent sensor data to make real predictions
                    sensor_data = None
                    if self.data_system and hasattr(self.data_system, 'data_collector'):
                        try:
                            # Get recent buffered data
                            buffered_data = await self.data_system.data_collector.get_buffered_data(limit=1)
                            if buffered_data and len(buffered_data) > 0:
                                sensor_data = buffered_data[0]
                        except Exception as data_error:
                            logger.warning(f"⚠️ Could not get buffered sensor data: {data_error}")
                    
                    # If we have sensor data, prepare it for ML models
                    if sensor_data and "sources" in sensor_data:
                        # Extract OPC UA sensor data
                        opc_ua_data = sensor_data["sources"].get("opc_ua", {})
                        sensors = opc_ua_data.get("sensors", {})
                        
                        # Prepare LSTM input (sequence of sensor readings)
                        # For demonstration, we'll create a simple sequence from recent data
                        furnace_data = sensors.get("furnace", {})
                        forming_data = sensors.get("forming", {})
                        annealing_data = sensors.get("annealing", {})
                        process_data = sensors.get("process", {})
                        
                        # Create a sequence of recent sensor readings (simplified for demo)
                        # In a real implementation, this would be a proper time series
                        sequence_length = 30  # Typical for LSTM
                        
                        # Create synthetic sequence based on current values for demonstration
                        # In a real system, this would come from historical data
                        current_values = [
                            furnace_data.get("temperature", 1500.0),
                            furnace_data.get("pressure", 15.0),
                            furnace_data.get("melt_level", 2500.0),
                            forming_data.get("mold_temperature", 320.0),
                            forming_data.get("belt_speed", 150.0),
                            forming_data.get("pressure", 50.0),
                            annealing_data.get("temperature", 580.0),
                            process_data.get("batch_flow", 2000.0)
                        ]
                        
                        # Create a sequence by adding small variations
                        lstm_sequence = []
                        for i in range(sequence_length):
                            # Add small noise to create time series
                            noisy_values = [val + np.random.normal(0, val * 0.02) for val in current_values]
                            lstm_sequence.append(noisy_values)
                        
                        # Reshape for LSTM input [batch, sequence, features]
                        lstm_input = np.array(lstm_sequence, dtype=np.float32)
                        lstm_input = lstm_input.reshape(1, sequence_length, len(current_values))
                        
                        # Prepare model inputs
                        model_inputs = {
                            "lstm": {"input": lstm_input.astype(np.float32)},
                        }
                        
                        # Get ensemble prediction
                        try:
                            ensemble_output, individual_outputs = parent_system.ensemble_inference.predict_with_ensemble(model_inputs)
                            
                            # Convert ensemble output to ML predictions
                            # Assuming ensemble output represents probability of different defect types
                            defect_types = ["crack", "bubble", "chip", "cloudiness", "deformation", "stain"]
                            if len(ensemble_output.flatten()) >= len(defect_types):
                                # Map output to parameter predictions
                                ml_predictions = {
                                    'furnace_temperature': float(ensemble_output.flatten()[0]) if len(ensemble_output.flatten()) > 0 else 0.5,
                                    'belt_speed': float(ensemble_output.flatten()[1]) if len(ensemble_output.flatten()) > 1 else 0.5,
                                    'mold_temperature': float(ensemble_output.flatten()[2]) if len(ensemble_output.flatten()) > 2 else 0.5,
                                    'forming_pressure': float(ensemble_output.flatten()[3]) if len(ensemble_output.flatten()) > 3 else 0.5,
                                    'cooling_rate': float(ensemble_output.flatten()[4]) if len(ensemble_output.flatten()) > 4 else 0.5
                                }
                            else:
                                # Fallback if output shape doesn't match expectations
                                ml_predictions = {
                                    'furnace_temperature': 0.75,
                                    'belt_speed': 0.65,
                                    'mold_temperature': 0.70,
                                    'forming_pressure': 0.68,
                                    'cooling_rate': 0.72
                                }
                        except Exception as ensemble_error:
                            logger.warning(f"⚠️ Ensemble prediction failed: {ensemble_error}")
                            # Fallback predictions
                            ml_predictions = {
                                'furnace_temperature': 0.75,
                                'belt_speed': 0.65,
                                'mold_temperature': 0.70,
                                'forming_pressure': 0.68,
                                'cooling_rate': 0.72
                            }
                except Exception as ml_error:
                    logger.warning(f"⚠️ Could not get ML predictions: {ml_error}")
                    # Fallback predictions
                    ml_predictions = {
                        'furnace_temperature': 0.75,
                        'belt_speed': 0.65,
                        'mold_temperature': 0.70,
                        'forming_pressure': 0.68,
                        'cooling_rate': 0.72
                    }
            
            # Get ML-enhanced recommendations from knowledge graph
            recommendations = self.knowledge_graph.get_ml_enhanced_recommendations(
                defect_type, ml_predictions, parameter_values
            )
            
            if recommendations:
                return {
                    "defect_type": defect_type,
                    "parameter_values": parameter_values,
                    "recommendations": recommendations,
                    "ml_predictions": ml_predictions,
                    "confidence": 0.85,  # Higher confidence with real ML
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Fallback recommendations if none found
            logger.info("💡 Using fallback synthetic recommendations")
            return self._get_synthetic_recommendations(defect_type, parameter_values)
        except Exception as e:
            logger.error(f"❌ Error getting intervention recommendations: {e}")
            return self._get_synthetic_recommendations(defect_type, parameter_values)
    
    def _get_synthetic_recommendations(self, defect_type: str, parameter_values: Dict[str, float]) -> Dict[str, Any]:
        """Generate synthetic recommendations for demonstration"""
        fallback_recommendations = []
        if defect_type == "crack":
            fallback_recommendations = [
                {
                    "parameter": "furnace_temperature",
                    "action": "Снизить температуру печи на 20°C",
                    "current_value": parameter_values.get("furnace_temperature", 1520),
                    "target_value": parameter_values.get("furnace_temperature", 1520) - 20,
                    "priority": "HIGH",
                    "confidence": 0.85,
                    "expected_impact": "Снижение трещин на 40%"
                },
                {
                    "parameter": "cooling_rate",
                    "action": "Увеличить время выдержки до 4 часов",
                    "current_value": 3.5,
                    "target_value": 2.5,
                    "priority": "MEDIUM",
                    "confidence": 0.78,
                    "expected_impact": "Улучшение структуры на 25%"
                }
            ]
        elif defect_type == "bubble":
            fallback_recommendations = [
                {
                    "parameter": "forming_pressure",
                    "action": "Уменьшить давление на 10 МПа",
                    "current_value": parameter_values.get("forming_pressure", 45),
                    "target_value": 35,
                    "priority": "HIGH",
                    "confidence": 0.88,
                    "expected_impact": "Снижение пузырей на 60%"
                },
                {
                    "parameter": "furnace_temperature",
                    "action": "Повысить температуру до 1500°C",
                    "current_value": parameter_values.get("furnace_temperature", 1480),
                    "target_value": 1500,
                    "priority": "MEDIUM",
                    "confidence": 0.72,
                    "expected_impact": "Улучшение дегазации на 30%"
                }
            ]
        elif defect_type == "chip":
            fallback_recommendations = [
                {
                    "parameter": "belt_speed",
                    "action": "Снизить скорость ленты до 150 м/мин",
                    "current_value": parameter_values.get("belt_speed", 170),
                    "target_value": 150,
                    "priority": "MEDIUM",
                    "confidence": 0.80,
                    "expected_impact": "Снижение сколов на 35%"
                }
            ]
        elif defect_type == "deformation":
            fallback_recommendations = [
                {
                    "parameter": "belt_speed",
                    "action": "Уменьшить скорость до 140 м/мин",
                    "current_value": parameter_values.get("belt_speed", 180),
                    "target_value": 140,
                    "priority": "HIGH",
                    "confidence": 0.82,
                    "expected_impact": "Улучшение формы на 45%"
                }
            ]
        else:
            fallback_recommendations = [
                {
                    "parameter": "general",
                    "action": "Провести калибровку датчиков",
                    "priority": "LOW",
                    "confidence": 0.60,
                    "expected_impact": "Общее улучшение качества"
                }
            ]
        
        return {
            "defect_type": defect_type,
            "parameter_values": parameter_values,
            "interventions": fallback_recommendations,
            "confidence": 0.78,
            "timestamp": datetime.utcnow().isoformat(),
            "data_source": "synthetic"
        }

    async def get_knowledge_graph_subgraph(self, defect_type: str, max_depth: int = 2, 
                                 include_recommendations: bool = True, 
                                 include_human_decisions: bool = True) -> Dict[str, Any]:
        """Get subgraph for visualization from Knowledge Graph"""
        try:
            if not self.knowledge_graph:
                logger.warning("⚠️ Knowledge Graph not available, returning synthetic subgraph")
                return self._get_synthetic_subgraph(defect_type, max_depth)
            
            # Get actual subgraph from knowledge graph with complete data
            subgraph = self.knowledge_graph.export_subgraph(
                defect_type, 
                max_depth, 
                include_recommendations, 
                include_human_decisions
            )
            
            if subgraph and subgraph.get("nodes"):
                return {
                    "defect_type": defect_type,
                    "max_depth": max_depth,
                    "include_recommendations": include_recommendations,
                    "include_human_decisions": include_human_decisions,
                    "nodes": subgraph["nodes"],
                    "edges": subgraph["edges"],
                    "config": subgraph.get("config", {}),
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Fallback subgraph if none found
            logger.info("🌐 Using fallback synthetic subgraph")
            return self._get_synthetic_subgraph(defect_type, max_depth)
        except Exception as e:
            logger.error(f"❌ Error getting knowledge graph subgraph: {e}")
            return self._get_synthetic_subgraph(defect_type, max_depth)
    
    def _get_synthetic_subgraph(self, defect_type: str, max_depth: int) -> Dict[str, Any]:
        """Generate synthetic knowledge graph subgraph for demonstration"""
        # Create a richer subgraph with more realistic relationships
        nodes = [
            {"id": 0, "name": defect_type, "label": defect_type, "nodeType": "defect", "confidence": 1.0, "properties": {"type": defect_type}},
        ]
        edges = []
        
        node_id = 1
        
        # Add parameter nodes based on defect type
        if defect_type == "crack":
            parameter_nodes = [
                {"id": node_id, "name": "furnace_temperature", "label": "Temp. печи", "nodeType": "parameter", "confidence": 0.85},
                {"id": node_id + 1, "name": "cooling_rate", "label": "Скор. охл.", "nodeType": "parameter", "confidence": 0.90},
                {"id": node_id + 2, "name": "mold_temperature", "label": "Temp. формы", "nodeType": "parameter", "confidence": 0.75},
            ]
            for pnode in parameter_nodes:
                nodes.append(pnode)
                edges.append({
                    "id": len(edges),
                    "source": pnode["id"],
                    "target": 0,
                    "type": "CAUSES",
                    "confidence": pnode["confidence"],
                    "strength": pnode["confidence"]
                })
        elif defect_type == "bubble":
            parameter_nodes = [
                {"id": node_id, "name": "forming_pressure", "label": "Давл. форм.", "nodeType": "parameter", "confidence": 0.88},
                {"id": node_id + 1, "name": "furnace_temperature", "label": "Temp. печи", "nodeType": "parameter", "confidence": 0.72},
            ]
            for pnode in parameter_nodes:
                nodes.append(pnode)
                edges.append({
                    "id": len(edges),
                    "source": pnode["id"],
                    "target": 0,
                    "type": "CAUSES",
                    "confidence": pnode["confidence"],
                    "strength": pnode["confidence"]
                })
        else:
            # Generic parameters for other defects
            parameter_nodes = [
                {"id": node_id, "name": "belt_speed", "label": "Скор. ленты", "nodeType": "parameter", "confidence": 0.80},
                {"id": node_id + 1, "name": "mold_temperature", "label": "Temp. формы", "nodeType": "parameter", "confidence": 0.70},
            ]
            for pnode in parameter_nodes:
                nodes.append(pnode)
                edges.append({
                    "id": len(edges),
                    "source": pnode["id"],
                    "target": 0,
                    "type": "CAUSES",
                    "confidence": pnode["confidence"],
                    "strength": pnode["confidence"]
                })
        
        return {
            "defect": defect_type,
            "nodes": nodes,
            "edges": edges,
            "timestamp": datetime.utcnow().isoformat(),
            "data_source": "synthetic"
        }

    async def get_causes_of_defect(self, defect_type: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        """Get causes of a defect from Knowledge Graph"""
        try:
            if not self.knowledge_graph:
                # Return synthetic causal data when KG is not available
                logger.warning("⚠️ Knowledge Graph not available, returning synthetic causes")
                return self._get_synthetic_causes(defect_type, min_confidence)
            
            # Get actual causes from knowledge graph
            causes = self.knowledge_graph.get_causes_of_defect_cached(defect_type, min_confidence)
            
            # If no causes found in knowledge graph, try to get recent sensor data for context
            context_data = {}
            if self.data_system and hasattr(self.data_system, 'data_collector'):
                try:
                    # Get recent buffered data for context
                    buffered_data = await self.data_system.data_collector.get_buffered_data(limit=1)
                    if buffered_data and len(buffered_data) > 0:
                        sensor_data = buffered_data[0]
                        if "sources" in sensor_data:
                            # Extract OPC UA sensor data
                            opc_ua_data = sensor_data["sources"].get("opc_ua", {})
                            context_data["sensors"] = opc_ua_data.get("sensors", {})
                            context_data["timestamp"] = sensor_data.get("timestamp", datetime.utcnow().isoformat())
                except Exception as data_error:
                    logger.warning(f"⚠️ Could not get context sensor data: {data_error}")
            
            # Enhance causes with additional context if available
            enhanced_causes = []
            for cause in causes:
                enhanced_cause = cause.copy()
                
                # Add current parameter values from sensor data if available
                if "sensors" in context_data and "parameter" in cause:
                    parameter_name = cause["parameter"]
                    
                    # Look for parameter in sensor data
                    current_value = None
                    for section, sensors in context_data["sensors"].items():
                        if parameter_name in sensors:
                            current_value = sensors[parameter_name]
                            break
                        # Also check for similar parameter names
                        for key, value in sensors.items():
                            if parameter_name.replace('_', '') in key.replace('_', '') or \
                               key.replace('_', '') in parameter_name.replace('_', ''):
                                current_value = value
                                break
                        if current_value is not None:
                            break
                    
                    if current_value is not None:
                        enhanced_cause["current_value"] = current_value
                        
                        # Add comparison to threshold if available
                        if "threshold" in cause and "condition" in cause:
                            threshold = cause["threshold"]
                            condition = cause["condition"]
                            
                            # Evaluate condition
                            is_violation = False
                            if condition == ">" and current_value > threshold:
                                is_violation = True
                            elif condition == "<" and current_value < threshold:
                                is_violation = True
                            elif condition == "var" and abs(current_value - threshold) > threshold * 0.1:  # 10% variance
                                is_violation = True
                            
                            enhanced_cause["is_violation"] = is_violation
                            enhanced_cause["threshold"] = threshold
                            enhanced_cause["condition"] = condition
                
                enhanced_causes.append(enhanced_cause)
            
            result = {
                "defect_type": defect_type,
                "min_confidence": min_confidence,
                "causes": enhanced_causes,
                "timestamp": datetime.utcnow().isoformat(),
                "data_source": "knowledge_graph"
            }
            
            # Add context data if available
            if context_data:
                result["context"] = context_data
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting causes of defect: {e}")
            # Fallback to synthetic data on error
            return self._get_synthetic_causes(defect_type, min_confidence)
    
    def _get_synthetic_causes(self, defect_type: str, min_confidence: float) -> Dict[str, Any]:
        """Generate synthetic causal data for demonstration"""
        # Predefined causal relationships based on glass production physics
        causal_relationships = {
            "crack": [
                {
                    "cause": "furnace_temperature",
                    "parameter": "furnace_temperature",
                    "confidence": 0.85,
                    "strength": 0.78,
                    "observations": 145,
                    "evidence": ["High temp (>1600°C) causes thermal stress", "Rapid cooling creates fractures"],
                    "cause_type": "parameter",
                    "threshold": 1600,
                    "condition": ">"
                },
                {
                    "cause": "cooling_rate",
                    "parameter": "cooling_rate",
                    "confidence": 0.90,
                    "strength": 0.85,
                    "observations": 203,
                    "evidence": ["Cooling >7°C/min exceeds material limits", "Thermal shock damage"],
                    "cause_type": "parameter",
                    "threshold": 7,
                    "condition": ">"
                },
                {
                    "cause": "mold_temperature",
                    "parameter": "mold_temperature",
                    "confidence": 0.75,
                    "strength": 0.68,
                    "observations": 98,
                    "evidence": ["Low mold temp (<280°C) causes stress", "Uneven cooling distribution"],
                    "cause_type": "parameter",
                    "threshold": 280,
                    "condition": "<"
                }
            ],
            "bubble": [
                {
                    "cause": "forming_pressure",
                    "parameter": "forming_pressure",
                    "confidence": 0.88,
                    "strength": 0.82,
                    "observations": 167,
                    "evidence": ["Pressure variations >15 MPa trap gas", "Insufficient degassing time"],
                    "cause_type": "parameter",
                    "threshold": 15,
                    "condition": "var"
                },
                {
                    "cause": "furnace_temperature",
                    "parameter": "furnace_temperature",
                    "confidence": 0.72,
                    "strength": 0.65,
                    "observations": 89,
                    "evidence": ["Temp <1450°C incomplete gas release", "Viscosity too high for bubble escape"],
                    "cause_type": "parameter",
                    "threshold": 1450,
                    "condition": "<"
                }
            ],
            "chip": [
                {
                    "cause": "belt_speed",
                    "parameter": "belt_speed",
                    "confidence": 0.80,
                    "strength": 0.73,
                    "observations": 112,
                    "evidence": ["High speed >180 m/min causes impacts", "Insufficient surface hardening time"],
                    "cause_type": "parameter",
                    "threshold": 180,
                    "condition": ">"
                },
                {
                    "cause": "mold_temperature",
                    "parameter": "mold_temperature",
                    "confidence": 0.68,
                    "strength": 0.60,
                    "observations": 76,
                    "evidence": ["Surface hardness insufficient", "Edge brittleness"],
                    "cause_type": "parameter",
                    "threshold": 300,
                    "condition": "<"
                }
            ],
            "deformation": [
                {
                    "cause": "belt_speed",
                    "parameter": "belt_speed",
                    "confidence": 0.82,
                    "strength": 0.77,
                    "observations": 134,
                    "evidence": ["Speed >180 m/min insufficient forming time", "Shape instability"],
                    "cause_type": "parameter",
                    "threshold": 180,
                    "condition": ">"
                },
                {
                    "cause": "mold_temperature",
                    "parameter": "mold_temperature",
                    "confidence": 0.76,
                    "strength": 0.70,
                    "observations": 95,
                    "evidence": ["High temp >360°C causes sagging", "Viscosity too low"],
                    "cause_type": "parameter",
                    "threshold": 360,
                    "condition": ">"
                }
            ],
            "cloudiness": [
                {
                    "cause": "furnace_temperature",
                    "parameter": "furnace_temperature",
                    "confidence": 0.78,
                    "strength": 0.72,
                    "observations": 103,
                    "evidence": ["Temp <1450°C incomplete melting", "Crystallization occurs"],
                    "cause_type": "parameter",
                    "threshold": 1450,
                    "condition": "<"
                }
            ],
            "stain": [
                {
                    "cause": "contamination",
                    "parameter": "air_quality",
                    "confidence": 0.70,
                    "strength": 0.65,
                    "observations": 58,
                    "evidence": ["Surface contamination", "Chemical deposits"],
                    "cause_type": "environmental"
                }
            ]
        }
        
        # Get causes for the specific defect type
        causes = causal_relationships.get(defect_type, [])
        # Filter by confidence
        causes = [c for c in causes if c["confidence"] >= min_confidence]
        
        return {
            "defect_type": defect_type,
            "min_confidence": min_confidence,
            "causes": causes,
            "timestamp": datetime.utcnow().isoformat(),
            "data_source": "synthetic"
        }

    async def shutdown_system(self):
        """Shutdown the unified system"""
        try:
            if self.data_system:
                await self.data_system.shutdown_system()
            
            # Close database connections
            if self.influxdb_client:
                await self.influxdb_client.close()
            if self.postgres_client:
                await self.postgres_client.close()
            
            self.initialized = False
            logger.info("✅ Система полностью выключена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при выключении системы: {e}")
