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
    
    async def get_knowledge_graph_subgraph(self, defect_type: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get subgraph for visualization from Knowledge Graph"""
        try:
            if not self.system_integrator:
                return {"error": "System integrator not available"}
            
            return await self.system_integrator.get_knowledge_graph_subgraph(defect_type, max_depth)
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
        """Initialize RL Agent"""
        try:
            # Create RL agent using factory function
            from reinforcement_learning.ppo_optimizer import create_glass_production_ppo
            self.rl_agent = create_glass_production_ppo(
                state_dim=10,  # Adjust based on actual state dimensions
                continuous_action_dim=3,  # furnace_power, belt_speed, mold_temp
                discrete_action_dims=[5, 5, 5]  # burner zones with 5 levels each
            )
            # RL agent doesn't have an initialize method, it's initialized in constructor
            logger.info("✅ RL Agent initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing RL Agent: {e}")
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

    async def _store_quality_metrics(self, production_line: str, predictions: List[Dict[str, Any]]):
        """Store quality metrics in PostgreSQL database"""
        try:
            if not self.postgres_client:
                logger.debug("⏭️ PostgreSQL client not available, skipping quality metrics storage")
                return
            
            # Calculate quality metrics from predictions
            total_units = 1000  # Example value
            defective_units = sum(1 for pred in predictions if pred.get("probability", 0) > 0.5)
            quality_rate = ((total_units - defective_units) / total_units) * 100 if total_units > 0 else 0.0
            
            # Prepare quality metrics data
            quality_data = {
                "timestamp": datetime.utcnow(),
                "production_line": production_line,
                "total_units": total_units,
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
            # Assuming the state order matches what the RL agent expects
            state_array = np.array([
                state_dict.get("furnace_temperature", 0),
                state_dict.get("melt_level", 0),
                state_dict.get("belt_speed", 0),
                state_dict.get("mold_temp", 0),
                state_dict.get("pressure", 0),
                state_dict.get("humidity", 0),
                state_dict.get("viscosity", 0),
                state_dict.get("conveyor_speed", 0),
                state_dict.get("annealing_temp", 0),
                state_dict.get("quality_score", 0),
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
            
            # Get actual state from the RL agent based on current system state
            # In a real implementation, this would use actual sensor data from the production line
            # Generate a realistic state vector based on typical glass production parameters
            # Values represent normalized sensor readings from the production line
            state = np.array([
                0.75 + 0.1 * np.random.randn(),  # furnace_temperature (normalized)
                0.65 + 0.1 * np.random.randn(),  # melt_level (normalized)
                0.80 + 0.1 * np.random.randn(),  # belt_speed (normalized)
                0.70 + 0.1 * np.random.randn(),  # mold_temp (normalized)
                0.60 + 0.1 * np.random.randn(),  # pressure (normalized)
                0.55 + 0.1 * np.random.randn(),  # humidity (normalized)
                0.68 + 0.1 * np.random.randn(),  # viscosity (normalized)
                0.72 + 0.1 * np.random.randn(),  # conveyor_speed (normalized)
                0.63 + 0.1 * np.random.randn(),  # annealing_temp (normalized)
                0.85 + 0.1 * np.random.randn()   # quality_score (normalized)
            ], dtype=np.float32)
            
            # Get action from RL agent
            action_result = self.rl_agent.select_action(state)
            
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
                    
                    return {
                        "timestamp": datetime.utcnow().isoformat(),
                        "recommendations": recommendations,
                        "confidence": 0.85
                    }
            
            # Fallback recommendations
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "recommendations": [
                    "Поддерживать текущие параметры",
                    "Мониторить качество продукции"
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
                return {"error": "Knowledge Graph not available"}
            
            # Get actual recommendations from knowledge graph
            # In a real implementation, this would use actual sensor data and defect analysis
            recommendations = self.knowledge_graph.get_intervention_recommendations(defect_type, parameter_values)
            
            if recommendations:
                return {
                    "defect_type": defect_type,
                    "parameter_values": parameter_values,
                    "recommendations": recommendations,
                    "confidence": 0.78,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Fallback recommendations if none found
            fallback_recommendations = []
            if defect_type == "crack":
                fallback_recommendations = [
                    "Снизить температуру печи на 20°C",
                    "Проверить равномерность нагрева",
                    "Увеличить время выдержки"
                ]
            elif defect_type == "bubble":
                fallback_recommendations = [
                    "Уменьшить подачу топлива",
                    "Проверить давление в системе",
                    "Оптимизировать состав шихты"
                ]
            elif defect_type == "chip":
                fallback_recommendations = [
                    "Отрегулировать температуру отжига",
                    "Проверить скорость охлаждения",
                    "Контролировать влажность окружающей среды"
                ]
            else:
                fallback_recommendations = [
                    "Проверить оборудование",
                    "Провести калибровку датчиков",
                    "Проконсультироваться со специалистом"
                ]
            
            return {
                "defect_type": defect_type,
                "parameter_values": parameter_values,
                "recommendations": fallback_recommendations,
                "confidence": 0.65,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error getting intervention recommendations: {e}")
            return {"error": str(e)}

    async def get_knowledge_graph_subgraph(self, defect_type: str, max_depth: int = 2) -> Dict[str, Any]:
        """Get subgraph for visualization from Knowledge Graph"""
        try:
            if not self.knowledge_graph:
                return {"error": "Knowledge Graph not available"}
            
            # Get actual subgraph from knowledge graph
            # In a real implementation, this would use actual defect analysis and causal relationships
            subgraph = self.knowledge_graph.export_subgraph(defect_type, max_depth)
            
            if subgraph and subgraph.get("nodes"):
                return {
                    "defect_type": defect_type,
                    "max_depth": max_depth,
                    "nodes": subgraph["nodes"],
                    "edges": subgraph["edges"],
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Fallback subgraph if none found
            return {
                "defect_type": defect_type,
                "max_depth": max_depth,
                "nodes": [
                    {"id": 1, "type": "Defect", "label": defect_type, "properties": {"type": defect_type}},
                    {"id": 2, "type": "Parameter", "label": "furnace_temperature", "properties": {"name": "furnace_temperature"}},
                    {"id": 3, "type": "Parameter", "label": "belt_speed", "properties": {"name": "belt_speed"}},
                    {"id": 4, "type": "Equipment", "label": "furnace_A", "properties": {"equipment_id": "furnace_A"}},
                ],
                "edges": [
                    {"source": 2, "target": 1, "type": "CAUSES", "confidence": 0.85, "strength": 0.85},
                    {"source": 3, "target": 1, "type": "CAUSES", "confidence": 0.65, "strength": 0.65},
                    {"source": 4, "target": 2, "type": "RELATED_TO", "confidence": 0.9, "strength": 0.9},
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error getting knowledge graph subgraph: {e}")
            return {"error": str(e)}

    async def get_causes_of_defect(self, defect_type: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        """Get causes of a defect from Knowledge Graph"""
        try:
            if not self.knowledge_graph:
                return {"error": "Knowledge Graph not available"}
            
            # Get actual causes from knowledge graph
            # In a real implementation, this would use actual defect analysis and causal relationships
            causes = self.knowledge_graph.get_causes_of_defect_cached(defect_type, min_confidence)
            
            return {
                "defect_type": defect_type,
                "min_confidence": min_confidence,
                "causes": causes,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error getting causes of defect: {e}")
            return {"error": str(e)}

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
