"""
Setup and Configuration for Data Ingestion System
Handles system configuration, environment setup, and initialization
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestionSetup:
    """Setup and configuration manager for data ingestion system"""
    
    def __init__(self, config_file: str = "data_ingestion_config.json"):
        self.config_file = config_file
        self.config = self._load_default_config()
        self.environment = self._detect_environment()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "system": {
                "name": "GlassProductionDataIngestion",
                "version": "1.0.0",
                "environment": "development"
            },
            "collector": {
                "collection_interval": 1.0,
                "buffer_size": 1000,
                "sources": {
                    "opc_ua": {
                        "enabled": True,
                        "server_url": "opc.tcp://localhost:4840",
                        "namespace": "http://glass.factory/UA/",
                        "timeout": 5.0
                    },
                    "modbus": {
                        "enabled": True,
                        "protocol": "tcp",
                        "host": "localhost",
                        "port": 502,
                        "timeout": 3.0
                    },
                    "mik1_camera": {
                        "enabled": True,
                        "camera_source": "0",
                        "resolution": [1920, 1080],
                        "fps": 30
                    },
                    "mqtt": {
                        "enabled": True,
                        "broker_host": "localhost",
                        "broker_port": 1883,
                        "client_id": "glass_production_collector",
                        "keepalive": 60
                    }
                }
            },
            "router": {
                "default_routes": ["kafka", "feature_extractor", "validator"],
                "routing_rules": {
                    "sensor_data": ["kafka", "feature_extractor", "validator"],
                    "defect_data": ["kafka", "validator"],
                    "image_data": ["kafka", "buffer"],
                    "control_data": ["kafka", "websocket"],
                    "alarm_data": ["kafka", "websocket", "database"],
                    "quality_data": ["kafka", "feature_extractor", "database"],
                    "prediction_data": ["kafka", "websocket"],
                    "recommendation_data": ["kafka", "websocket"]
                }
            },
            "kafka": {
                "bootstrap_servers": ["localhost:9093"],
                "enable_idempotence": True,
                "compression_type": "gzip",
                "topics": {
                    "sensors_raw": "glass.sensors.raw",
                    "sensors_processed": "glass.sensors.processed",
                    "defects": "glass.defects",
                    "predictions": "glass.predictions",
                    "alerts": "glass.alerts",
                    "recommendations": "glass.recommendations",
                    "quality_metrics": "glass.quality.metrics"
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_logging": {
                    "enabled": True,
                    "file_path": "logs/data_ingestion.log",
                    "max_size_mb": 10,
                    "backup_count": 5
                }
            },
            "monitoring": {
                "health_check_interval": 30,
                "metrics_export": {
                    "enabled": True,
                    "port": 8001,
                    "endpoint": "/metrics"
                }
            }
        }
    
    def _detect_environment(self) -> str:
        """Detect runtime environment"""
        env = os.getenv("ENVIRONMENT", "development")
        
        # Check for common environment indicators
        if os.getenv("DOCKER_CONTAINER"):
            env = "docker"
        elif os.getenv("KUBERNETES_SERVICE_HOST"):
            env = "kubernetes"
        elif os.getenv("PRODUCTION", "").lower() in ["true", "1", "yes"]:
            env = "production"
        
        return env
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file"""
        # Auto-select configuration file based on environment
        if config_file is None:
            if self.environment == "docker":
                # Check if Docker-specific config exists
                docker_config_path = "data_ingestion_config_docker.json"
                if os.path.exists(docker_config_path):
                    config_file = docker_config_path
                else:
                    config_file = self.config_file
            else:
                config_file = self.config_file
        
        config_path = config_file or self.config_file
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    # Merge with default config
                    self.config = self._merge_configs(self.config, file_config)
                    logger.info(f"✅ Configuration loaded from {config_path}")
            else:
                logger.info("📝 Using default configuration")
                # Save default config
                self.save_config(config_path)
            
            return self.config
            
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
            return self.config
    
    def _merge_configs(self, default: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries"""
        merged = default.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        config_path = config_file or self.config_file
        
        try:
            # Create directory if it doesn't exist
            config_dir = os.path.dirname(config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving configuration: {e}")
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Validate required fields
            required_fields = [
                "collector.sources.opc_ua.server_url",
                "collector.sources.modbus.host",
                "collector.sources.mik1_camera.camera_source",
                "collector.sources.mqtt.broker_host",
                "kafka.bootstrap_servers"
            ]
            
            for field_path in required_fields:
                value = self._get_nested_value(self.config, field_path)
                if value is None:
                    logger.error(f"❌ Required configuration field missing: {field_path}")
                    return False
            
            # Validate data types
            if not isinstance(self.config["collector"]["collection_interval"], (int, float)):
                logger.error("❌ collector.collection_interval must be a number")
                return False
            
            if not isinstance(self.config["collector"]["buffer_size"], int):
                logger.error("❌ collector.buffer_size must be an integer")
                return False
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating configuration: {e}")
            return False
    
    def _get_nested_value(self, config: Dict, path: str):
        """Get nested configuration value using dot notation"""
        keys = path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        try:
            log_config = self.config["logging"]
            
            # Set logging level
            level = getattr(logging, log_config["level"].upper(), logging.INFO)
            logging.getLogger().setLevel(level)
            
            # Setup file logging if enabled
            if log_config["file_logging"]["enabled"]:
                file_config = log_config["file_logging"]
                log_file = file_config["file_path"]
                
                # Create log directory
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                # Add file handler with UTF-8 encoding
                from logging.handlers import RotatingFileHandler
                # Handle encoding properly for Windows
                try:
                    file_handler = RotatingFileHandler(
                        log_file,
                        maxBytes=file_config["max_size_mb"] * 1024 * 1024,
                        backupCount=file_config["backup_count"],
                        encoding='utf-8'
                    )
                except TypeError:
                    # Fallback for older Python versions
                    file_handler = RotatingFileHandler(
                        log_file,
                        maxBytes=file_config["max_size_mb"] * 1024 * 1024,
                        backupCount=file_config["backup_count"]
                    )
                file_handler.setFormatter(
                    logging.Formatter(log_config["format"])
                )
                logging.getLogger().addHandler(file_handler)
            
            logger.info("✅ Logging setup complete")
            
        except Exception as e:
            logger.error(f"❌ Error setting up logging: {e}")
    
    def setup_environment(self):
        """Setup environment variables and directories"""
        try:
            # Create necessary directories
            dirs_to_create = [
                "logs",
                "data",
                "config",
                "tmp"
            ]
            
            for dir_name in dirs_to_create:
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                    logger.info(f"📁 Created directory: {dir_name}")
            
            # Set environment variables if not already set
            env_vars = {
                "ENVIRONMENT": self.environment,
                "DATA_INGESTION_CONFIG": self.config_file
            }
            
            for var_name, var_value in env_vars.items():
                if not os.getenv(var_name):
                    os.environ[var_name] = str(var_value)
            
            logger.info(f"✅ Environment setup complete (environment: {self.environment})")
            
        except Exception as e:
            logger.error(f"❌ Error setting up environment: {e}")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about configured connections"""
        try:
            sources = self.config["collector"]["sources"]
            connection_info = {
                "timestamp": datetime.utcnow().isoformat(),
                "environment": self.environment,
                "sources": {}
            }
            
            for source_name, source_config in sources.items():
                if source_config["enabled"]:
                    connection_info["sources"][source_name] = {
                        "enabled": True,
                        "config": {k: v for k, v in source_config.items() 
                                 if k not in ['enabled'] and not k.endswith('_password')}
                    }
                else:
                    connection_info["sources"][source_name] = {"enabled": False}
            
            return connection_info
            
        except Exception as e:
            logger.error(f"❌ Error getting connection info: {e}")
            return {}
    
    def generate_env_file(self, env_file_path: str = ".env"):
        """Generate .env file with configuration"""
        try:
            # Determine appropriate Kafka bootstrap servers based on environment
            kafka_servers = self.config['kafka']['bootstrap_servers']
            
            # In Docker environment, we should use the internal service name
            # But for the .env file, we'll keep the default unless explicitly in Docker mode
            if self.environment == "docker":
                kafka_servers = ["kafka:9092"]
            
            env_content = f"""# Data Ingestion System Environment Variables
# Generated on {datetime.utcnow().isoformat()}

ENVIRONMENT={self.environment}
DATA_INGESTION_CONFIG={self.config_file}

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS={','.join(kafka_servers)}

# Logging
LOG_LEVEL={self.config['logging']['level']}

# Collector Settings
COLLECTION_INTERVAL={self.config['collector']['collection_interval']}
BUFFER_SIZE={self.config['collector']['buffer_size']}"""
            
            with open(env_file_path, 'w', encoding='utf-8') as f:
                f.write(env_content)
            
            logger.info(f"✅ Environment file generated: {env_file_path}")
            
        except Exception as e:
            logger.error(f"❌ Error generating env file: {e}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check for required dependencies"""
        dependencies = {
            "asyncua": False,  # OPC UA
            "pymodbus": False,  # Modbus
            "cv2": False,  # OpenCV for camera
            "aiomqtt": False,  # MQTT
            "aiokafka": False,  # Kafka
            "numpy": False,  # NumPy
            "scipy": False,  # SciPy
        }
        
        try:
            import asyncua
            dependencies["asyncua"] = True
        except ImportError:
            pass
        
        try:
            import pymodbus
            dependencies["pymodbus"] = True
        except ImportError:
            pass
        
        try:
            import cv2
            dependencies["cv2"] = True
        except ImportError:
            pass
        
        try:
            import aiomqtt
            dependencies["aiomqtt"] = True
        except ImportError:
            pass
        
        try:
            import aiokafka
            dependencies["aiokafka"] = True
        except ImportError:
            pass
        
        try:
            import numpy
            dependencies["numpy"] = True
        except ImportError:
            pass
        
        try:
            import scipy
            dependencies["scipy"] = True
        except ImportError:
            pass
        
        missing_deps = [dep for dep, installed in dependencies.items() if not installed]
        if missing_deps:
            logger.warning(f"⚠️ Missing dependencies: {missing_deps}")
        else:
            logger.info("✅ All dependencies satisfied")
        
        return dependencies


def main():
    """Main setup function"""
    logger.info("🔧 Data Ingestion System Setup")
    logger.info("=" * 40)
    
    # Create setup instance
    setup = DataIngestionSetup()
    
    # Load configuration
    config = setup.load_config()
    
    # Validate configuration
    if not setup.validate_config():
        logger.error("❌ Configuration validation failed")
        return 1
    
    # Setup environment
    setup.setup_environment()
    
    # Setup logging
    setup.setup_logging()
    
    # Check dependencies
    dependencies = setup.check_dependencies()
    
    # Show connection info
    connection_info = setup.get_connection_info()
    logger.info(f"🔌 Configured sources: {list(connection_info['sources'].keys())}")
    
    # Generate .env file
    setup.generate_env_file()
    
    # Summary
    logger.info("\n✅ Setup complete!")
    logger.info(f"   Environment: {setup.environment}")
    logger.info(f"   Configuration: {setup.config_file}")
    logger.info(f"   Dependencies: {sum(dependencies.values())}/{len(dependencies)} satisfied")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)