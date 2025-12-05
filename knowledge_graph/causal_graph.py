"""
Enhanced Knowledge Graph for Glass Production System
Implements causal relationship modeling, root cause analysis, and recommendation engine
Uses Neo4j for graph storage and Redis for caching
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import numpy as np
from neo4j import GraphDatabase
import redis
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGlassProductionKnowledgeGraph:
    """Enhanced knowledge graph with Neo4j backend and Redis caching"""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4jpassword",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.redis_client = None
        self.cache_enabled = False
        
        # Initialize connections with graceful fallback
        self._initialize_connections(redis_host, redis_port, redis_db)
    
    def _initialize_connections(self, redis_host: str, redis_port: int, redis_db: int):
        """Initialize Neo4j and Redis connections with graceful fallback"""
        # Initialize Neo4j connection
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✅ Neo4j database connected")
        except Exception as e:
            logger.warning(f"⚠️ Neo4j connection failed: {e}")
            self.driver = None
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.cache_enabled = True
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self.redis_client = None
            self.cache_enabled = False
    
    def initialize_knowledge_base(self):
        """Initialize knowledge base with schema and constraints"""
        if not self.driver:
            logger.warning("⚠️ Neo4j not available, skipping knowledge base initialization")
            return
        
        try:
            with self.driver.session() as session:
                # Create constraints for uniqueness
                session.run("""
                    CREATE CONSTRAINT equipment_id IF NOT EXISTS 
                    FOR (e:Equipment) REQUIRE e.equipment_id IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT parameter_name IF NOT EXISTS 
                    FOR (p:Parameter) REQUIRE p.name IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT defect_type IF NOT EXISTS 
                    FOR (d:Defect) REQUIRE d.type IS UNIQUE
                """)
                
                # Create indexes for performance
                session.run("CREATE INDEX equipment_type IF NOT EXISTS FOR (e:Equipment) ON (e.type)")
                session.run("CREATE INDEX parameter_category IF NOT EXISTS FOR (p:Parameter) ON (p.category)")
                session.run("CREATE INDEX defect_severity IF NOT EXISTS FOR (d:Defect) ON (d.severity)")
                
                # Create initial equipment nodes
                session.run("""
                    MERGE (furnace:Equipment {equipment_id: 'furnace_A', type: 'Furnace'})
                    MERGE (forming:Equipment {equipment_id: 'forming_A', type: 'Forming'})
                    MERGE (annealing:Equipment {equipment_id: 'annealing_A', type: 'Annealing'})
                """)
                
                # Create common parameters
                session.run("""
                    MERGE (furnace_temp:Parameter {name: 'furnace_temperature', category: 'thermal'})
                    MERGE (melt_level:Parameter {name: 'melt_level', category: 'process'})
                    MERGE (belt_speed:Parameter {name: 'belt_speed', category: 'mechanical'})
                    MERGE (mold_temp:Parameter {name: 'mold_temperature', category: 'thermal'})
                """)
                
                # Create common defects
                session.run("""
                    MERGE (crack:Defect {type: 'crack', severity: 'HIGH'})
                    MERGE (bubble:Defect {type: 'bubble', severity: 'MEDIUM'})
                    MERGE (chip:Defect {type: 'chip', severity: 'LOW'})
                    MERGE (cloudiness:Defect {type: 'cloudiness', severity: 'MEDIUM'})
                    MERGE (deformation:Defect {type: 'deformation', severity: 'HIGH'})
                """)
                
                logger.info("✅ Knowledge graph schema initialized")
                
        except Exception as e:
            logger.error(f"❌ Error initializing knowledge base: {e}")
    
    def add_measurement(
        self,
        parameter_name: str,
        value: float,
        timestamp: datetime,
        equipment_id: str,
        sensor_id: Optional[str] = None
    ):
        """Add a measurement to the knowledge graph"""
        if not self.driver:
            logger.debug("⚠️ Neo4j not available, skipping measurement addition")
            return
        
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (param:Parameter {name: $param_name})
                    MERGE (equip:Equipment {equipment_id: $equip_id})
                    CREATE (m:Measurement {
                        value: $value,
                        timestamp: $timestamp,
                        sensor_id: $sensor_id
                    })-[:MEASURES]->(param)
                    CREATE (m)-[:FROM_EQUIPMENT]->(equip)
                """, {
                    "param_name": parameter_name,
                    "equip_id": equipment_id,
                    "value": value,
                    "timestamp": timestamp.isoformat(),
                    "sensor_id": sensor_id
                })
                
                # Cache the latest measurement
                if self.cache_enabled:
                    cache_key = f"measurement:{equipment_id}:{parameter_name}:latest"
                    cache_data = {
                        "value": value,
                        "timestamp": timestamp.isoformat(),
                        "sensor_id": sensor_id
                    }
                    try:
                        self.redis_client.setex(cache_key, 3600, json.dumps(cache_data))  # 1 hour expiry
                    except Exception as e:
                        logger.warning(f"⚠️ Error caching measurement: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error adding measurement: {e}")
    
    def add_defect_occurrence(
        self,
        defect_type: str,
        severity: str,
        timestamp: datetime,
        description: Optional[str] = None
    ):
        """Add a defect occurrence to the knowledge graph"""
        if not self.driver:
            logger.debug("⚠️ Neo4j not available, skipping defect addition")
            return
        
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (defect:Defect {type: $defect_type})
                    SET defect.severity = $severity
                    CREATE (occ:DefectOccurrence {
                        timestamp: $timestamp,
                        description: $description
                    })-[:IS_INSTANCE_OF]->(defect)
                """, {
                    "defect_type": defect_type,
                    "severity": severity,
                    "timestamp": timestamp.isoformat(),
                    "description": description
                })
                
                # Cache the defect occurrence
                if self.cache_enabled:
                    cache_key = f"defect:{defect_type}:recent"
                    try:
                        # Add to a list of recent defects
                        defect_data = {
                            "timestamp": timestamp.isoformat(),
                            "severity": severity,
                            "description": description
                        }
                        self.redis_client.lpush(cache_key, json.dumps(defect_data))
                        self.redis_client.ltrim(cache_key, 0, 99)  # Keep last 100
                    except Exception as e:
                        logger.warning(f"⚠️ Error caching defect: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error adding defect occurrence: {e}")
    
    def get_causes_of_defect_cached(self, defect_type: str, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Get causes of defect with caching"""
        # Try to get from cache first
        if self.cache_enabled:
            try:
                cache_key = f"causes:{defect_type}:{min_confidence}"
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
            except Exception as e:
                logger.warning(f"⚠️ Error reading from cache: {e}")
        
        # Get from database
        causes = self.get_causes_of_defect(defect_type, min_confidence)
        
        # Cache the result
        if self.cache_enabled and causes:
            try:
                cache_key = f"causes:{defect_type}:{min_confidence}"
                self.redis_client.setex(cache_key, 1800, json.dumps(causes))  # 30 minutes expiry
            except Exception as e:
                logger.warning(f"⚠️ Error writing to cache: {e}")
        
        return causes
    
    def get_causes_of_defect(self, defect_type: str, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Get likely causes of a defect type"""
        if not self.driver:
            logger.warning("⚠️ Neo4j not available, returning empty causes")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (defect:Defect {type: $defect_type})<-[:CAUSES]-(cause)
                    WHERE cause.confidence >= $min_confidence
                    RETURN cause.parameter AS parameter, 
                           cause.equipment AS equipment,
                           cause.confidence AS confidence,
                           cause.description AS description
                    ORDER BY cause.confidence DESC
                """, {
                    "defect_type": defect_type,
                    "min_confidence": min_confidence
                })
                
                causes = []
                for record in result:
                    causes.append({
                        "parameter": record["parameter"],
                        "equipment": record["equipment"],
                        "confidence": record["confidence"],
                        "description": record["description"]
                    })
                
                return causes
                
        except Exception as e:
            logger.error(f"❌ Error getting causes of defect: {e}")
            return []
    
    def add_causal_relationship(
        self,
        parameter_name: str,
        equipment_id: str,
        defect_type: str,
        confidence: float,
        description: Optional[str] = None
    ):
        """Add a causal relationship between parameter and defect"""
        if not self.driver:
            logger.debug("⚠️ Neo4j not available, skipping causal relationship addition")
            return
        
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (param:Parameter {name: $param_name})
                    MERGE (equip:Equipment {equipment_id: $equip_id})
                    MERGE (defect:Defect {type: $defect_type})
                    MERGE (param)<-[:RELATED_TO]-(equip)
                    CREATE (cause:Cause {
                        confidence: $confidence,
                        description: $description,
                        timestamp: $timestamp
                    })-[:CAUSES]->(defect)
                    CREATE (cause)-[:RELATED_TO]->(param)
                    CREATE (cause)-[:FROM_EQUIPMENT]->(equip)
                """, {
                    "param_name": parameter_name,
                    "equip_id": equipment_id,
                    "defect_type": defect_type,
                    "confidence": confidence,
                    "description": description,
                    "timestamp": datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"❌ Error adding causal relationship: {e}")
    
    def get_recommendations_for_defect(self, defect_type: str) -> List[Dict[str, Any]]:
        """Get recommendations for addressing a defect"""
        if not self.driver:
            logger.warning("⚠️ Neo4j not available, returning empty recommendations")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (defect:Defect {type: $defect_type})<-[:ADDRESSES]-(rec:Recommendation)
                    RETURN rec.action AS action,
                           rec.priority AS priority,
                           rec.expected_impact AS expected_impact,
                           rec.description AS description
                    ORDER BY rec.priority DESC
                """, {
                    "defect_type": defect_type
                })
                
                recommendations = []
                for record in result:
                    recommendations.append({
                        "action": record["action"],
                        "priority": record["priority"],
                        "expected_impact": record["expected_impact"],
                        "description": record["description"]
                    })
                
                return recommendations
                
        except Exception as e:
            logger.error(f"❌ Error getting recommendations: {e}")
            return []
    
    def close(self):
        """Close database connections"""
        if self.driver:
            self.driver.close()
        if self.redis_client:
            self.redis_client.close()
    
    def update_relationship_weights(self, parameter_name: str, defect_type: str, 
                                  observed_effect: bool, effect_strength: float = 0.5):
        """Update relationship weights based on observed effects"""
        if not self.driver:
            logger.debug("⚠️ Neo4j not available, skipping relationship weight update")
            return
        
        try:
            with self.driver.session() as session:
                # Update confidence based on observation
                if observed_effect:
                    # Increase confidence if effect was observed
                    new_confidence = min(1.0, 0.5 + effect_strength * 0.5)
                else:
                    # Decrease confidence if effect was not observed
                    new_confidence = max(0.1, 0.5 - effect_strength * 0.3)
                
                session.run("""
                    MATCH (param:Parameter {name: $param_name})-[:RELATED_TO]->(equip:Equipment)
                    MATCH (defect:Defect {type: $defect_type})
                    MERGE (cause:Cause)-[:CAUSES]->(defect)
                    WHERE (cause)-[:RELATED_TO]->(param) AND (cause)-[:FROM_EQUIPMENT]->(equip)
                    SET cause.confidence = $new_confidence,
                        cause.last_updated = $timestamp
                """, {
                    "param_name": parameter_name,
                    "defect_type": defect_type,
                    "new_confidence": new_confidence,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.debug(f"Updated relationship weight for {parameter_name} -> {defect_type}: {new_confidence}")
                
        except Exception as e:
            logger.error(f"❌ Error updating relationship weights: {e}")
    
    def get_intervention_recommendations(self, defect_type: str, 
                                       current_parameters: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get intervention recommendations based on current parameters and defect type"""
        if not self.driver:
            logger.warning("⚠️ Neo4j not available, returning empty recommendations")
            return []
        
        try:
            with self.driver.session() as session:
                # Get causes and their recommended interventions
                result = session.run("""
                    MATCH (defect:Defect {type: $defect_type})<-[:CAUSES]-(cause)-[:RECOMMENDS]->(rec:Recommendation)
                    RETURN cause.parameter AS parameter,
                           rec.action AS action,
                           rec.target_value AS target_value,
                           rec.unit AS unit,
                           rec.priority AS priority,
                           rec.expected_impact AS expected_impact,
                           cause.confidence AS confidence
                    ORDER BY rec.priority DESC, cause.confidence DESC
                """, {
                    "defect_type": defect_type
                })
                
                recommendations = []
                for record in result:
                    parameter = record["parameter"]
                    current_value = current_parameters.get(parameter, 0)
                    target_value = record["target_value"]
                    
                    recommendations.append({
                        "parameter": parameter,
                        "current_value": current_value,
                        "target_value": target_value,
                        "unit": record["unit"],
                        "action": record["action"],
                        "confidence": record["confidence"],
                        "priority": record["priority"],
                        "expected_impact": record["expected_impact"]
                    })
                
                return recommendations
                
        except Exception as e:
            logger.error(f"❌ Error getting intervention recommendations: {e}")
            return []
    
    def export_subgraph(self, defect_type: str, max_depth: int = 2) -> Dict[str, Any]:
        """Export subgraph for visualization"""
        if not self.driver:
            logger.warning("⚠️ Neo4j not available, returning empty subgraph")
            return {"nodes": [], "edges": []}
        
        try:
            nodes = []
            edges = []
            node_ids = set()
            
            with self.driver.session() as session:
                # Get defect node
                defect_result = session.run("""
                    MATCH (d:Defect {type: $defect_type})
                    RETURN d.type AS type, id(d) AS id
                """, {
                    "defect_type": defect_type
                })
                
                defect_node = None
                for record in defect_result:
                    defect_node = {
                        "id": record["id"],
                        "label": record["type"],
                        "name": record["type"],
                        "properties": {"type": record["type"]},
                        "nodeType": "defect"
                    }
                    nodes.append(defect_node)
                    node_ids.add(record["id"])
                
                if not defect_node:
                    return {"nodes": [], "edges": []}
                
                # Get related parameters and equipment (up to max_depth)
                for depth in range(1, max_depth + 1):
                    # Get causes
                    causes_result = session.run("""
                        MATCH (d:Defect {type: $defect_type})<-[:CAUSES]-(cause)
                        OPTIONAL MATCH (cause)-[:RELATED_TO]->(param:Parameter)
                        OPTIONAL MATCH (cause)-[:FROM_EQUIPMENT]->(equip:Equipment)
                        RETURN cause.confidence AS confidence,
                               id(cause) AS cause_id,
                               param.name AS param_name,
                               id(param) AS param_id,
                               equip.equipment_id AS equip_id,
                               id(equip) AS equip_id_internal
                    """, {
                        "defect_type": defect_type
                    })
                    
                    for record in causes_result:
                        # Add parameter node
                        if record["param_id"] and record["param_id"] not in node_ids:
                            param_node = {
                                "id": record["param_id"],
                                "label": record["param_name"],
                                "name": record["param_name"],
                                "properties": {"name": record["param_name"]},
                                "nodeType": "parameter"
                            }
                            nodes.append(param_node)
                            node_ids.add(record["param_id"])
                        
                        # Add equipment node
                        if record["equip_id_internal"] and record["equip_id_internal"] not in node_ids:
                            equip_node = {
                                "id": record["equip_id_internal"],
                                "label": record["equip_id"],
                                "name": record["equip_id"],
                                "properties": {"equipment_id": record["equip_id"]},
                                "nodeType": "equipment"
                            }
                            nodes.append(equip_node)
                            node_ids.add(record["equip_id_internal"])
                        
                        # Add edges
                        if record["param_id"] and record["cause_id"]:
                            edges.append({
                                "source": record["param_id"],
                                "target": record["cause_id"],
                                "type": "RELATED_TO",
                                "confidence": record["confidence"] or 0.5,
                                "strength": record["confidence"] or 0.5
                            })
                        
                        if record["equip_id_internal"] and record["cause_id"]:
                            edges.append({
                                "source": record["equip_id_internal"],
                                "target": record["cause_id"],
                                "type": "FROM_EQUIPMENT",
                                "confidence": record["confidence"] or 0.5,
                                "strength": record["confidence"] or 0.5
                            })
                        
                        if record["cause_id"] and defect_node["id"]:
                            edges.append({
                                "source": record["cause_id"],
                                "target": defect_node["id"],
                                "type": "CAUSES",
                                "confidence": record["confidence"] or 0.5,
                                "strength": record["confidence"] or 0.5
                            })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "defect": defect_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error exporting subgraph: {e}")
            return {"nodes": [], "edges": []}

# Пример использования# if __name__ == "__main__":
#     import os
#     
#     # Load environment variables only if not in Docker
#     # Check if we're running in Docker by looking for typical Docker environment variables
#     is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
#     
#     if not is_docker:
#         # Only load .env file when not in Docker
#         from dotenv import load_dotenv
#         load_dotenv()
#     else:
#         # In Docker, ensure ENVIRONMENT is set correctly
#         if not os.environ.get('ENVIRONMENT'):
#             os.environ['ENVIRONMENT'] = 'docker'
#     
#     neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
#     neo4j_user = os.getenv("NEO4J_USER", "neo4j")
#     neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
#     redis_host = os.getenv("REDIS_HOST", "localhost")
#     redis_port = int(os.getenv("REDIS_PORT", 6379))
#     redis_db = int(os.getenv("REDIS_DB", 0))
#     
#     kg = EnhancedGlassProductionKnowledgeGraph(
#         uri=neo4j_uri,
#         user=neo4j_user,
#         password=neo4j_password,
#         redis_host=redis_host,
#         redis_port=redis_port,
#         redis_db=redis_db
#     )
#     
#     print("🧠 Enhanced Knowledge Graph инициализирован")
#     
#     # Инициализация базы знаний
#     kg.initialize_knowledge_base()
#     
#     # Тестирование кэширования
#     import time
#     
#     print("\n📊 Тест производительности с кэшированием:")
#     start = time.time()
#     causes1 = kg.get_causes_of_defect_cached("crack")
#     time1 = time.time() - start
#     
#     start = time.time()
#     causes2 = kg.get_causes_of_defect_cached("crack")  # Из кэша
#     time2 = time.time() - start
#     
#     print(f"  Первый запрос: {time1*1000:.2f}ms")
#     print(f"  Кэшированный запрос: {time2*1000:.2f}ms")
#     print(f"  Ускорение: {time1/time2:.1f}x")
#     
#     # Add some test data to demonstrate functionality
#     print("\n🧪 Добавление тестовых данных:")
#     timestamp = datetime.now()
#     kg.add_measurement("furnace_temperature", 1580.0, timestamp, "furnace_A", "temp_001")
#     kg.add_defect_occurrence("crack", "HIGH", timestamp, "Temperature fluctuation detected")
#     kg.add_causal_relationship("furnace_temperature", "furnace_A", "crack", 0.85, "High temperature causes cracking")
#     
#     # Получение причин дефекта
#     print("\n🔍 Получение причин дефекта:")
#     causes = kg.get_causes_of_defect_cached("crack", 0.5)
#     for i, cause in enumerate(causes[:3], 1):
#         print(f"  {i}. {cause['parameter']} ({cause['equipment']})")
#         print(f"     Confidence: {cause['confidence']:.2f}")
#         print(f"     Description: {cause['description']}")
#     
#     # Получение рекомендаций
#     print("\n💡 Получение рекомендаций:")
#     recommendations = kg.get_recommendations_for_defect("crack")
#     for i, rec in enumerate(recommendations[:3], 1):
#         print(f"  {i}. {rec['action']}")
#         print(f"     Priority: {rec['priority']}")
#         print(f"     Expected impact: {rec['expected_impact']}")
#     
#     # Обновление весов на основе наблюдений
#     print("\n🔄 Обновление весов связей:")
#     kg.update_relationship_weights("furnace_temperature", "crack", 
#                                    observed_effect=True, effect_strength=0.85)
#     
#     # Получение улучшенных рекомендаций
#     print("\n💡 Улучшенные рекомендации:")
#     current_params = {
#         "furnace_temperature": 1620.0,
#         "belt_speed": 175.0,
#         "mold_temperature": 350.0
#     }
#     
#     recommendations = kg.get_intervention_recommendations("crack", current_params)
#     for i, rec in enumerate(recommendations[:3], 1):
#         print(f"\n  {i}. {rec['action']}")
#         print(f"     Приоритет: {rec['priority']}")
#         print(f"     Confidence: {rec['confidence']:.2f}")
#         print(f"     Ожидаемый эффект: {rec['expected_impact']}")
#     
#     # Экспорт подграфа
#     print("\n📤 Экспорт подграфа для визуализации:")
#     subgraph = kg.export_subgraph("crack", max_depth=2)
#     print(f"  Узлов: {len(subgraph['nodes'])}")
#     print(f"  Связей: {len(subgraph['edges'])}")
#     
#     kg.close()
#     print("\n✅ Тестирование завершено!")