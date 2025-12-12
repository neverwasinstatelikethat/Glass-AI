"""
Enhanced Knowledge Graph for Glass Production System
Implements causal relationship modeling, root cause analysis, and recommendation engine
Uses Neo4j for graph storage and Redis for caching

Real-time Enrichment Pipeline:
- ML predictions (LSTM defects, GNN anomalies) -> KG nodes/edges
- RL recommendations -> KG recommendation nodes
- Human decisions -> update relationship weights
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
                
                session.run("""
                    CREATE CONSTRAINT cause_id IF NOT EXISTS 
                    FOR (c:Cause) REQUIRE c.cause_id IS UNIQUE
                """)
                
                session.run("""
                    CREATE CONSTRAINT recommendation_id IF NOT EXISTS 
                    FOR (r:Recommendation) REQUIRE r.recommendation_id IS UNIQUE
                """)
                
                # Create indexes for performance
                session.run("CREATE INDEX equipment_type IF NOT EXISTS FOR (e:Equipment) ON (e.type)")
                session.run("CREATE INDEX parameter_category IF NOT EXISTS FOR (p:Parameter) ON (p.category)")
                session.run("CREATE INDEX defect_severity IF NOT EXISTS FOR (d:Defect) ON (d.severity)")
                session.run("CREATE INDEX cause_confidence IF NOT EXISTS FOR (c:Cause) ON (c.confidence)")
                session.run("CREATE INDEX recommendation_priority IF NOT EXISTS FOR (r:Recommendation) ON (r.priority)")
                
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
                    MERGE (forming_pressure:Parameter {name: 'forming_pressure', category: 'pressure'})
                    MERGE (cooling_rate:Parameter {name: 'cooling_rate', category: 'thermal'})
                """)
                
                # Create common defects with SET to avoid constraint conflicts
                session.run("""
                    MERGE (crack:Defect {type: 'crack'})
                    SET crack.severity = 'HIGH'
                    MERGE (bubble:Defect {type: 'bubble'})
                    SET bubble.severity = 'MEDIUM'
                    MERGE (chip:Defect {type: 'chip'})
                    SET chip.severity = 'LOW'
                    MERGE (cloudiness:Defect {type: 'cloudiness'})
                    SET cloudiness.severity = 'MEDIUM'
                    MERGE (deformation:Defect {type: 'deformation'})
                    SET deformation.severity = 'HIGH'
                    MERGE (stain:Defect {type: 'stain'})
                    SET stain.severity = 'LOW'
                """)
                
                # Create relationship constraints - removed rel_id constraints as they don't exist on relationships
                # Relationships in Neo4j don't typically need unique constraints unless there's a specific business requirement
                # If unique constraints are needed, they should be implemented differently or with actual properties that exist
                
                # Ensure all labels exist by creating and immediately deleting sample nodes
                session.run("CREATE (:Cause {cause_id: 'init_cause'})")
                session.run("CREATE (:Recommendation {recommendation_id: 'init_rec'})")
                session.run("MATCH (c:Cause {cause_id: 'init_cause'}) DELETE c")
                session.run("MATCH (r:Recommendation {recommendation_id: 'init_rec'}) DELETE r")
                
                logger.info("✅ Knowledge graph schema initialized with all required labels and relationships")
                
        except Exception as e:
            logger.error(f"❌ Error initializing knowledge base: {e}")
            # Try to create nodes even if constraints fail
            try:
                with self.driver.session() as session:
                    # Create sample nodes to ensure labels exist
                    session.run("CREATE (:Cause {cause_id: 'sample_cause', parameter: 'test'})")
                    session.run("CREATE (:Recommendation {recommendation_id: 'sample_rec', action: 'test'})")
                    session.run("MATCH (c:Cause {cause_id: 'sample_cause'}) DELETE c")
                    session.run("MATCH (r:Recommendation {recommendation_id: 'sample_rec'}) DELETE r")
                    logger.info("✅ Fallback node creation successful")
            except Exception as fallback_e:
                logger.error(f"❌ Fallback node creation failed: {fallback_e}")
    
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
                # Use transaction for better error handling
                with session.begin_transaction() as tx:
                    result = tx.run("""
                        MATCH (defect:Defect {type: $defect_type})<-[:CAUSES]-(cause:Cause)
                        WHERE cause.confidence >= $min_confidence
                        RETURN coalesce(cause.parameter, 'unknown') AS parameter, 
                               coalesce(cause.equipment, 'unknown') AS equipment,
                               cause.confidence AS confidence,
                               cause.description AS description,
                               elementId(cause) AS cause_id
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
                            "description": record["description"],
                            "cause_id": record["cause_id"]
                        })
                    
                    return causes
                
        except neo4j.exceptions.ServiceUnavailable as e:
            logger.error(f"❌ Neo4j service unavailable: {e}")
            return []
        except neo4j.exceptions.ClientError as e:
            logger.error(f"❌ Neo4j client error: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error getting causes of defect: {e}")
            return []
    
    def query_causes(self, effect: str) -> List[Dict[str, Any]]:
        """Query causes for a given effect (used by RootCauseAnalyzer)"""
        if not self.driver:
            logger.warning("⚠️ Neo4j not available, returning empty causes")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (defect:Defect {type: $defect_type})<-[:CAUSES]-(cause:Cause)
                    RETURN cause.parameter AS cause,
                           cause.confidence AS confidence,
                           cause.description AS mechanism,
                           elementId(cause) AS cause_id
                    ORDER BY cause.confidence DESC
                """, {
                    "defect_type": effect
                })
                
                causes = []
                for record in result:
                    causes.append({
                        "cause": record["cause"],
                        "confidence": record["confidence"],
                        "mechanism": record["mechanism"],
                        "cause_id": record["cause_id"]
                    })
                
                return causes
                
        except Exception as e:
            logger.error(f"❌ Error querying causes: {e}")
            # Return fallback causes if Neo4j query fails
            return [
                {
                    "cause": "furnace_temperature",
                    "confidence": 0.85,
                    "mechanism": "High temperature causes thermal stress",
                    "cause_id": "fallback_1"
                },
                {
                    "cause": "cooling_rate",
                    "confidence": 0.90,
                    "mechanism": "Rapid cooling causes thermal shock",
                    "cause_id": "fallback_2"
                }
            ]
    
    def add_cause(
        self,
        cause: str,
        effect: str,
        confidence: float,
        mechanism: Optional[str] = None
    ):
        """Add a cause-effect relationship to the knowledge graph"""
        if not self.driver:
            logger.debug("⚠️ Neo4j not available, skipping cause addition")
            return
        
        try:
            with self.driver.session() as session:
                # For parameter-defect causations
                if '_' in cause and any(op in cause for op in ['>', '<', 'var']):
                    # Parse parameter, condition, and threshold from cause
                    parts = cause.split('_')
                    if len(parts) >= 3:
                        parameter_name = parts[0]
                        condition = parts[1]
                        threshold = float(parts[2])
                        
                        # Create causal relationship
                        session.run("""
                            MERGE (param:Parameter {name: $param_name})
                            MERGE (defect:Defect {type: $defect_type})
                            CREATE (cause:Cause {
                                parameter: $param_name,
                                equipment: 'unknown',
                                confidence: $confidence,
                                description: $mechanism,
                                timestamp: $timestamp
                            })-[:CAUSES]->(defect)
                            CREATE (cause)-[:RELATED_TO]->(param)
                        """, {
                            "param_name": parameter_name,
                            "defect_type": effect,
                            "confidence": confidence,
                            "mechanism": mechanism,
                            "timestamp": datetime.now().isoformat()
                        })
                else:
                    # For equipment-parameter or other relationships
                    # Ensure the cause node has the proper properties
                    session.run("""
                        MERGE (defect:Defect {type: $effect})
                        CREATE (cause:Cause {
                            parameter: $cause,
                            equipment: 'unknown',
                            confidence: $confidence,
                            description: $mechanism,
                            timestamp: $timestamp
                        })-[:CAUSES]->(defect)
                    """, {
                        "cause": cause,
                        "effect": effect,
                        "confidence": confidence,
                        "mechanism": mechanism,
                        "timestamp": datetime.now().isoformat()
                    })
        except Exception as e:
            logger.error(f"❌ Error adding cause: {e}")
    
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
                        parameter: $param_name,
                        equipment: $equip_id,
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
    
    def add_equipment(self, equipment_id: str, equipment_type: str, zone: Optional[str] = None):
        """Add an equipment node to the knowledge graph"""
        if not self.driver:
            logger.debug("⚠️ Neo4j not available, skipping equipment addition")
            return
        
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (equip:Equipment {equipment_id: $equip_id})
                    SET equip.type = $type, 
                        equip.zone = $zone,
                        equip.last_updated = $timestamp
                """, {
                    "equip_id": equipment_id,
                    "type": equipment_type,
                    "zone": zone,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"❌ Error adding equipment: {e}")
    
    def add_recommendation(self, defect_type: str, action: str, 
                          expected_impact: str, priority: str = "MEDIUM",
                          description: Optional[str] = None,
                          target_parameter: Optional[str] = None,
                          target_value: Optional[float] = None,
                          unit: Optional[str] = None):
        """Add a recommendation node to the knowledge graph"""
        if not self.driver:
            logger.debug("⚠️ Neo4j not available, skipping recommendation addition")
            return
        
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (defect:Defect {type: $defect_type})
                    CREATE (rec:Recommendation {
                        action: $action,
                        expected_impact: $expected_impact,
                        priority: $priority,
                        description: $description,
                        target_parameter: $target_parameter,
                        target_value: $target_value,
                        unit: $unit,
                        timestamp: $timestamp,
                        source: 'KnowledgeBaseInitializer'
                    })-[:ADDRESSES]->(defect)
                """, {
                    "defect_type": defect_type,
                    "action": action,
                    "expected_impact": expected_impact,
                    "priority": priority,
                    "description": description,
                    "target_parameter": target_parameter,
                    "target_value": target_value,
                    "unit": unit,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"❌ Error adding recommendation: {e}")
    
    def get_ml_enhanced_recommendations(self, defect_type: str, 
                                      ml_predictions: Dict[str, float],
                                      current_parameters: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get ML-enhanced recommendations that combine KG insights with ML predictions"""
        if not self.driver:
            logger.warning("⚠️ Neo4j not available, returning empty recommendations")
            return []
        
        try:
            # Get base recommendations from knowledge graph
            base_recommendations = self.get_intervention_recommendations(defect_type, current_parameters)
            
            # Enhance with ML predictions
            enhanced_recommendations = []
            
            for rec in base_recommendations:
                parameter = rec["parameter"]
                
                # Get ML prediction for this parameter if available
                ml_confidence = ml_predictions.get(parameter, 0.5)
                
                # Combine KG confidence with ML confidence
                combined_confidence = (rec["confidence"] + ml_confidence) / 2
                
                # Adjust expected impact based on ML prediction strength
                base_impact = rec["expected_impact"]
                if isinstance(base_impact, str) and '%' in base_impact:
                    # Extract percentage and adjust based on ML confidence
                    try:
                        impact_percent = int(base_impact.split('%')[0])
                        adjusted_impact = int(impact_percent * (0.7 + 0.3 * ml_confidence))
                        enhanced_impact = f"{adjusted_impact}% improvement estimated"
                    except:
                        enhanced_impact = base_impact
                else:
                    enhanced_impact = base_impact
                
                enhanced_recommendations.append({
                    **rec,
                    "confidence": combined_confidence,
                    "expected_impact": enhanced_impact,
                    "ml_enhanced": True,
                    "ml_confidence": ml_confidence
                })
            
            # Sort by combined confidence
            enhanced_recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            
            return enhanced_recommendations
            
        except Exception as e:
            logger.error(f"❌ Error getting ML-enhanced recommendations: {e}")
            # Fallback to base recommendations
            return self.get_intervention_recommendations(defect_type, current_parameters)
    
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
                    MATCH (param:Parameter {name: $param_name})<-[:RELATED_TO]-(cause:Cause)-[:FROM_EQUIPMENT]->(equip:Equipment)
                    MATCH (defect:Defect {type: $defect_type})<-[:CAUSES]-(cause)
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
                # Get all recommendations for this defect, not just those linked to causes
                result = session.run("""
                    MATCH (defect:Defect {type: $defect_type})<-[:ADDRESSES]-(rec:Recommendation)
                    OPTIONAL MATCH (cause:Cause)-[:CAUSES]->(defect)
                    WHERE coalesce(cause.parameter, '') = coalesce(rec.target_parameter, '')
                    RETURN rec.action AS action,
                           rec.target_parameter AS parameter,
                           rec.target_value AS target_value,
                           rec.unit AS unit,
                           rec.priority AS priority,
                           rec.expected_impact AS expected_impact,
                           rec.description AS description,
                           cause.confidence AS cause_confidence
                    ORDER BY rec.priority DESC, cause.confidence DESC
                """, {
                    "defect_type": defect_type
                })
                
                recommendations = []
                for record in result:
                    parameter = record["parameter"] or "general"
                    current_value = current_parameters.get(parameter, 0) if parameter != "general" else 0
                    target_value = record["target_value"] or (current_value * 0.95 if parameter != "general" else 0)
                    
                    # Calculate impact based on current vs target values
                    if parameter != "general" and current_value != 0:
                        deviation = abs(current_value - target_value)
                        impact_factor = min(1.0, deviation / max(1.0, abs(current_value)))
                    else:
                        impact_factor = 0.5
                    
                    # Use cause confidence if available, otherwise use default
                    cause_confidence = record["cause_confidence"] or 0.7
                    
                    # Adjust confidence based on how far we are from target
                    adjusted_confidence = cause_confidence * (0.5 + 0.5 * impact_factor)
                    
                    recommendations.append({
                        "parameter": parameter,
                        "current_value": current_value,
                        "target_value": target_value,
                        "unit": record["unit"] or "",
                        "action": record["action"] or f"Adjust {parameter} towards optimal value",
                        "confidence": adjusted_confidence,
                        "priority": record["priority"] or "MEDIUM",
                        "expected_impact": record["expected_impact"] or f"{int(impact_factor * 100)}% improvement estimated",
                        "mechanism": record["description"] or "Based on expert knowledge"
                    })
                
                # If no recommendations found, get general recommendations
                if not recommendations:
                    general_result = session.run("""
                        MATCH (defect:Defect {type: $defect_type})<-[:ADDRESSES]-(rec:Recommendation)
                        WHERE rec.target_parameter IS NULL
                        RETURN rec.action AS action,
                               rec.target_value AS target_value,
                               rec.unit AS unit,
                               rec.priority AS priority,
                               rec.expected_impact AS expected_impact,
                               rec.description AS description
                        ORDER BY rec.priority DESC
                    """, {
                        "defect_type": defect_type
                    })
                    
                    for record in general_result:
                        recommendations.append({
                            "parameter": "general",
                            "current_value": 0,
                            "target_value": record["target_value"] or 0,
                            "unit": record["unit"] or "",
                            "action": record["action"] or "General recommendation",
                            "confidence": 0.7,
                            "priority": record["priority"] or "MEDIUM",
                            "expected_impact": record["expected_impact"] or "Significant improvement expected",
                            "mechanism": record["description"] or "Based on expert knowledge"
                        })
                
                # Sort by confidence
                recommendations.sort(key=lambda x: x["confidence"], reverse=True)
                
                return recommendations
                
        except Exception as e:
            logger.error(f"❌ Error getting intervention recommendations: {e}")
            return []
    
    def export_subgraph(self, defect_type: str, max_depth: int = 2, include_recommendations: bool = True, 
                 include_human_decisions: bool = True) -> Dict[str, Any]:
        """Export subgraph for visualization with complete graph data
        
        Args:
            defect_type: Type of defect to visualize
            max_depth: Maximum depth of relationships to include
            include_recommendations: Whether to include recommendation nodes
            include_human_decisions: Whether to include human decision nodes
            
        Returns:
            Dict containing nodes and edges for visualization
        """
        if not self.driver:
            logger.warning("⚠️ Neo4j not available, returning empty subgraph")
            return {"nodes": [], "edges": []}
        
        try:
            nodes = []
            edges = []
            node_ids = set()
            
            with self.driver.session() as session:
                # Use transaction for better error handling and consistency
                with session.begin_transaction() as tx:
                    # Get defect node
                    defect_result = tx.run("""
                        MATCH (d:Defect {type: $defect_type})
                        RETURN d.type AS type, elementId(d) AS id, d.severity AS severity, d.description AS description
                    """, {
                        "defect_type": defect_type
                    })
                    
                    defect_node = None
                    for record in defect_result:
                        defect_node = {
                            "id": record["id"],
                            "label": record["type"],
                            "name": record["type"],
                            "properties": {
                                "type": record["type"],
                                "severity": record["severity"],
                                "description": record["description"]
                            },
                            "nodeType": "defect",
                            "confidence": 1.0
                        }
                        nodes.append(defect_node)
                        node_ids.add(record["id"])
                    
                    if not defect_node:
                        logger.info(f"No defect node found for type: {defect_type}")
                        return {"nodes": [], "edges": []}
                    
                    # Get related causes, parameters and equipment
                    causes_result = tx.run("""
                        MATCH (d:Defect {type: $defect_type})<-[:CAUSES]-(cause:Cause)
                        OPTIONAL MATCH (cause)-[:RELATED_TO]->(param:Parameter)
                        OPTIONAL MATCH (cause)-[:FROM_EQUIPMENT]->(equip:Equipment)
                        RETURN cause.confidence AS confidence,
                               cause.description AS description,
                               elementId(cause) AS cause_id,
                               coalesce(cause.parameter, 'unknown') AS cause_parameter,
                               coalesce(param.name, 'unknown') AS param_name,
                               elementId(param) AS param_id,
                               coalesce(equip.equipment_id, 'unknown') AS equip_id,
                               elementId(equip) AS equip_id_internal
                    """, {
                        "defect_type": defect_type
                    })
                    
                    for record in causes_result:
                        # Add cause node
                        if record["cause_id"] and record["cause_id"] not in node_ids:
                            cause_node = {
                                "id": record["cause_id"],
                                "label": record["cause_parameter"],
                                "name": record["cause_parameter"],
                                "properties": {
                                    "parameter": record["cause_parameter"],
                                    "description": record["description"] or "Cause of defect"
                                },
                                "nodeType": "cause",
                                "confidence": record["confidence"] or 0.5
                            }
                            nodes.append(cause_node)
                            node_ids.add(record["cause_id"])
                        
                        # Add parameter node
                        if record["param_id"] and record["param_id"] not in node_ids:
                            param_node = {
                                "id": record["param_id"],
                                "label": record["param_name"],
                                "name": record["param_name"],
                                "properties": {"name": record["param_name"]},
                                "nodeType": "parameter",
                                "confidence": record["confidence"] or 0.5
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
                                "nodeType": "equipment",
                                "confidence": record["confidence"] or 0.5
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
                    
                    # Include recommendation nodes if requested
                    if include_recommendations:
                        rec_result = tx.run("""
                            MATCH (d:Defect {type: $defect_type})<-[:ADDRESSES]-(rec:Recommendation)
                            RETURN rec.recommendation_id AS rec_id,
                                   rec.action AS action,
                                   rec.confidence AS confidence,
                                   rec.urgency AS urgency,
                                   rec.expected_impact AS expected_impact,
                                   rec.applied AS applied,
                                   elementId(rec) AS id
                            ORDER BY rec.timestamp DESC
                            LIMIT 10
                        """, {
                            "defect_type": defect_type
                        })
                        
                        for record in rec_result:
                            if record["id"] and record["id"] not in node_ids:
                                rec_node = {
                                    "id": record["id"],
                                    "label": record["action"] or "Recommendation",
                                    "name": record["action"] or "Recommendation",
                                    "properties": {
                                        "recommendation_id": record["rec_id"],
                                        "action": record["action"],
                                        "urgency": record["urgency"],
                                        "expected_impact": record["expected_impact"],
                                        "applied": record["applied"]
                                    },
                                    "nodeType": "recommendation",
                                    "confidence": record["confidence"] or 0.5
                                }
                                nodes.append(rec_node)
                                node_ids.add(record["id"])
                            
                            # Edge from recommendation to defect
                            if record["id"] and defect_node["id"]:
                                edges.append({
                                    "source": record["id"],
                                    "target": defect_node["id"],
                                    "type": "ADDRESSES",
                                    "confidence": record["confidence"] or 0.5,
                                    "strength": record["confidence"] or 0.5
                                })
                    
                    # Include human decision nodes if requested
                    if include_human_decisions:
                        decision_result = tx.run("""
                            MATCH (d:Defect {type: $defect_type})<-[:REGARDING]-(hd:HumanDecision)
                            RETURN hd.decision_id AS decision_id,
                                   hd.decision AS decision,
                                   hd.notes AS notes,
                                   hd.timestamp AS timestamp,
                                   elementId(hd) AS id
                            ORDER BY hd.timestamp DESC
                            LIMIT 10
                        """, {
                            "defect_type": defect_type
                        })
                        
                        for record in decision_result:
                            if record["id"] and record["id"] not in node_ids:
                                decision_node = {
                                    "id": record["id"],
                                    "label": record["decision"].upper() if record["decision"] else "DECISION",
                                    "name": record["decision"].upper() if record["decision"] else "DECISION",
                                    "properties": {
                                        "decision_id": record["decision_id"],
                                        "decision": record["decision"],
                                        "notes": record["notes"],
                                        "timestamp": record["timestamp"]
                                    },
                                    "nodeType": "human_decision",
                                    "confidence": 0.8  # Default confidence for human decisions
                                }
                                nodes.append(decision_node)
                                node_ids.add(record["id"])
                            
                            # Edge from decision to defect
                            if record["id"] and defect_node["id"]:
                                edges.append({
                                    "source": record["id"],
                                    "target": defect_node["id"],
                                    "type": "REGARDING",
                                    "confidence": 0.8,
                                    "strength": 0.8
                                })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "defect": defect_type,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "max_depth": max_depth,
                    "include_recommendations": include_recommendations,
                    "include_human_decisions": include_human_decisions
                }
            }
            
        except neo4j.exceptions.ServiceUnavailable as e:
            logger.error(f"❌ Neo4j service unavailable while exporting subgraph: {e}")
            return {"nodes": [], "edges": [], "error": "Neo4j service unavailable"}
        except neo4j.exceptions.ClientError as e:
            logger.error(f"❌ Neo4j client error while exporting subgraph: {e}")
            return {"nodes": [], "edges": [], "error": "Neo4j client error"}
        except Exception as e:
            logger.error(f"❌ Unexpected error exporting subgraph: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}

    # ==================== REAL-TIME ENRICHMENT PIPELINE ====================
    
    def enrich_from_ml_prediction(
        self,
        defect_type: str,
        probability: float,
        severity: str,
        sensor_snapshot: Dict[str, float],
        model_source: str = "LSTM",
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Enrich KG from ML prediction (LSTM/GNN).
        Creates defect occurrence and updates parameter-defect causal weights.
        
        Args:
            defect_type: Type of defect (crack, bubble, etc.)
            probability: ML prediction probability
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            sensor_snapshot: Current sensor readings
            model_source: ML model that made prediction (LSTM, GNN, etc.)
            timestamp: Prediction timestamp
            
        Returns:
            Enrichment result with created nodes/edges
        """
        timestamp = timestamp or datetime.now()
        result = {
            "enriched": False,
            "defect_added": False,
            "defect_occurrence_id": None,
            "relationships_updated": [],
            "measurements_added": [],
            "timestamp": timestamp.isoformat()
        }
        
        # Add defect occurrence with more detailed information
        try:
            with self.driver.session() as session:
                # Create defect occurrence with detailed metadata
                defect_occurrence_id = f"defect_occ_{timestamp.timestamp()}_{defect_type}"
                session.run("""
                    MERGE (defect:Defect {type: $defect_type})
                    CREATE (occ:DefectOccurrence {
                        occurrence_id: $occurrence_id,
                        probability: $probability,
                        severity: $severity,
                        model_source: $model_source,
                        timestamp: $timestamp,
                        description: $description
                    })-[:IS_INSTANCE_OF]->(defect)
                """, {
                    "defect_type": defect_type,
                    "occurrence_id": defect_occurrence_id,
                    "probability": probability,
                    "severity": severity,
                    "model_source": model_source,
                    "timestamp": timestamp.isoformat(),
                    "description": f"ML prediction from {model_source}: {probability:.1%} confidence"
                })
                result["defect_occurrence_id"] = defect_occurrence_id
                result["defect_added"] = True
        except Exception as e:
            logger.error(f"❌ Error adding defect occurrence: {e}")
        
        # Add measurements for all sensor data
        for param_name, current_value in sensor_snapshot.items():
            try:
                self.add_measurement(
                    parameter_name=param_name,
                    value=current_value,
                    timestamp=timestamp,
                    equipment_id="prediction_source",
                    sensor_id=f"{model_source}_sensor"
                )
                result["measurements_added"].append({
                    "parameter": param_name,
                    "value": current_value
                })
            except Exception as e:
                logger.warning(f"⚠️ Could not add measurement for {param_name}: {e}")
        
        # Update causal relationships based on sensor readings with enhanced logic
        # High values increase confidence, low values decrease
        sensor_thresholds = {
            "furnace_temperature": (1400, 1600),
            "furnace_pressure": (10, 20),
            "belt_speed": (120, 180),
            "mold_temperature": (250, 400),
            "forming_pressure": (30, 70),
            "annealing_temperature": (500, 700),
            "cooling_rate": (2, 5),
            "oxygen_content": (0, 5)
        }
        
        updated_relationships = []
        
        for param_name, current_value in sensor_snapshot.items():
            if param_name in sensor_thresholds:
                min_val, max_val = sensor_thresholds[param_name]
                
                # Check if parameter is anomalous
                is_anomalous = current_value < min_val or current_value > max_val
                deviation = 0
                if current_value < min_val:
                    deviation = (min_val - current_value) / min_val
                elif current_value > max_val:
                    deviation = (current_value - max_val) / max_val
                
                # Update relationship weight if parameter is anomalous
                if is_anomalous and deviation > 0.05:
                    effect_strength = min(1.0, probability * (1 + deviation))
                    
                    # More sophisticated relationship weight update
                    with self.driver.session() as session:
                        # Find existing causal relationships
                        rel_result = session.run("""
                            MATCH (param:Parameter {name: $param_name})<-[:RELATED_TO]-(cause:Cause)-[:CAUSES]->(defect:Defect {type: $defect_type})
                            RETURN cause, elementId(cause) as cause_id
                        """, {
                            "param_name": param_name,
                            "defect_type": defect_type
                        })
                        
                        for record in rel_result:
                            cause_node = record["cause"]
                            cause_id = record["cause_id"]
                            
                            # Update confidence based on observation
                            current_confidence = cause_node.get("confidence", 0.5)
                            # Weighted update: new_confidence = (old_confidence * 0.7) + (effect_strength * 0.3)
                            new_confidence = (current_confidence * 0.7) + (effect_strength * 0.3)
                            new_confidence = min(1.0, max(0.1, new_confidence))  # Clamp between 0.1 and 1.0
                            
                            session.run("""
                                MATCH (cause:Cause)
                                WHERE elementId(cause) = $cause_id
                                SET cause.confidence = $new_confidence,
                                    cause.last_updated = $timestamp,
                                    cause.observation_count = coalesce(cause.observation_count, 0) + 1
                            """, {
                                "cause_id": cause_id,
                                "new_confidence": new_confidence,
                                "timestamp": timestamp.isoformat()
                            })
                            
                            updated_relationships.append({
                                "parameter": param_name,
                                "defect": defect_type,
                                "effect_strength": effect_strength,
                                "new_confidence": new_confidence,
                                "observation_count": cause_node.get("observation_count", 0) + 1
                            })
                
                # Even if not anomalous, still log the observation for learning
                else:
                    # For non-anomalous values, slightly decrease confidence if it was high
                    with self.driver.session() as session:
                        rel_result = session.run("""
                            MATCH (param:Parameter {name: $param_name})<-[:RELATED_TO]-(cause:Cause)-[:CAUSES]->(defect:Defect {type: $defect_type})
                            RETURN cause, elementId(cause) as cause_id
                        """, {
                            "param_name": param_name,
                            "defect_type": defect_type
                        })
                        
                        for record in rel_result:
                            cause_node = record["cause"]
                            cause_id = record["cause_id"]
                            
                            current_confidence = cause_node.get("confidence", 0.5)
                            # If confidence was high but no correlation observed, slightly decrease it
                            if current_confidence > 0.7:
                                new_confidence = max(0.1, current_confidence * 0.95)  # Small decrease
                                
                                session.run("""
                                    MATCH (cause:Cause)
                                    WHERE elementId(cause) = $cause_id
                                    SET cause.confidence = $new_confidence,
                                        cause.last_updated = $timestamp,
                                        cause.observation_count = coalesce(cause.observation_count, 0) + 1
                                """, {
                                    "cause_id": cause_id,
                                    "new_confidence": new_confidence,
                                    "timestamp": timestamp.isoformat()
                                })
                                
                                updated_relationships.append({
                                    "parameter": param_name,
                                    "defect": defect_type,
                                    "effect_strength": 0,  # No effect observed
                                    "new_confidence": new_confidence,
                                    "observation_count": cause_node.get("observation_count", 0) + 1
                                })
        
        result["relationships_updated"] = updated_relationships
        result["enriched"] = True
        logger.info(f"🧠 KG enriched from {model_source} prediction: {defect_type} ({severity}), {len(updated_relationships)} relationships updated")
        
        # Add monitoring and logging
        if self.cache_enabled:
            try:
                # Cache enrichment metrics for monitoring
                metrics_key = f"ml_enrichment_metrics:{defect_type}:{timestamp.date().isoformat()}"
                metrics_data = {
                    "defect_type": defect_type,
                    "probability": probability,
                    "severity": severity,
                    "model_source": model_source,
                    "relationships_updated": len(updated_relationships),
                    "measurements_added": len(result["measurements_added"]),
                    "timestamp": timestamp.isoformat()
                }
                self.redis_client.lpush(metrics_key, json.dumps(metrics_data))
                self.redis_client.expire(metrics_key, 86400)  # Expire after 24 hours
            except Exception as e:
                logger.warning(f"⚠️ Error caching ML enrichment metrics: {e}")
        
        return result
    
    def enrich_from_rl_recommendation(
        self,
        defect_type: str,
        recommendation: Dict[str, Any],
        expected_impact: float = 0.5,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Enrich KG from RL agent recommendation.
        Creates recommendation node linked to defect with proper tracking and effectiveness measurement.
        
        Args:
            defect_type: Type of defect being addressed
            recommendation: RL recommendation data (action, parameters, confidence)
            expected_impact: Expected improvement percentage
            timestamp: Recommendation timestamp
            
        Returns:
            Enrichment result
        """
        timestamp = timestamp or datetime.now()
        result = {
            "enriched": False,
            "recommendation_id": None,
            "tracking_id": None,
            "timestamp": timestamp.isoformat()
        }
        
        if not self.driver:
            logger.debug("⚠️ Neo4j not available, skipping RL recommendation enrichment")
            return result
        
        try:
            # Generate unique identifiers
            rec_id = f"rl_rec_{timestamp.timestamp()}_{defect_type}"
            tracking_id = f"track_{timestamp.timestamp()}_{defect_type}"
            
            # Extract recommendation details
            action = recommendation.get("action", "unknown")
            parameters = recommendation.get("parameters", {})
            confidence = recommendation.get("confidence", 0.5)
            urgency = recommendation.get("urgency", "MEDIUM")
            priority = recommendation.get("priority", "MEDIUM")
            target_parameter = recommendation.get("target_parameter")
            target_value = recommendation.get("target_value")
            unit = recommendation.get("unit")
            
            with self.driver.session() as session:
                # Create recommendation node with complete properties
                session.run("""
                    MERGE (defect:Defect {type: $defect_type})
                    CREATE (rec:Recommendation {
                        recommendation_id: $rec_id,
                        tracking_id: $tracking_id,
                        action: $action,
                        parameters: $parameters,
                        target_parameter: $target_parameter,
                        target_value: $target_value,
                        unit: $unit,
                        confidence: $confidence,
                        urgency: $urgency,
                        priority: $priority,
                        expected_impact: $expected_impact,
                        source: 'RL_Agent',
                        timestamp: $timestamp,
                        applied: false,
                        application_count: 0,
                        effectiveness_score: 0.0
                    })-[:ADDRESSES]->(defect)
                """, {
                    "defect_type": defect_type,
                    "rec_id": rec_id,
                    "tracking_id": tracking_id,
                    "action": action,
                    "parameters": json.dumps(parameters) if parameters else "{}",
                    "target_parameter": target_parameter,
                    "target_value": target_value,
                    "unit": unit,
                    "confidence": confidence,
                    "urgency": urgency,
                    "priority": priority,
                    "expected_impact": expected_impact,
                    "timestamp": timestamp.isoformat()
                })
                
                result["recommendation_id"] = rec_id
                result["tracking_id"] = tracking_id
                result["enriched"] = True
                
                logger.info(f"🎮 KG enriched from RL: {action} for {defect_type} (ID: {rec_id})")
                
                # Add recommendation tracking for effectiveness measurement
                session.run("""
                    CREATE (track:RecommendationTracking {
                        tracking_id: $tracking_id,
                        recommendation_id: $rec_id,
                        defect_type: $defect_type,
                        action: $action,
                        expected_impact: $expected_impact,
                        timestamp: $timestamp,
                        status: 'PENDING',
                        feedback_count: 0
                    })
                """, {
                    "tracking_id": tracking_id,
                    "rec_id": rec_id,
                    "defect_type": defect_type,
                    "action": action,
                    "expected_impact": expected_impact,
                    "timestamp": timestamp.isoformat()
                })
                
        except Exception as e:
            logger.error(f"❌ Error enriching KG from RL: {e}")
            result["error"] = str(e)
        
        # Add monitoring and caching for tracking
        if self.cache_enabled:
            try:
                # Cache recommendation for quick access
                cache_key = f"rl_recommendation:{rec_id}"
                cache_data = {
                    "recommendation_id": rec_id,
                    "defect_type": defect_type,
                    "action": action,
                    "confidence": confidence,
                    "expected_impact": expected_impact,
                    "timestamp": timestamp.isoformat(),
                    "status": "created"
                }
                self.redis_client.setex(cache_key, 86400, json.dumps(cache_data))  # 24 hours
                
                # Add to recent recommendations list
                recent_key = f"recent_rl_recommendations:{defect_type}"
                self.redis_client.lpush(recent_key, rec_id)
                self.redis_client.ltrim(recent_key, 0, 49)  # Keep last 50 recommendations
            except Exception as e:
                logger.warning(f"⚠️ Error caching RL recommendation: {e}")
        
        return result
    
    def enrich_from_human_decision(
        self,
        notification_id: str,
        decision: str,  # "applied", "dismissed", "modified"
        defect_type: str,
        recommendation_id: Optional[str] = None,
        notes: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Enrich KG from human operator decision.
        Updates recommendation effectiveness and causal weights with complete integration.
        
        Args:
            notification_id: ID of the notification acted upon
            decision: Human decision (applied, dismissed, modified)
            defect_type: Related defect type
            recommendation_id: Related RL recommendation ID
            notes: Operator notes
            timestamp: Decision timestamp
            
        Returns:
            Enrichment result
        """
        timestamp = timestamp or datetime.now()
        result = {
            "enriched": False,
            "decision_recorded": False,
            "decision_id": None,
            "recommendation_updated": False,
            "causal_weights_updated": False,
            "rl_feedback_provided": False,
            "timestamp": timestamp.isoformat()
        }
        
        if not self.driver:
            logger.debug("⚠️ Neo4j not available, skipping human decision enrichment")
            return result
        
        try:
            decision_id = f"human_dec_{timestamp.timestamp()}_{defect_type}"
            
            with self.driver.session() as session:
                # Record human decision as event with complete properties
                session.run("""
                    MERGE (defect:Defect {type: $defect_type})
                    CREATE (decision:HumanDecision {
                        decision_id: $decision_id,
                        notification_id: $notification_id,
                        decision: $decision_type,
                        notes: $notes,
                        timestamp: $timestamp,
                        operator_id: 'default_operator'  # In real implementation, this would come from auth context
                    })-[:REGARDING]->(defect)
                """, {
                    "defect_type": defect_type,
                    "decision_id": decision_id,
                    "notification_id": notification_id,
                    "decision_type": decision,
                    "notes": notes,
                    "timestamp": timestamp.isoformat()
                })
                
                result["decision_recorded"] = True
                result["decision_id"] = decision_id
                
                # If recommendation was applied, update its status and effectiveness
                if recommendation_id:
                    # Get current recommendation data for effectiveness calculation
                    rec_result = session.run("""
                        MATCH (rec:Recommendation {recommendation_id: $rec_id})
                        RETURN rec.expected_impact as expected_impact,
                               rec.application_count as application_count,
                               rec.effectiveness_score as effectiveness_score
                    """, {"rec_id": recommendation_id})
                    
                    rec_data = rec_result.single() if rec_result.peek() else None
                    
                    if rec_data:
                        # Update recommendation based on decision
                        if decision == "applied":
                            # Update recommendation status and effectiveness
                            new_application_count = (rec_data.get("application_count", 0) or 0) + 1
                            
                            # Calculate effectiveness score (simplified)
                            # In real implementation, this would be based on actual defect reduction
                            effectiveness_score = min(1.0, (rec_data.get("effectiveness_score", 0.0) or 0.0) + 0.1)
                            
                            session.run("""
                                MATCH (rec:Recommendation {recommendation_id: $rec_id})
                                SET rec.applied = true,
                                    rec.applied_timestamp = $timestamp,
                                    rec.applied_by = 'operator',
                                    rec.application_count = $application_count,
                                    rec.effectiveness_score = $effectiveness_score
                            """, {
                                "rec_id": recommendation_id,
                                "timestamp": timestamp.isoformat(),
                                "application_count": new_application_count,
                                "effectiveness_score": effectiveness_score
                            })
                            
                            result["recommendation_updated"] = True
                        
                        # Update recommendation tracking
                        session.run("""
                            MATCH (track:RecommendationTracking {recommendation_id: $rec_id})
                            SET track.status = $status,
                                track.feedback_count = track.feedback_count + 1,
                                track.last_updated = $timestamp
                        """, {
                            "rec_id": recommendation_id,
                            "status": "APPLIED" if decision == "applied" else decision.upper(),
                            "timestamp": timestamp.isoformat()
                        })
                
                # Update causal weights based on decision with enhanced logic
                effect_strength = 0.7 if decision == "applied" else (0.3 if decision == "dismissed" else 0.5)
                
                # Update causal relationship weights based on human feedback
                if recommendation_id:
                    # Find the causal relationships related to this recommendation
                    causal_result = session.run("""
                        MATCH (rec:Recommendation {recommendation_id: $rec_id})-[:ADDRESSES]->(defect:Defect)
                        MATCH (cause:Cause)-[:CAUSES]->(defect)
                        RETURN cause, elementId(cause) as cause_id
                    """, {"rec_id": recommendation_id})
                    
                    for record in causal_result:
                        cause_node = record["cause"]
                        cause_id = record["cause_id"]
                        
                        current_confidence = cause_node.get("confidence", 0.5)
                        # Weighted update based on human feedback
                        new_confidence = (current_confidence * 0.8) + (effect_strength * 0.2)
                        new_confidence = min(1.0, max(0.1, new_confidence))
                        
                        session.run("""
                            MATCH (cause:Cause)
                            WHERE elementId(cause) = $cause_id
                            SET cause.confidence = $new_confidence,
                                cause.last_updated = $timestamp,
                                cause.human_feedback_count = coalesce(cause.human_feedback_count, 0) + 1
                        """, {
                            "cause_id": cause_id,
                            "new_confidence": new_confidence,
                            "timestamp": timestamp.isoformat()
                        })
                        
                        result["causal_weights_updated"] = True
                
                # Provide RL feedback for continuous learning
                if self.cache_enabled:
                    feedback_key = f"rl_feedback:{defect_type}:{timestamp.timestamp()}"
                    feedback_data = {
                        "decision": decision,
                        "recommendation_id": recommendation_id,
                        "effect_strength": effect_strength,
                        "timestamp": timestamp.isoformat(),
                        "defect_type": defect_type
                    }
                    try:
                        self.redis_client.setex(feedback_key, 86400, json.dumps(feedback_data))  # 24h
                        result["rl_feedback_provided"] = True
                    except Exception as e:
                        logger.warning(f"⚠️ Error caching RL feedback: {e}")
                
                result["enriched"] = True
                logger.info(f"👤 KG enriched from human decision: {decision} for {defect_type} (ID: {decision_id})")
                
        except Exception as e:
            logger.error(f"❌ Error enriching KG from human decision: {e}")
            result["error"] = str(e)
        
        return result

    
    def get_defect_recommendation_graph(
        self,
        defect_type: str,
        include_human_decisions: bool = True,
        max_recommendations: int = 10
    ) -> Dict[str, Any]:
        """
        Get defect-recommendation relationship graph for visualization.
        Shows defect causes, recommendations, and human decision outcomes.
        
        Args:
            defect_type: Type of defect to analyze
            include_human_decisions: Include human decision history
            max_recommendations: Maximum recommendations to return
            
        Returns:
            Graph data for frontend visualization
        """
        if not self.driver:
            logger.warning("⚠️ Neo4j not available, returning mock data")
            return self._get_mock_defect_recommendation_graph(defect_type)
        
        try:
            nodes = []
            edges = []
            node_ids = set()
            
            with self.driver.session() as session:
                # Get defect node
                defect_result = session.run("""
                    MATCH (d:Defect {type: $defect_type})
                    RETURN d.type AS type, d.severity AS severity, elementId(d) AS id
                """, {"defect_type": defect_type})
                
                for record in defect_result:
                    defect_node = {
                        "id": f"defect_{record['id']}",
                        "label": record["type"].upper(),
                        "type": "defect",
                        "severity": record["severity"] or "MEDIUM",
                        "properties": {"type": record["type"]}
                    }
                    nodes.append(defect_node)
                    node_ids.add(defect_node["id"])
                
                # Get recommendations
                rec_result = session.run("""
                    MATCH (d:Defect {type: $defect_type})<-[:ADDRESSES]-(rec:Recommendation)
                    RETURN rec.recommendation_id AS rec_id,
                           rec.action AS action,
                           rec.confidence AS confidence,
                           rec.applied AS applied,
                           rec.expected_impact AS expected_impact,
                           rec.source AS source,
                           elementId(rec) AS id
                    ORDER BY rec.timestamp DESC
                    LIMIT $limit
                """, {"defect_type": defect_type, "limit": max_recommendations})
                
                for record in rec_result:
                    rec_node = {
                        "id": f"rec_{record['id']}",
                        "label": record["action"] or "Recommendation",
                        "type": "recommendation",
                        "applied": record["applied"] or False,
                        "confidence": record["confidence"] or 0.5,
                        "expected_impact": record["expected_impact"] or 0.5,
                        "source": record["source"] or "RL_Agent"
                    }
                    nodes.append(rec_node)
                    node_ids.add(rec_node["id"])
                    
                    # Edge from recommendation to defect
                    edges.append({
                        "source": rec_node["id"],
                        "target": f"defect_{defect_node['id'].split('_')[1]}",
                        "type": "ADDRESSES",
                        "confidence": record["confidence"] or 0.5
                    })
                
                # Get human decisions if requested
                if include_human_decisions:
                    decision_result = session.run("""
                        MATCH (d:Defect {type: $defect_type})<-[:REGARDING]-(hd:HumanDecision)
                        RETURN hd.notification_id AS notification_id,
                               hd.decision AS decision,
                               hd.notes AS notes,
                               hd.timestamp AS timestamp,
                               elementId(hd) AS id
                        ORDER BY hd.timestamp DESC
                        LIMIT 20
                    """, {"defect_type": defect_type})
                    
                    for record in decision_result:
                        decision_node = {
                            "id": f"decision_{record['id']}",
                            "label": record["decision"].upper(),
                            "type": "human_decision",
                            "decision": record["decision"],
                            "notes": record["notes"],
                            "timestamp": record["timestamp"]
                        }
                        nodes.append(decision_node)
                        node_ids.add(decision_node["id"])
                        
                        edges.append({
                            "source": decision_node["id"],
                            "target": f"defect_{defect_node['id'].split('_')[1]}",
                            "type": "REGARDING"
                        })
                
                # Get causes
                causes_result = session.run("""
                    MATCH (d:Defect {type: $defect_type})<-[:CAUSES]-(c:Cause)
                    OPTIONAL MATCH (c)-[:RELATED_TO]->(p:Parameter)
                    RETURN c.confidence AS confidence, 
                           p.name AS parameter,
                           elementId(c) AS cause_id,
                           elementId(p) AS param_id
                """, {"defect_type": defect_type})
                
                for record in causes_result:
                    if record["param_id"]:
                        param_node_id = f"param_{record['param_id']}"
                        if param_node_id not in node_ids:
                            param_node = {
                                "id": param_node_id,
                                "label": record["parameter"],
                                "type": "parameter",
                                "confidence": record["confidence"] or 0.5
                            }
                            nodes.append(param_node)
                            node_ids.add(param_node_id)
                        
                        edges.append({
                            "source": param_node_id,
                            "target": f"defect_{defect_node['id'].split('_')[1]}",
                            "type": "CAUSES",
                            "confidence": record["confidence"] or 0.5
                        })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "defect_type": defect_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting defect-recommendation graph: {e}")
            return self._get_mock_defect_recommendation_graph(defect_type)
    
    def _get_mock_defect_recommendation_graph(self, defect_type: str) -> Dict[str, Any]:
        """Return mock graph data when Neo4j is unavailable"""
        # Create mock data for frontend visualization
        mock_nodes = [
            {"id": "defect_1", "label": defect_type.upper(), "type": "defect", "severity": "HIGH"},
            {"id": "param_1", "label": "furnace_temperature", "type": "parameter", "confidence": 0.85},
            {"id": "param_2", "label": "belt_speed", "type": "parameter", "confidence": 0.72},
            {"id": "rec_1", "label": "Снизить температуру на 30°C", "type": "recommendation", "applied": False, "confidence": 0.88, "source": "RL_Agent"},
            {"id": "rec_2", "label": "Увеличить скорость ленты", "type": "recommendation", "applied": True, "confidence": 0.75, "source": "RL_Agent"},
        ]
        
        mock_edges = [
            {"source": "param_1", "target": "defect_1", "type": "CAUSES", "confidence": 0.85},
            {"source": "param_2", "target": "defect_1", "type": "CAUSES", "confidence": 0.72},
            {"source": "rec_1", "target": "defect_1", "type": "ADDRESSES", "confidence": 0.88},
            {"source": "rec_2", "target": "defect_1", "type": "ADDRESSES", "confidence": 0.75},
        ]
        
        return {
            "nodes": mock_nodes,
            "edges": mock_edges,
            "defect_type": defect_type,
            "timestamp": datetime.now().isoformat(),
            "is_mock": True
        }
    
    def get_rl_feedback_history(
        self,
        defect_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get RL feedback history from human decisions for continuous learning.
        
        Args:
            defect_type: Filter by defect type (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of feedback records for RL training
        """
        feedback_records = []
        
        if self.cache_enabled:
            try:
                # Get all feedback keys
                pattern = f"rl_feedback:{defect_type or '*'}:*"
                keys = self.redis_client.keys(pattern)
                
                for key in keys[:limit]:
                    data = self.redis_client.get(key)
                    if data:
                        record = json.loads(data)
                        feedback_records.append(record)
                        
            except Exception as e:
                logger.warning(f"⚠️ Error getting RL feedback history: {e}")
        
        # Sort by timestamp descending
        feedback_records.sort(
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        
        return feedback_records
    
    def populate_with_synthetic_data(self) -> Dict[str, Any]:
        """
        Populate Knowledge Graph with synthetic production data.
        Creates realistic causal relationships between parameters and defects.
        
        Returns:
            Summary of populated data
        """
        result = {
            "equipments_created": 0,
            "parameters_created": 0,
            "defects_created": 0,
            "causal_relationships_created": 0,
            "recommendations_created": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Define equipment nodes
        equipments = [
            {"equipment_id": "furnace_A", "type": "Furnace", "zone": "melting"},
            {"equipment_id": "furnace_B", "type": "Furnace", "zone": "refining"},
            {"equipment_id": "forming_A", "type": "Forming", "zone": "shaping"},
            {"equipment_id": "annealing_A", "type": "Annealing", "zone": "cooling"},
            {"equipment_id": "mik_1", "type": "Quality_Control", "zone": "inspection"}
        ]
        
        # Define parameters and their normal ranges
        parameters = [
            {"name": "furnace_temperature", "category": "thermal", "min": 1400, "max": 1600, "unit": "°C"},
            {"name": "furnace_pressure", "category": "pressure", "min": 10, "max": 20, "unit": "kPa"},
            {"name": "belt_speed", "category": "mechanical", "min": 120, "max": 180, "unit": "m/min"},
            {"name": "mold_temperature", "category": "thermal", "min": 250, "max": 400, "unit": "°C"},
            {"name": "forming_pressure", "category": "pressure", "min": 30, "max": 70, "unit": "MPa"},
            {"name": "annealing_temperature", "category": "thermal", "min": 500, "max": 700, "unit": "°C"},
            {"name": "cooling_rate", "category": "thermal", "min": 2, "max": 5, "unit": "°C/min"},
            {"name": "melt_level", "category": "process", "min": 0, "max": 5000, "unit": "mm"},
            {"name": "o2_level", "category": "gas", "min": 0, "max": 25, "unit": "%"},
            {"name": "humidity", "category": "environment", "min": 30, "max": 70, "unit": "%"}
        ]
        
        # Define defects with severity and descriptions
        defects = [
            {"type": "crack", "severity": "HIGH", "description": "Surface or internal cracks in glass"},
            {"type": "bubble", "severity": "MEDIUM", "description": "Air bubbles trapped in glass"},
            {"type": "chip", "severity": "LOW", "description": "Edge chips or small missing pieces"},
            {"type": "cloudiness", "severity": "MEDIUM", "description": "Hazy or cloudy appearance"},
            {"type": "deformation", "severity": "HIGH", "description": "Shape distortion or warping"},
            {"type": "stain", "severity": "LOW", "description": "Surface discoloration or marks"}
        ]
        
        # Define causal relationships (parameter -> defect)
        causal_relationships = [
            # Temperature-related causes
            {"param": "furnace_temperature", "equip": "furnace_A", "defect": "crack", "confidence": 0.85,
             "description": "High temperature fluctuations cause thermal stress leading to cracks"},
            {"param": "furnace_temperature", "equip": "furnace_A", "defect": "bubble", "confidence": 0.72,
             "description": "Low temperature prevents proper degassing"},
            {"param": "furnace_temperature", "equip": "furnace_A", "defect": "cloudiness", "confidence": 0.65,
             "description": "Temperature variations affect glass homogeneity"},
            
            # Speed-related causes  
            {"param": "belt_speed", "equip": "forming_A", "defect": "deformation", "confidence": 0.88,
             "description": "High speed causes uneven cooling and deformation"},
            {"param": "belt_speed", "equip": "forming_A", "defect": "bubble", "confidence": 0.58,
             "description": "High speed traps air bubbles in forming"},
            {"param": "belt_speed", "equip": "forming_A", "defect": "chip", "confidence": 0.62,
             "description": "High speed causes edge damage"},
            
            # Pressure-related causes
            {"param": "forming_pressure", "equip": "forming_A", "defect": "crack", "confidence": 0.78,
             "description": "Excessive pressure causes stress fractures"},
            {"param": "forming_pressure", "equip": "forming_A", "defect": "deformation", "confidence": 0.82,
             "description": "Uneven pressure causes shape distortion"},
            
            # Cooling-related causes
            {"param": "cooling_rate", "equip": "annealing_A", "defect": "crack", "confidence": 0.92,
             "description": "Rapid cooling creates thermal shock and cracks"},
            {"param": "cooling_rate", "equip": "annealing_A", "defect": "cloudiness", "confidence": 0.55,
             "description": "Improper cooling affects optical clarity"},
            {"param": "annealing_temperature", "equip": "annealing_A", "defect": "crack", "confidence": 0.75,
             "description": "Wrong annealing temp causes residual stress"},
            
            # Mold-related causes
            {"param": "mold_temperature", "equip": "forming_A", "defect": "stain", "confidence": 0.68,
             "description": "Hot mold causes surface marks"},
            {"param": "mold_temperature", "equip": "forming_A", "defect": "deformation", "confidence": 0.72,
             "description": "Cold mold causes premature cooling"},
        ]
        
        # Define recommendations for each defect
        recommendations = [
            {"defect": "crack", "action": "Reduce cooling rate", "priority": "HIGH", 
             "expected_impact": "70% reduction in cracks", "target_param": "cooling_rate", "target_value": 2.5},
            {"defect": "crack", "action": "Stabilize furnace temperature", "priority": "HIGH",
             "expected_impact": "60% reduction in thermal stress", "target_param": "furnace_temperature", "target_value": 1520},
            {"defect": "bubble", "action": "Increase furnace temperature", "priority": "MEDIUM",
             "expected_impact": "50% reduction in bubbles", "target_param": "furnace_temperature", "target_value": 1560},
            {"defect": "bubble", "action": "Reduce belt speed", "priority": "MEDIUM",
             "expected_impact": "40% reduction in trapped air", "target_param": "belt_speed", "target_value": 140},
            {"defect": "deformation", "action": "Optimize belt speed", "priority": "HIGH",
             "expected_impact": "65% reduction in deformation", "target_param": "belt_speed", "target_value": 145},
            {"defect": "deformation", "action": "Adjust forming pressure", "priority": "HIGH",
             "expected_impact": "55% shape improvement", "target_param": "forming_pressure", "target_value": 50},
            {"defect": "cloudiness", "action": "Maintain stable cooling", "priority": "MEDIUM",
             "expected_impact": "45% clarity improvement", "target_param": "cooling_rate", "target_value": 3.0},
            {"defect": "chip", "action": "Reduce belt speed", "priority": "LOW",
             "expected_impact": "50% reduction in edge damage", "target_param": "belt_speed", "target_value": 135},
            {"defect": "stain", "action": "Optimize mold temperature", "priority": "LOW",
             "expected_impact": "60% surface quality improvement", "target_param": "mold_temperature", "target_value": 320},
        ]
        
        if not self.driver:
            logger.warning("⚠️ Neo4j not available, skipping KG population")
            return result
        
        try:
            with self.driver.session() as session:
                # Create equipment nodes
                for equip in equipments:
                    session.run("""
                        MERGE (e:Equipment {equipment_id: $equipment_id})
                        SET e.type = $type, e.zone = $zone
                    """, equip)
                    result["equipments_created"] += 1
                
                # Create parameter nodes
                for param in parameters:
                    session.run("""
                        MERGE (p:Parameter {name: $name})
                        SET p.category = $category, p.min_value = $min, 
                            p.max_value = $max, p.unit = $unit
                    """, param)
                    result["parameters_created"] += 1
                
                # Create defect nodes
                for defect in defects:
                    session.run("""
                        MERGE (d:Defect {type: $type})
                        SET d.severity = $severity, d.description = $description
                    """, defect)
                    result["defects_created"] += 1
                
                # Create causal relationships
                for rel in causal_relationships:
                    session.run("""
                        MATCH (p:Parameter {name: $param})
                        MATCH (e:Equipment {equipment_id: $equip})
                        MATCH (d:Defect {type: $defect})
                        MERGE (c:Cause {parameter: $param, equipment: $equip, defect: $defect})
                        SET c.confidence = $confidence, c.description = $description,
                            c.timestamp = $timestamp
                        MERGE (c)-[:CAUSES]->(d)
                        MERGE (c)-[:RELATED_TO]->(p)
                        MERGE (c)-[:FROM_EQUIPMENT]->(e)
                    """, {
                        **rel,
                        "timestamp": datetime.now().isoformat()
                    })
                    result["causal_relationships_created"] += 1
                
                # Create recommendations
                for rec in recommendations:
                    session.run("""
                        MATCH (d:Defect {type: $defect})
                        MERGE (r:Recommendation {
                            action: $action,
                            defect_type: $defect
                        })
                        SET r.priority = $priority, r.expected_impact = $expected_impact,
                            r.target_parameter = $target_param, r.target_value = $target_value,
                            r.source = 'KnowledgeGraph', r.timestamp = $timestamp
                        MERGE (r)-[:ADDRESSES]->(d)
                    """, {
                        **rec,
                        "timestamp": datetime.now().isoformat()
                    })
                    result["recommendations_created"] += 1
                
                logger.info(f"🧠 KG populated: {result}")
                
        except Exception as e:
            logger.error(f"❌ Error populating KG: {e}")
        
        return result

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

