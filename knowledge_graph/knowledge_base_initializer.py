"""
Knowledge Base Initializer for Causal Knowledge Graph
Populates Neo4j with domain knowledge about glass production
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class KnowledgeBaseInitializer:
    """Initialize and populate the Knowledge Graph with domain knowledge"""
    
    def __init__(self, knowledge_graph):
        """
        Initialize knowledge base populator
        
        Args:
            knowledge_graph: CausalKnowledgeGraph instance
        """
        self.kg = knowledge_graph
        self.initialized = False
        
    def populate_all_knowledge(self):
        """Populate all domain knowledge into the graph"""
        try:
            logger.info("🔄 Populating Knowledge Graph with domain knowledge...")
            
            # Add parameter-defect causations
            self._add_parameter_defect_causations()
            
            # Add equipment-parameter relationships
            self._add_equipment_parameter_relationships()
            
            # Add intervention strategies
            self._add_intervention_strategies()
            
            # Verify data population
            self._verify_data_population()
            
            self.initialized = True
            logger.info("✅ Knowledge Graph populated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error populating knowledge graph: {e}")
            raise
    
    def _add_parameter_defect_causations(self):
        """Add parameter → defect causation relationships with complete node properties"""
        logger.info("Adding parameter-defect causations...")
        
        # First, ensure all parameter nodes have complete properties
        parameters = [
            {'name': 'furnace_temperature', 'category': 'thermal', 'min_value': 1200, 'max_value': 1700, 'unit': '°C'},
            {'name': 'furnace_pressure', 'category': 'pressure', 'min_value': 0, 'max_value': 50, 'unit': 'kPa'},
            {'name': 'melt_level', 'category': 'process', 'min_value': 0, 'max_value': 5000, 'unit': 'mm'},
            {'name': 'belt_speed', 'category': 'mechanical', 'min_value': 0, 'max_value': 200, 'unit': 'm/min'},
            {'name': 'mold_temperature', 'category': 'thermal', 'min_value': 20, 'max_value': 600, 'unit': '°C'},
            {'name': 'forming_pressure', 'category': 'pressure', 'min_value': 0, 'max_value': 100, 'unit': 'MPa'},
            {'name': 'cooling_rate', 'category': 'thermal', 'min_value': 0, 'max_value': 10, 'unit': '°C/min'},
            {'name': 'annealing_temperature', 'category': 'thermal', 'min_value': 20, 'max_value': 1200, 'unit': '°C'},
            {'name': 'oxygen_content', 'category': 'chemical', 'min_value': 0, 'max_value': 25, 'unit': '%'}
        ]
        
        # Add parameter nodes with complete properties
        for param in parameters:
            try:
                with self.kg.driver.session() as session:
                    session.run("""
                        MERGE (p:Parameter {name: $name})
                        SET p.category = $category,
                            p.min_value = $min_value,
                            p.max_value = $max_value,
                            p.unit = $unit
                    """, param)
            except Exception as e:
                logger.warning(f"⚠️ Could not add parameter {param['name']}: {e}")
        
        # Add defect nodes with complete properties
        defects = [
            {'type': 'crack', 'severity': 'HIGH', 'description': 'Micro-fractures in glass surface'},
            {'type': 'bubble', 'severity': 'MEDIUM', 'description': 'Gas bubbles trapped in glass'},
            {'type': 'chip', 'severity': 'LOW', 'description': 'Small pieces broken from edge'},
            {'type': 'cloudiness', 'severity': 'MEDIUM', 'description': 'Reduced transparency due to crystallization'},
            {'type': 'deformation', 'severity': 'HIGH', 'description': 'Distorted shape during forming'},
            {'type': 'stain', 'severity': 'LOW', 'description': 'Surface contamination or discoloration'},
            {'type': 'scratch', 'severity': 'MEDIUM', 'description': 'Surface abrasions during handling'}
        ]
        
        for defect in defects:
            try:
                with self.kg.driver.session() as session:
                    session.run("""
                        MERGE (d:Defect {type: $type})
                        SET d.severity = $severity,
                            d.description = $description
                    """, defect)
            except Exception as e:
                logger.warning(f"⚠️ Could not add defect {defect['type']}: {e}")
        
        # Add causation relationships
        causations = [
            # furnace_temperature > 1600°C → crack
            {
                'parameter': 'furnace_temperature',
                'condition': '>',
                'threshold': 1600,
                'defect': 'crack',
                'confidence': 0.85,
                'mechanism': 'Excessive thermal stress during cooling'
            },
            # furnace_temperature < 1450°C → cloudiness
            {
                'parameter': 'furnace_temperature',
                'condition': '<',
                'threshold': 1450,
                'defect': 'cloudiness',
                'confidence': 0.78,
                'mechanism': 'Incomplete melting, crystallization'
            },
            # belt_speed > 180 m/min → deformation
            {
                'parameter': 'belt_speed',
                'condition': '>',
                'threshold': 180,
                'defect': 'deformation',
                'confidence': 0.82,
                'mechanism': 'Insufficient forming time'
            },
            # mold_temperature < 280°C → stress
            {
                'parameter': 'mold_temperature',
                'condition': '<',
                'threshold': 280,
                'defect': 'stress',
                'confidence': 0.75,
                'mechanism': 'Rapid cooling creates internal stress'
            },
            # pressure_variation > 15 MPa → bubble
            {
                'parameter': 'forming_pressure',
                'condition': 'var',
                'threshold': 15,
                'defect': 'bubble',
                'confidence': 0.88,
                'mechanism': 'Gas entrapment during forming'
            },
            # cooling_rate > 7°C/min → crack
            {
                'parameter': 'cooling_rate',
                'condition': '>',
                'threshold': 7,
                'defect': 'crack',
                'confidence': 0.90,
                'mechanism': 'Thermal shock exceeds material limits'
            },
            # oxygen_content > 5% → bubble
            {
                'parameter': 'oxygen_content',
                'condition': '>',
                'threshold': 5,
                'defect': 'bubble',
                'confidence': 0.72,
                'mechanism': 'Excess oxygen leads to gas formation'
            },
            # mold_temperature > 400°C → deformation
            {
                'parameter': 'mold_temperature',
                'condition': '>',
                'threshold': 400,
                'defect': 'deformation',
                'confidence': 0.76,
                'mechanism': 'Overheated mold causes softening'
            },
            # furnace_pressure < 5 kPa → cloudiness
            {
                'parameter': 'furnace_pressure',
                'condition': '<',
                'threshold': 5,
                'defect': 'cloudiness',
                'confidence': 0.68,
                'mechanism': 'Low pressure affects melting uniformity'
            }
        ]
        
        for causation in causations:
            try:
                # Add to knowledge graph using existing methods
                self.kg.add_cause(
                    cause=f"{causation['parameter']}_{causation['condition']}_{causation['threshold']}",
                    effect=causation['defect'],
                    confidence=causation['confidence'],
                    mechanism=causation['mechanism']
                )
                
                # Also add the parameter node
                self.kg.add_measurement(
                    parameter_name=causation['parameter'],
                    value=float(causation['threshold']),
                    timestamp=datetime.now(),
                    equipment_id='simulation'
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not add causation {causation}: {e}")
        
        logger.info(f"✅ Added {len(causations)} parameter-defect causations")
    
    def _add_equipment_parameter_relationships(self):
        """Add equipment → parameter control relationships with complete node properties"""
        logger.info("Adding equipment-parameter relationships...")
        
        # Add equipment nodes with complete properties
        equipments = [
            {'equipment_id': 'furnace_burner_A', 'type': 'Burner', 'zone': 'Melting'},
            {'equipment_id': 'furnace_burner_B', 'type': 'Burner', 'zone': 'Melting'},
            {'equipment_id': 'belt_motor_1', 'type': 'Motor', 'zone': 'Forming'},
            {'equipment_id': 'belt_motor_2', 'type': 'Motor', 'zone': 'Forming'},
            {'equipment_id': 'mold_heater_front', 'type': 'Heater', 'zone': 'Forming'},
            {'equipment_id': 'mold_heater_back', 'type': 'Heater', 'zone': 'Forming'},
            {'equipment_id': 'cooling_fan_1', 'type': 'Fan', 'zone': 'Annealing'},
            {'equipment_id': 'cooling_fan_2', 'type': 'Fan', 'zone': 'Annealing'},
            {'equipment_id': 'annealing_oven', 'type': 'Oven', 'zone': 'Annealing'}
        ]
        
        for equip in equipments:
            try:
                self.kg.add_equipment(equip['equipment_id'], equip['type'], equip['zone'])
            except Exception as e:
                logger.warning(f"⚠️ Could not add equipment {equip['equipment_id']}: {e}")
        
        # Define relationships with proper properties
        relationships = [
            {
                'equipment_id': 'furnace_burner_A',
                'controls': 'furnace_temperature',
                'affects': 'melt_viscosity',
                'response_time_min': 15,
                'response_time_max': 30
            },
            {
                'equipment_id': 'furnace_burner_B',
                'controls': 'furnace_temperature',
                'affects': 'melt_uniformity',
                'response_time_min': 20,
                'response_time_max': 35
            },
            {
                'equipment_id': 'belt_motor_1',
                'controls': 'belt_speed',
                'affects': 'forming_pressure',
                'response_time_min': 5,
                'response_time_max': 10
            },
            {
                'equipment_id': 'belt_motor_2',
                'controls': 'belt_speed',
                'affects': 'thickness_consistency',
                'response_time_min': 5,
                'response_time_max': 10
            },
            {
                'equipment_id': 'mold_heater_front',
                'controls': 'mold_temperature',
                'affects': 'product_stress',
                'response_time_min': 10,
                'response_time_max': 20
            },
            {
                'equipment_id': 'mold_heater_back',
                'controls': 'mold_temperature',
                'affects': 'surface_quality',
                'response_time_min': 12,
                'response_time_max': 22
            },
            {
                'equipment_id': 'cooling_fan_1',
                'controls': 'cooling_rate',
                'affects': 'annealing_quality',
                'response_time_min': 2,
                'response_time_max': 5
            },
            {
                'equipment_id': 'cooling_fan_2',
                'controls': 'cooling_rate',
                'affects': 'residual_stress',
                'response_time_min': 2,
                'response_time_max': 5
            },
            {
                'equipment_id': 'annealing_oven',
                'controls': 'annealing_temperature',
                'affects': 'internal_stress',
                'response_time_min': 30,
                'response_time_max': 60
            }
        ]
        
        for rel in relationships:
            try:
                # Add parameter node if not exists
                self.kg.add_measurement(
                    parameter_name=rel['controls'],
                    value=0.0,  # placeholder value
                    timestamp=datetime.now(),
                    equipment_id=rel['equipment_id']
                )
                
                # Add causal relationship between equipment and parameter
                self.kg.add_causal_relationship(
                    parameter_name=rel['controls'],
                    equipment_id=rel['equipment_id'],
                    defect_type=rel['affects'],
                    confidence=0.95,
                    description=f"Equipment controls parameter with {rel['response_time_min']}-{rel['response_time_max']}min response time"
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not add relationship {rel}: {e}")
        
        logger.info(f"✅ Added {len(relationships)} equipment-parameter relationships")
    
    def _add_intervention_strategies(self):
        """Add defect → intervention recommendation strategies with complete node properties"""
        logger.info("Adding intervention strategies...")
        
        # Add more comprehensive intervention strategies
        interventions = [
            {
                'defect': 'crack',
                'condition': 'probability > 0.7',
                'action': 'Reduce furnace temp by 20-30°C',
                'target_parameter': 'furnace_temperature',
                'target_value': -25,
                'unit': '°C',
                'expected_outcome': 'Defect rate -40%',
                'implementation_time_min': 20,
                'implementation_time_max': 40,
                'confidence': 0.85,
                'priority': 'HIGH',
                'urgency': 'IMMEDIATE'
            },
            {
                'defect': 'crack',
                'condition': 'cooling_rate > 5',
                'action': 'Decrease cooling rate to <5°C/min',
                'target_parameter': 'cooling_rate',
                'target_value': 4,
                'unit': '°C/min',
                'expected_outcome': 'Defect rate -35%',
                'implementation_time_min': 5,
                'implementation_time_max': 15,
                'confidence': 0.82,
                'priority': 'HIGH',
                'urgency': 'HIGH'
            },
            {
                'defect': 'bubble',
                'condition': 'count > 5/min',
                'action': 'Decrease forming pressure by 10 MPa',
                'target_parameter': 'forming_pressure',
                'target_value': -10,
                'unit': 'MPa',
                'expected_outcome': 'Defect rate -60%',
                'implementation_time_min': 1,
                'implementation_time_max': 2,
                'confidence': 0.88,
                'priority': 'HIGH',
                'urgency': 'IMMEDIATE'
            },
            {
                'defect': 'bubble',
                'condition': 'furnace_temp < 1450',
                'action': 'Increase furnace temp to 1500°C',
                'target_parameter': 'furnace_temperature',
                'target_value': 1500,
                'unit': '°C',
                'expected_outcome': 'Defect rate -30%',
                'implementation_time_min': 20,
                'implementation_time_max': 40,
                'confidence': 0.75,
                'priority': 'MEDIUM',
                'urgency': 'MEDIUM'
            },
            {
                'defect': 'deformation',
                'condition': 'score > 0.3',
                'action': 'Reduce belt speed by 15%',
                'target_parameter': 'belt_speed',
                'target_value': -0.15,  # Percentage reduction
                'unit': '%',
                'expected_outcome': 'Defect rate -50%',
                'implementation_time_min': 0,
                'implementation_time_max': 1,
                'confidence': 0.82,
                'priority': 'HIGH',
                'urgency': 'IMMEDIATE'
            },
            {
                'defect': 'deformation',
                'condition': 'mold_temp < 300',
                'action': 'Increase mold temperature to 320°C',
                'target_parameter': 'mold_temperature',
                'target_value': 320,
                'unit': '°C',
                'expected_outcome': 'Defect rate -25%',
                'implementation_time_min': 10,
                'implementation_time_max': 20,
                'confidence': 0.78,
                'priority': 'MEDIUM',
                'urgency': 'HIGH'
            },
            {
                'defect': 'cloudiness',
                'condition': 'probability > 0.5',
                'action': 'Increase furnace temp by 30-50°C',
                'target_parameter': 'furnace_temperature',
                'target_value': 40,
                'unit': '°C',
                'expected_outcome': 'Defect rate -35%',
                'implementation_time_min': 20,
                'implementation_time_max': 40,
                'confidence': 0.78,
                'priority': 'MEDIUM',
                'urgency': 'MEDIUM'
            },
            {
                'defect': 'chip',
                'condition': 'belt_speed > 170',
                'action': 'Reduce belt speed to 150 m/min',
                'target_parameter': 'belt_speed',
                'target_value': 150,
                'unit': 'm/min',
                'expected_outcome': 'Defect rate -40%',
                'implementation_time_min': 0,
                'implementation_time_max': 1,
                'confidence': 0.80,
                'priority': 'MEDIUM',
                'urgency': 'HIGH'
            },
            {
                'defect': 'stain',
                'condition': 'mold_temp > 350',
                'action': 'Adjust mold temperature to 320-340°C',
                'target_parameter': 'mold_temperature',
                'target_value': 330,
                'unit': '°C',
                'expected_outcome': 'Defect rate -30%',
                'implementation_time_min': 15,
                'implementation_time_max': 30,
                'confidence': 0.75,
                'priority': 'LOW',
                'urgency': 'LOW'
            },
            {
                'defect': 'scratch',
                'condition': 'handling_speed > 2 m/s',
                'action': 'Reduce handling speed to < 1.5 m/s',
                'target_parameter': 'handling_speed',
                'target_value': 1.5,
                'unit': 'm/s',
                'expected_outcome': 'Scratch rate -50%',
                'implementation_time_min': 0,
                'implementation_time_max': 1,
                'confidence': 0.85,
                'priority': 'MEDIUM',
                'urgency': 'HIGH'
            },
            {
                'defect': 'bubble',
                'condition': 'oxygen_content > 5%',
                'action': 'Reduce oxygen content to < 3%',
                'target_parameter': 'oxygen_content',
                'target_value': 3,
                'unit': '%',
                'expected_outcome': 'Bubble rate -40%',
                'implementation_time_min': 30,
                'implementation_time_max': 60,
                'confidence': 0.78,
                'priority': 'MEDIUM',
                'urgency': 'MEDIUM'
            }
        ]
        
        for intervention in interventions:
            try:
                # Generate unique recommendation ID
                recommendation_id = f"rec_{uuid.uuid4().hex[:8]}"
                
                # Add recommendation with complete properties
                with self.kg.driver.session() as session:
                    session.run("""
                        MERGE (defect:Defect {type: $defect_type})
                        CREATE (rec:Recommendation {
                            recommendation_id: $recommendation_id,
                            action: $action,
                            target_parameter: $target_parameter,
                            target_value: $target_value,
                            unit: $unit,
                            confidence: $confidence,
                            urgency: $urgency,
                            priority: $priority,
                            expected_impact: $expected_outcome,
                            source: 'KnowledgeBaseInitializer',
                            timestamp: $timestamp
                        })-[:ADDRESSES]->(defect)
                    """, {
                        "defect_type": intervention['defect'],
                        "recommendation_id": recommendation_id,
                        "action": intervention['action'],
                        "target_parameter": intervention.get('target_parameter'),
                        "target_value": intervention.get('target_value'),
                        "unit": intervention.get('unit'),
                        "confidence": intervention['confidence'],
                        "urgency": intervention['urgency'],
                        "priority": intervention['priority'],
                        "expected_outcome": intervention['expected_outcome'],
                        "timestamp": datetime.now().isoformat()
                    })
            except Exception as e:
                logger.warning(f"⚠️ Could not add intervention {intervention}: {e}")
        
        logger.info(f"✅ Added {len(interventions)} intervention strategies")
    
    def get_initialization_status(self) -> Dict:
        """Get status of knowledge base initialization"""
        return {
            'initialized': self.initialized,
            'timestamp': datetime.utcnow().isoformat(),
            'verification_results': getattr(self, '_last_verification', {})
        }
    
    def _verify_data_population(self):
        """Verify that data population was successful"""
        if not self.kg.driver:
            logger.warning("⚠️ Neo4j not available, skipping verification")
            return
        
        try:
            verification_results = {}
            
            with self.kg.driver.session() as session:
                # Count nodes by type
                node_counts = session.run("""
                    MATCH (n)
                    RETURN labels(n)[0] as label, count(n) as count
                """)
                
                for record in node_counts:
                    label = record["label"]
                    count = record["count"]
                    verification_results[f"{label}_nodes"] = count
                
                # Count relationships by type
                rel_counts = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(r) as count
                """)
                
                for record in rel_counts:
                    rel_type = record["rel_type"]
                    count = record["count"]
                    verification_results[f"{rel_type}_relationships"] = count
                
                # Check for key properties
                property_checks = [
                    ("Parameter nodes with category", "MATCH (p:Parameter) WHERE exists(p.category) RETURN count(p) as count"),
                    ("Defect nodes with severity", "MATCH (d:Defect) WHERE exists(d.severity) RETURN count(d) as count"),
                    ("Equipment nodes with zone", "MATCH (e:Equipment) WHERE exists(e.zone) RETURN count(e) as count"),
                    ("Recommendation nodes with urgency", "MATCH (r:Recommendation) WHERE exists(r.urgency) RETURN count(r) as count")
                ]
                
                for check_name, query in property_checks:
                    result = session.run(query)
                    count = result.single()["count"]
                    verification_results[check_name] = count
            
            self._last_verification = verification_results
            logger.info(f"✅ Data population verification completed: {verification_results}")
            
        except Exception as e:
            logger.error(f"❌ Error verifying data population: {e}")
            self._last_verification = {"error": str(e)}


def initialize_knowledge_base(knowledge_graph) -> bool:
    """
    Initialize knowledge base with domain knowledge
    
    Args:
        knowledge_graph: CausalKnowledgeGraph instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        initializer = KnowledgeBaseInitializer(knowledge_graph)
        initializer.populate_all_knowledge()
        
        # Log verification results
        status = initializer.get_initialization_status()
        if 'verification_results' in status:
            logger.info(f"📊 Knowledge base verification results: {status['verification_results']}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Knowledge base initialization failed: {e}")
        return False
