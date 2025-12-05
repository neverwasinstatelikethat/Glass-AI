"""
Knowledge Base Initializer for Causal Knowledge Graph
Populates Neo4j with domain knowledge about glass production
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import os

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
            
            self.initialized = True
            logger.info("✅ Knowledge Graph populated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error populating knowledge graph: {e}")
            raise
    
    def _add_parameter_defect_causations(self):
        """Add parameter → defect causation relationships"""
        logger.info("Adding parameter-defect causations...")
        
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
            # mold_temperature < 280°C → stress_fracture
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
            except Exception as e:
                logger.warning(f"⚠️ Could not add causation {causation}: {e}")
        
        logger.info(f"✅ Added {len(causations)} parameter-defect causations")
    
    def _add_equipment_parameter_relationships(self):
        """Add equipment → parameter control relationships"""
        logger.info("Adding equipment-parameter relationships...")
        
        relationships = [
            {
                'equipment': 'furnace_burner',
                'controls': 'furnace_temperature',
                'affects': 'melt_viscosity',
                'response_time_min': 15,
                'response_time_max': 30
            },
            {
                'equipment': 'belt_motor',
                'controls': 'belt_speed',
                'affects': 'forming_pressure',
                'response_time_min': 5,
                'response_time_max': 10
            },
            {
                'equipment': 'mold_heater',
                'controls': 'mold_temperature',
                'affects': 'product_stress',
                'response_time_min': 10,
                'response_time_max': 20
            },
            {
                'equipment': 'cooling_fan',
                'controls': 'cooling_rate',
                'affects': 'annealing_quality',
                'response_time_min': 2,
                'response_time_max': 5
            }
        ]
        
        for rel in relationships:
            try:
                # Add as causal relationship
                self.kg.add_cause(
                    cause=rel['equipment'],
                    effect=rel['controls'],
                    confidence=0.95,
                    mechanism=f"Equipment controls parameter with {rel['response_time_min']}-{rel['response_time_max']}min response time"
                )
                
                # Add secondary effect
                self.kg.add_cause(
                    cause=rel['controls'],
                    effect=rel['affects'],
                    confidence=0.85,
                    mechanism="Parameter affects downstream process"
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not add relationship {rel}: {e}")
        
        logger.info(f"✅ Added {len(relationships)} equipment-parameter relationships")
    
    def _add_intervention_strategies(self):
        """Add defect → intervention recommendation strategies"""
        logger.info("Adding intervention strategies...")
        
        interventions = [
            {
                'defect': 'crack',
                'condition': 'probability > 0.7',
                'action': 'Reduce furnace temp by 20-30°C',
                'expected_outcome': 'Defect rate -40%',
                'implementation_time_min': 20,
                'implementation_time_max': 40,
                'confidence': 0.85
            },
            {
                'defect': 'bubble',
                'condition': 'count > 5/min',
                'action': 'Decrease forming pressure by 10 MPa',
                'expected_outcome': 'Defect rate -60%',
                'implementation_time_min': 1,
                'implementation_time_max': 2,
                'confidence': 0.88
            },
            {
                'defect': 'deformation',
                'condition': 'score > 0.3',
                'action': 'Reduce belt speed by 15%',
                'expected_outcome': 'Defect rate -50%',
                'implementation_time_min': 0,
                'implementation_time_max': 1,
                'confidence': 0.82
            },
            {
                'defect': 'cloudiness',
                'condition': 'probability > 0.5',
                'action': 'Increase furnace temp by 30-50°C',
                'expected_outcome': 'Defect rate -35%',
                'implementation_time_min': 20,
                'implementation_time_max': 40,
                'confidence': 0.78
            }
        ]
        
        for intervention in interventions:
            try:
                # Add intervention as relationship
                intervention_id = f"intervention_{intervention['defect']}"
                
                self.kg.add_cause(
                    cause=intervention['defect'],
                    effect=intervention_id,
                    confidence=intervention['confidence'],
                    mechanism=f"Apply: {intervention['action']}. Expected: {intervention['expected_outcome']}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not add intervention {intervention}: {e}")
        
        logger.info(f"✅ Added {len(interventions)} intervention strategies")
    
    def get_initialization_status(self) -> Dict:
        """Get status of knowledge base initialization"""
        return {
            'initialized': self.initialized,
            'timestamp': datetime.utcnow().isoformat()
        }


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
        return True
    except Exception as e:
        logger.error(f"❌ Knowledge base initialization failed: {e}")
        return False
