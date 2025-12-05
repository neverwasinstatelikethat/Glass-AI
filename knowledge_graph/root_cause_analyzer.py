"""
Root Cause Analysis Engine
Analyzes defects and identifies root causes using the Knowledge Graph
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RootCause:
    """Representation of a root cause"""
    root_cause_id: str
    parameter_name: str
    deviation_magnitude: float  # In standard deviations
    confidence: float  # 0-1
    evidence: List[Dict]
    recommended_action: str
    timestamp: datetime


class RootCauseAnalyzer:
    """
    Analyzes defects and identifies root causes using causal knowledge
    """
    
    def __init__(self, knowledge_graph, influxdb_client=None):
        """
        Initialize Root Cause Analyzer
        
        Args:
            knowledge_graph: CausalKnowledgeGraph instance
            influxdb_client: Optional InfluxDB client for historical data
        """
        self.kg = knowledge_graph
        self.influxdb = influxdb_client
        
        # Parameter normal ranges (mean, std) for deviation calculation
        self.parameter_stats = {
            'furnace_temperature': (1500, 50),
            'furnace_pressure': (15, 5),
            'melt_level': (2500, 300),
            'belt_speed': (150, 20),
            'mold_temperature': (320, 40),
            'forming_pressure': (50, 15),
            'cooling_rate': (3.5, 1.5),
            'annealing_temp': (580, 40)
        }
        
        logger.info("✅ Root Cause Analyzer initialized")
    
    def analyze_defect(self, defect_type: str, 
                      current_parameters: Dict[str, float],
                      timestamp: Optional[datetime] = None,
                      min_confidence: float = 0.5) -> List[RootCause]:
        """
        Analyze a defect and identify possible root causes
        
        Workflow:
        1. Query knowledge graph for possible causes
        2. Retrieve recent sensor data
        3. Match patterns to known causes
        4. Rank by confidence
        5. Return top 3 root causes with evidence
        
        Args:
            defect_type: Type of defect detected
            current_parameters: Current parameter values
            timestamp: When defect was detected
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of RootCause objects, ranked by confidence
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        logger.info(f"🔍 Analyzing root cause for defect: {defect_type}")
        
        # Step 1: Query knowledge graph for possible causes
        possible_causes = self._query_possible_causes(defect_type)
        
        # Step 2: Analyze current parameters for deviations
        parameter_deviations = self._calculate_parameter_deviations(current_parameters)
        
        # Step 3: Match deviations to known causes
        matched_causes = self._match_causes_to_deviations(
            possible_causes, 
            parameter_deviations,
            current_parameters
        )
        
        # Step 4: Rank by confidence
        ranked_causes = self._rank_root_causes(matched_causes, min_confidence)
        
        # Step 5: Add evidence and recommendations
        root_causes = []
        for i, cause_data in enumerate(ranked_causes[:3]):  # Top 3
            root_cause = self._create_root_cause_object(
                cause_data,
                defect_type,
                parameter_deviations,
                current_parameters,
                timestamp,
                rank=i+1
            )
            root_causes.append(root_cause)
        
        logger.info(f"✅ Found {len(root_causes)} root causes for {defect_type}")
        
        return root_causes
    
    def _query_possible_causes(self, defect_type: str) -> List[Dict]:
        """Query knowledge graph for parameters that cause this defect"""
        possible_causes = []
        
        try:
            # Query knowledge graph for causes
            causes_data = self.kg.query_causes(effect=defect_type)
            
            if causes_data:
                for cause in causes_data:
                    possible_causes.append({
                        'parameter': cause.get('cause', ''),
                        'confidence': cause.get('confidence', 0.5),
                        'mechanism': cause.get('mechanism', '')
                    })
        except Exception as e:
            logger.warning(f"⚠️ Error querying knowledge graph: {e}")
        
        # Fallback to hardcoded rules if knowledge graph query fails
        if not possible_causes:
            possible_causes = self._get_fallback_causes(defect_type)
        
        return possible_causes
    
    def _get_fallback_causes(self, defect_type: str) -> List[Dict]:
        """Get fallback causes if knowledge graph unavailable"""
        fallback_rules = {
            'crack': [
                {'parameter': 'furnace_temperature', 'confidence': 0.85, 
                 'mechanism': 'High temperature causes thermal stress'},
                {'parameter': 'cooling_rate', 'confidence': 0.90,
                 'mechanism': 'Rapid cooling causes thermal shock'}
            ],
            'bubble': [
                {'parameter': 'forming_pressure', 'confidence': 0.88,
                 'mechanism': 'Pressure variations trap gas'},
                {'parameter': 'furnace_temperature', 'confidence': 0.75,
                 'mechanism': 'High temperature increases gas formation'}
            ],
            'deformation': [
                {'parameter': 'belt_speed', 'confidence': 0.82,
                 'mechanism': 'High speed reduces forming time'},
                {'parameter': 'mold_temperature', 'confidence': 0.75,
                 'mechanism': 'Low temperature increases viscosity'}
            ],
            'cloudiness': [
                {'parameter': 'furnace_temperature', 'confidence': 0.78,
                 'mechanism': 'Low temperature causes incomplete melting'}
            ],
            'chip': [
                {'parameter': 'belt_speed', 'confidence': 0.70,
                 'mechanism': 'High speed causes mechanical stress'}
            ],
            'stain': [
                {'parameter': 'mold_temperature', 'confidence': 0.65,
                 'mechanism': 'Temperature affects surface quality'}
            ]
        }
        
        return fallback_rules.get(defect_type, [])
    
    def _calculate_parameter_deviations(self, current_parameters: Dict[str, float]) -> Dict[str, float]:
        """Calculate how far parameters deviate from normal (in std deviations)"""
        deviations = {}
        
        for param_name, value in current_parameters.items():
            if param_name in self.parameter_stats:
                mean, std = self.parameter_stats[param_name]
                deviation = (value - mean) / std
                deviations[param_name] = deviation
        
        return deviations
    
    def _match_causes_to_deviations(self, possible_causes: List[Dict],
                                   deviations: Dict[str, float],
                                   current_params: Dict[str, float]) -> List[Dict]:
        """Match possible causes to actual parameter deviations"""
        matched_causes = []
        
        for cause in possible_causes:
            param_name = cause['parameter']
            
            # Extract parameter name from complex cause strings
            for known_param in self.parameter_stats.keys():
                if known_param in param_name:
                    param_name = known_param
                    break
            
            if param_name not in deviations:
                continue
            
            deviation = deviations[param_name]
            base_confidence = cause['confidence']
            
            # Adjust confidence based on deviation magnitude
            if abs(deviation) > 2.0:  # More than 2 std deviations
                adjusted_confidence = min(1.0, base_confidence * 1.2)
            elif abs(deviation) > 1.0:
                adjusted_confidence = base_confidence
            else:
                adjusted_confidence = base_confidence * 0.7
            
            matched_causes.append({
                'parameter': param_name,
                'deviation': deviation,
                'confidence': adjusted_confidence,
                'mechanism': cause.get('mechanism', ''),
                'current_value': current_params.get(param_name, 0)
            })
        
        return matched_causes
    
    def _rank_root_causes(self, causes: List[Dict], min_confidence: float) -> List[Dict]:
        """Rank root causes by confidence and deviation magnitude"""
        # Filter by minimum confidence
        filtered = [c for c in causes if c['confidence'] >= min_confidence]
        
        # Sort by confidence * abs(deviation)
        ranked = sorted(
            filtered,
            key=lambda x: x['confidence'] * abs(x['deviation']),
            reverse=True
        )
        
        return ranked
    
    def _create_root_cause_object(self, cause_data: Dict,
                                  defect_type: str,
                                  deviations: Dict,
                                  current_params: Dict,
                                  timestamp: datetime,
                                  rank: int) -> RootCause:
        """Create a RootCause object with full details"""
        param_name = cause_data['parameter']
        deviation = cause_data['deviation']
        confidence = cause_data['confidence']
        
        # Generate evidence
        evidence = [
            {
                'type': 'parameter_deviation',
                'parameter': param_name,
                'current_value': cause_data.get('current_value', 0),
                'normal_mean': self.parameter_stats.get(param_name, (0, 1))[0],
                'deviation_sigma': deviation,
                'timestamp': timestamp.isoformat()
            }
        ]
        
        # Generate recommended action
        recommended_action = self._generate_recommendation(
            defect_type, 
            param_name, 
            deviation,
            cause_data.get('current_value', 0)
        )
        
        root_cause_id = f"RC_{defect_type}_{param_name}_{rank}_{int(timestamp.timestamp())}"
        
        return RootCause(
            root_cause_id=root_cause_id,
            parameter_name=param_name,
            deviation_magnitude=deviation,
            confidence=confidence,
            evidence=evidence,
            recommended_action=recommended_action,
            timestamp=timestamp
        )
    
    def _generate_recommendation(self, defect_type: str, 
                                parameter: str,
                                deviation: float,
                                current_value: float) -> str:
        """Generate actionable recommendation"""
        recommendations = {
            ('crack', 'furnace_temperature'): "Reduce furnace temperature by 20-30°C",
            ('crack', 'cooling_rate'): "Decrease cooling rate to <5°C/min",
            ('bubble', 'forming_pressure'): "Stabilize forming pressure, reduce by 10 MPa",
            ('bubble', 'furnace_temperature'): "Reduce furnace temperature by 15-20°C",
            ('deformation', 'belt_speed'): "Reduce belt speed by 15%",
            ('deformation', 'mold_temperature'): "Increase mold temperature by 20-30°C",
            ('cloudiness', 'furnace_temperature'): "Increase furnace temperature by 30-50°C",
            ('chip', 'belt_speed'): "Reduce belt speed to <160 m/min",
            ('stain', 'mold_temperature'): "Adjust mold temperature to 300-340°C range"
        }
        
        key = (defect_type, parameter)
        recommendation = recommendations.get(key, f"Adjust {parameter} towards normal range")
        
        # Add specific value if deviation is significant
        if abs(deviation) > 2.0:
            recommendation += f" (URGENT: {abs(deviation):.1f}σ deviation)"
        
        return recommendation
    
    def get_historical_frequency(self, defect_type: str, 
                                parameter: str,
                                days_back: int = 7) -> float:
        """
        Get historical frequency of this parameter causing this defect
        
        Args:
            defect_type: Type of defect
            parameter: Parameter name
            days_back: How many days to look back
            
        Returns:
            Frequency (0-1) of this cause in historical data
        """
        # This would query InfluxDB or PostgreSQL for historical correlations
        # For now, return a mock value
        return 0.5


def analyze_root_cause(knowledge_graph, defect_type: str, 
                      current_parameters: Dict[str, float],
                      min_confidence: float = 0.5) -> List[Dict]:
    """
    Convenience function to analyze root cause
    
    Args:
        knowledge_graph: KnowledgeGraph instance
        defect_type: Type of defect
        current_parameters: Current parameter values
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of root cause dictionaries
    """
    analyzer = RootCauseAnalyzer(knowledge_graph)
    root_causes = analyzer.analyze_defect(defect_type, current_parameters, min_confidence=min_confidence)
    
    # Convert to dictionaries
    return [
        {
            'root_cause_id': rc.root_cause_id,
            'parameter_name': rc.parameter_name,
            'deviation_magnitude': rc.deviation_magnitude,
            'confidence': rc.confidence,
            'evidence': rc.evidence,
            'recommended_action': rc.recommended_action,
            'timestamp': rc.timestamp.isoformat()
        }
        for rc in root_causes
    ]
