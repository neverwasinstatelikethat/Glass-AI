"""
What-If Analyzer for Digital Twin
Allows operators to test parameter changes before applying them to production.

User Flow:
Operator selects parameter to adjust → Inputs new value → System runs simulation → 
Shows predicted outcome → Operator confirms or rejects
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of parameters that can be adjusted"""
    FURNACE_TEMPERATURE = "furnace_temperature"
    FURNACE_PRESSURE = "furnace_pressure"
    BELT_SPEED = "belt_speed"
    MOLD_TEMPERATURE = "mold_temperature"
    FORMING_PRESSURE = "forming_pressure"
    COOLING_RATE = "cooling_rate"


@dataclass
class ParameterConstraints:
    """Constraints for a parameter"""
    min_value: float
    max_value: float
    optimal_min: float
    optimal_max: float
    unit: str
    max_change_rate: float  # Maximum change per time unit
    critical_threshold: float  # Change requiring operator approval


# Define constraints for each parameter type
PARAMETER_CONSTRAINTS = {
    ParameterType.FURNACE_TEMPERATURE: ParameterConstraints(
        min_value=1200.0, max_value=1700.0,
        optimal_min=1450.0, optimal_max=1550.0,
        unit="°C", max_change_rate=50.0, critical_threshold=100.0
    ),
    ParameterType.FURNACE_PRESSURE: ParameterConstraints(
        min_value=0.0, max_value=50.0,
        optimal_min=10.0, optimal_max=20.0,
        unit="kPa", max_change_rate=5.0, critical_threshold=10.0
    ),
    ParameterType.BELT_SPEED: ParameterConstraints(
        min_value=50.0, max_value=200.0,
        optimal_min=130.0, optimal_max=170.0,
        unit="m/min", max_change_rate=20.0, critical_threshold=50.0
    ),
    ParameterType.MOLD_TEMPERATURE: ParameterConstraints(
        min_value=200.0, max_value=600.0,
        optimal_min=300.0, optimal_max=340.0,
        unit="°C", max_change_rate=30.0, critical_threshold=80.0
    ),
    ParameterType.FORMING_PRESSURE: ParameterConstraints(
        min_value=0.0, max_value=120.0,
        optimal_min=40.0, optimal_max=60.0,
        unit="MPa", max_change_rate=15.0, critical_threshold=30.0
    ),
    ParameterType.COOLING_RATE: ParameterConstraints(
        min_value=1.0, max_value=10.0,
        optimal_min=2.0, optimal_max=5.0,
        unit="°C/min", max_change_rate=2.0, critical_threshold=3.0
    )
}


@dataclass
class WhatIfScenario:
    """A what-if scenario for testing parameter changes"""
    scenario_id: str
    parameter_changes: Dict[ParameterType, float]
    baseline_state: Dict
    predicted_outcome: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ImpactAnalysis:
    """Analysis of impact for a what-if scenario"""
    defect_rate_change: float  # Percentage change in defect rate
    quality_score_impact: float  # Change in quality score
    production_rate_impact: float  # Units per hour change
    energy_consumption_change: float  # Percentage change in energy
    time_to_effect_minutes: int  # Time until change impacts output
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    warnings: List[str]  # List of warnings about the change
    recommendations: List[str]  # List of recommendations


class WhatIfAnalyzer:
    """
    What-If Analyzer that simulates parameter changes before applying them
    """
    
    def __init__(self, digital_twin):
        """
        Initialize What-If Analyzer
        
        Args:
            digital_twin: Digital Twin instance for running simulations
        """
        self.digital_twin = digital_twin
        self.scenario_history: List[WhatIfScenario] = []
        
        logger.info("✅ What-If Analyzer initialized")
    
    def validate_parameter_change(self, parameter: ParameterType, 
                                  current_value: float, 
                                  new_value: float) -> Tuple[bool, List[str]]:
        """
        Validate if a parameter change is safe and within constraints
        
        Args:
            parameter: Type of parameter to change
            current_value: Current parameter value
            new_value: Proposed new value
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        constraints = PARAMETER_CONSTRAINTS[parameter]
        warnings = []
        
        # Check absolute bounds
        if new_value < constraints.min_value or new_value > constraints.max_value:
            warnings.append(
                f"Value {new_value} {constraints.unit} is outside safe range "
                f"[{constraints.min_value}, {constraints.max_value}] {constraints.unit}"
            )
            return False, warnings
        
        # Check change magnitude
        change = abs(new_value - current_value)
        if change > constraints.critical_threshold:
            warnings.append(
                f"Large change detected: {change:.1f} {constraints.unit}. "
                f"Requires operator approval (threshold: {constraints.critical_threshold} {constraints.unit})"
            )
        
        # Check if moving away from optimal range
        in_optimal_before = constraints.optimal_min <= current_value <= constraints.optimal_max
        in_optimal_after = constraints.optimal_min <= new_value <= constraints.optimal_max
        
        if in_optimal_before and not in_optimal_after:
            warnings.append(
                f"Moving outside optimal range [{constraints.optimal_min}, {constraints.optimal_max}] {constraints.unit}"
            )
        
        return True, warnings
    
    def analyze_single_parameter_change(self, parameter: ParameterType, 
                                       new_value: float,
                                       current_state: Optional[Dict] = None) -> ImpactAnalysis:
        """
        Analyze impact of changing a single parameter
        
        Args:
            parameter: Parameter to change
            new_value: New value for the parameter
            current_state: Current production state (if None, uses digital twin state)
            
        Returns:
            ImpactAnalysis with predicted impacts
        """
        # Get current state
        if current_state is None:
            current_state = self.digital_twin.get_current_state()
        
        # Get current parameter value
        current_value = self._extract_parameter_value(parameter, current_state)
        
        # Validate change
        is_valid, warnings = self.validate_parameter_change(parameter, current_value, new_value)
        
        if not is_valid:
            return ImpactAnalysis(
                defect_rate_change=0.0,
                quality_score_impact=0.0,
                production_rate_impact=0.0,
                energy_consumption_change=0.0,
                time_to_effect_minutes=0,
                risk_level="CRITICAL",
                warnings=warnings,
                recommendations=["Parameter change is unsafe - rejected"]
            )
        
        # Create modified state
        modified_state = self._apply_parameter_change(current_state, parameter, new_value)
        
        # Simulate outcome
        predicted_outcome = self._simulate_state(modified_state)
        
        # Calculate impacts
        impacts = self._calculate_impacts(current_state, predicted_outcome, parameter, current_value, new_value)
        
        # Add warnings
        impacts.warnings.extend(warnings)
        
        return impacts
    
    def analyze_multi_parameter_optimization(self, parameter_changes: Dict[ParameterType, float],
                                            current_state: Optional[Dict] = None) -> ImpactAnalysis:
        """
        Analyze impact of changing multiple parameters simultaneously
        
        Args:
            parameter_changes: Dict mapping parameter types to new values
            current_state: Current production state
            
        Returns:
            ImpactAnalysis for combined changes
        """
        # Get current state
        if current_state is None:
            current_state = self.digital_twin.get_current_state()
        
        warnings = []
        
        # Validate all changes
        for parameter, new_value in parameter_changes.items():
            current_value = self._extract_parameter_value(parameter, current_state)
            is_valid, param_warnings = self.validate_parameter_change(parameter, current_value, new_value)
            
            if not is_valid:
                return ImpactAnalysis(
                    defect_rate_change=0.0, quality_score_impact=0.0,
                    production_rate_impact=0.0, energy_consumption_change=0.0,
                    time_to_effect_minutes=0, risk_level="CRITICAL",
                    warnings=param_warnings,
                    recommendations=[f"Parameter {parameter.value} change is unsafe"]
                )
            
            warnings.extend(param_warnings)
        
        # Apply all changes
        modified_state = current_state.copy()
        for parameter, new_value in parameter_changes.items():
            modified_state = self._apply_parameter_change(modified_state, parameter, new_value)
        
        # Simulate combined outcome
        predicted_outcome = self._simulate_state(modified_state)
        
        # Calculate impacts
        impacts = self._calculate_impacts(current_state, predicted_outcome, 
                                         list(parameter_changes.keys())[0],  # Use first param for time estimate
                                         0, 0)  # Dummy values for multi-param
        
        impacts.warnings = warnings
        
        return impacts
    
    def create_scenario(self, scenario_id: str, 
                       parameter_changes: Dict[ParameterType, float]) -> WhatIfScenario:
        """
        Create and store a what-if scenario
        
        Args:
            scenario_id: Unique identifier for the scenario
            parameter_changes: Parameters to change
            
        Returns:
            WhatIfScenario object
        """
        # Get baseline state
        baseline_state = self.digital_twin.get_current_state()
        
        # Create scenario
        scenario = WhatIfScenario(
            scenario_id=scenario_id,
            parameter_changes=parameter_changes,
            baseline_state=baseline_state
        )
        
        # Analyze scenario
        impact = self.analyze_multi_parameter_optimization(parameter_changes, baseline_state)
        
        # Apply changes and get predicted outcome
        modified_state = baseline_state.copy()
        for parameter, new_value in parameter_changes.items():
            modified_state = self._apply_parameter_change(modified_state, parameter, new_value)
        
        scenario.predicted_outcome = {
            'state': self._simulate_state(modified_state),
            'impact_analysis': impact
        }
        
        # Store scenario
        self.scenario_history.append(scenario)
        
        logger.info(f"Created what-if scenario: {scenario_id}")
        
        return scenario
    
    def compare_scenarios(self, scenario_ids: List[str]) -> Dict:
        """
        Compare multiple scenarios side by side
        
        Args:
            scenario_ids: List of scenario IDs to compare
            
        Returns:
            Comparison results
        """
        scenarios = [s for s in self.scenario_history if s.scenario_id in scenario_ids]
        
        if not scenarios:
            return {'error': 'No scenarios found with given IDs'}
        
        comparison = {
            'scenarios': [],
            'baseline': scenarios[0].baseline_state if scenarios else None
        }
        
        for scenario in scenarios:
            if scenario.predicted_outcome:
                comparison['scenarios'].append({
                    'id': scenario.scenario_id,
                    'changes': {p.value: v for p, v in scenario.parameter_changes.items()},
                    'impact': scenario.predicted_outcome.get('impact_analysis')
                })
        
        return comparison
    
    def _extract_parameter_value(self, parameter: ParameterType, state: Dict) -> float:
        """Extract current value of a parameter from state"""
        if parameter == ParameterType.FURNACE_TEMPERATURE:
            return state.get('furnace', {}).get('temperature', 1500.0)
        elif parameter == ParameterType.FURNACE_PRESSURE:
            return state.get('furnace', {}).get('pressure', 15.0)
        elif parameter == ParameterType.BELT_SPEED:
            return state.get('forming', {}).get('belt_speed', 150.0)
        elif parameter == ParameterType.MOLD_TEMPERATURE:
            return state.get('forming', {}).get('mold_temp', 320.0)
        elif parameter == ParameterType.FORMING_PRESSURE:
            return state.get('forming', {}).get('pressure', 50.0)
        elif parameter == ParameterType.COOLING_RATE:
            return state.get('annealing', {}).get('cooling_rate', 3.5)
        
        return 0.0
    
    def _apply_parameter_change(self, state: Dict, parameter: ParameterType, 
                               new_value: float) -> Dict:
        """Apply a parameter change to a state"""
        modified_state = state.copy()
        
        if parameter == ParameterType.FURNACE_TEMPERATURE:
            if 'furnace' not in modified_state:
                modified_state['furnace'] = {}
            modified_state['furnace']['temperature'] = new_value
        elif parameter == ParameterType.FURNACE_PRESSURE:
            if 'furnace' not in modified_state:
                modified_state['furnace'] = {}
            modified_state['furnace']['pressure'] = new_value
        elif parameter == ParameterType.BELT_SPEED:
            if 'forming' not in modified_state:
                modified_state['forming'] = {}
            modified_state['forming']['belt_speed'] = new_value
        elif parameter == ParameterType.MOLD_TEMPERATURE:
            if 'forming' not in modified_state:
                modified_state['forming'] = {}
            modified_state['forming']['mold_temp'] = new_value
        elif parameter == ParameterType.FORMING_PRESSURE:
            if 'forming' not in modified_state:
                modified_state['forming'] = {}
            modified_state['forming']['pressure'] = new_value
        
        return modified_state
    
    def _simulate_state(self, state: Dict) -> Dict:
        """Simulate production with modified state"""
        # Extract parameters from state
        furnace_temp = state.get('furnace', {}).get('temperature', 1500.0)
        furnace_pressure = state.get('furnace', {}).get('pressure', 15.0)
        melt_level = state.get('furnace', {}).get('melt_level', 2500.0)
        
        belt_speed = state.get('forming', {}).get('belt_speed', 150.0)
        mold_temp = state.get('forming', {}).get('mold_temp', 320.0)
        forming_pressure = state.get('forming', {}).get('pressure', 50.0)
        
        # Use digital twin's calculation methods
        quality_score = self.digital_twin._calculate_quality_score(
            furnace_temp, melt_level, belt_speed, mold_temp
        )
        
        defects = self.digital_twin._calculate_defect_probabilities(
            furnace_temp, melt_level, belt_speed, mold_temp, forming_pressure
        )
        
        return {
            'furnace': state.get('furnace', {}),
            'forming': state.get('forming', {}),
            'quality_score': quality_score,
            'defects': defects
        }
    
    def _calculate_impacts(self, baseline: Dict, predicted: Dict,
                          parameter: ParameterType, old_value: float, 
                          new_value: float) -> ImpactAnalysis:
        """Calculate impact metrics"""
        # Quality score impact
        baseline_quality = baseline.get('quality_score', 0.85)
        predicted_quality = predicted.get('quality_score', 0.85)
        quality_impact = predicted_quality - baseline_quality
        
        # Defect rate change
        baseline_defects = baseline.get('defects', {})
        predicted_defects = predicted.get('defects', {})
        
        baseline_defect_rate = sum(baseline_defects.values()) / len(baseline_defects) if baseline_defects else 0.1
        predicted_defect_rate = sum(predicted_defects.values()) / len(predicted_defects) if predicted_defects else 0.1
        
        defect_rate_change = ((predicted_defect_rate - baseline_defect_rate) / baseline_defect_rate * 100) if baseline_defect_rate > 0 else 0
        
        # Production rate impact (simplified)
        baseline_speed = baseline.get('forming', {}).get('belt_speed', 150.0)
        predicted_speed = predicted.get('forming', {}).get('belt_speed', 150.0)
        production_impact = (predicted_speed - baseline_speed) / baseline_speed * 100
        
        # Energy consumption (simplified estimation)
        energy_change = 0.0
        if parameter == ParameterType.FURNACE_TEMPERATURE:
            # Higher temperature = more energy
            energy_change = (new_value - old_value) / old_value * 100 if old_value > 0 else 0
        
        # Time to effect
        time_to_effect = self._estimate_time_to_effect(parameter)
        
        # Risk level
        risk_level = self._assess_risk_level(quality_impact, defect_rate_change, parameter)
        
        # Recommendations
        recommendations = self._generate_recommendations(
            quality_impact, defect_rate_change, parameter, old_value, new_value
        )
        
        return ImpactAnalysis(
            defect_rate_change=defect_rate_change,
            quality_score_impact=quality_impact,
            production_rate_impact=production_impact,
            energy_consumption_change=energy_change,
            time_to_effect_minutes=time_to_effect,
            risk_level=risk_level,
            warnings=[],
            recommendations=recommendations
        )
    
    def _estimate_time_to_effect(self, parameter: ParameterType) -> int:
        """Estimate time until parameter change affects output"""
        time_estimates = {
            ParameterType.FURNACE_TEMPERATURE: 25,  # 15-30 min
            ParameterType.FURNACE_PRESSURE: 10,
            ParameterType.BELT_SPEED: 1,  # Almost immediate
            ParameterType.MOLD_TEMPERATURE: 15,
            ParameterType.FORMING_PRESSURE: 2,
            ParameterType.COOLING_RATE: 3
        }
        return time_estimates.get(parameter, 10)
    
    def _assess_risk_level(self, quality_impact: float, defect_change: float,
                          parameter: ParameterType) -> str:
        """Assess risk level of the change"""
        # Critical if quality drops significantly
        if quality_impact < -0.15 or defect_change > 50:
            return "CRITICAL"
        
        # High risk if moving outside optimal range significantly
        if quality_impact < -0.05 or defect_change > 20:
            return "HIGH"
        
        # Medium risk if minor negative impact
        if quality_impact < 0 or defect_change > 5:
            return "MEDIUM"
        
        # Low risk if improving or neutral
        return "LOW"
    
    def _generate_recommendations(self, quality_impact: float, defect_change: float,
                                 parameter: ParameterType, old_value: float, 
                                 new_value: float) -> List[str]:
        """Generate recommendations based on impact analysis"""
        recommendations = []
        
        if quality_impact > 0.05:
            recommendations.append(f"✅ Change is expected to improve quality by {quality_impact*100:.1f}%")
        elif quality_impact < -0.05:
            recommendations.append(f"⚠️ Change may reduce quality by {abs(quality_impact)*100:.1f}%")
        
        if defect_change < -10:
            recommendations.append(f"✅ Expected defect reduction of {abs(defect_change):.1f}%")
        elif defect_change > 10:
            recommendations.append(f"⚠️ Expected defect increase of {defect_change:.1f}%")
        
        # Parameter-specific recommendations
        constraints = PARAMETER_CONSTRAINTS.get(parameter)
        if constraints:
            if new_value < constraints.optimal_min or new_value > constraints.optimal_max:
                recommendations.append(
                    f"💡 Consider targeting optimal range: "
                    f"{constraints.optimal_min}-{constraints.optimal_max} {constraints.unit}"
                )
        
        if not recommendations:
            recommendations.append("ℹ️ Change is within acceptable parameters")
        
        return recommendations
