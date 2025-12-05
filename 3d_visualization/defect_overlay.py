"""
Defect Overlay for 3D Visualization
Overlay defect information on 3D models with spatial mapping
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefectType(Enum):
    """Types of defects in glass manufacturing"""
    CRACK = "crack"
    BUBBLE = "bubble"
    CHIP = "chip"
    CLOUDINESS = "cloudiness"
    DEFORMATION = "deformation"
    STRESS = "stress"
    STRIATION = "striation"
    SEED = "seed"


@dataclass
class Defect:
    """Defect representation with spatial and severity information"""
    defect_type: DefectType
    position: Tuple[float, float, float]  # x, y, z coordinates (meters)
    severity: float  # 0.0 to 1.0
    size: Tuple[float, float, float]  # width, height, depth (meters)
    confidence: float  # Detection confidence 0.0 to 1.0
    timestamp: float = 0.0  # Time of detection (seconds)
    id: Optional[str] = None  # Unique identifier


class DefectOverlay:
    """
    Defect overlay system for 3D visualization
    Maps 2D defect detections to 3D positions and creates visual overlays
    """
    
    def __init__(
        self,
        furnace_dimensions: Tuple[float, float, float] = (50.0, 5.0, 3.0),
        camera_positions: Optional[List[Tuple[float, float, float]]] = None
    ):
        """
        Args:
            furnace_dimensions: Furnace dimensions (length, width, height) in meters
            camera_positions: List of camera positions for triangulation
        """
        self.furnace_length, self.furnace_width, self.furnace_height = furnace_dimensions
        self.camera_positions = camera_positions or [
            (0, -self.furnace_width/2 - 1, self.furnace_height/2),  # Left side camera
            (0, self.furnace_width/2 + 1, self.furnace_height/2),   # Right side camera
            (self.furnace_length/2, 0, self.furnace_height + 1),    # Top camera
        ]
        
        self.defects: List[Defect] = []
        self.defect_history: List[Defect] = []
        self.max_history_size = 1000
        
        # Defect visualization properties
        self.defect_colors = {
            DefectType.CRACK: (1.0, 0.0, 0.0),      # Red
            DefectType.BUBBLE: (1.0, 1.0, 0.0),     # Yellow
            DefectType.CHIP: (1.0, 0.5, 0.0),       # Orange
            DefectType.CLOUDINESS: (0.8, 0.8, 0.8), # Light gray
            DefectType.DEFORMATION: (0.5, 0.0, 1.0), # Purple
            DefectType.STRESS: (1.0, 0.0, 0.5),     # Magenta
            DefectType.STRIATION: (0.0, 1.0, 1.0),  # Cyan
            DefectType.SEED: (0.0, 0.5, 0.0),       # Dark green
        }
        
        logger.info(f"Initialized DefectOverlay for {furnace_dimensions} furnace")
    
    def add_defect(
        self,
        defect_type: Union[DefectType, str],
        position_2d: Tuple[float, float],
        camera_index: int,
        severity: float = 0.5,
        size_2d: Tuple[float, float] = (0.01, 0.01),
        confidence: float = 1.0,
        timestamp: float = 0.0
    ) -> Defect:
        """
        Add a 2D defect detection and map it to 3D space
        
        Args:
            defect_type: Type of defect
            position_2d: 2D position in camera coordinates (normalized -1 to 1)
            camera_index: Index of camera that detected the defect
            severity: Defect severity (0.0 to 1.0)
            size_2d: 2D size in camera coordinates
            confidence: Detection confidence
            timestamp: Detection time
            
        Returns:
            Created Defect object
        """
        # Convert string to enum if needed
        if isinstance(defect_type, str):
            try:
                defect_type = DefectType(defect_type.lower())
            except ValueError:
                logger.warning(f"Unknown defect type: {defect_type}")
                defect_type = DefectType.CRACK  # Default to crack
        
        # Map 2D position to 3D using camera projection
        position_3d = self._map_2d_to_3d(position_2d, camera_index)
        
        # Estimate 3D size based on 2D size and distance
        size_3d = self._estimate_3d_size(size_2d, position_3d, camera_index)
        
        # Create defect object
        defect = Defect(
            defect_type=defect_type,
            position=position_3d,
            severity=severity,
            size=size_3d,
            confidence=confidence,
            timestamp=timestamp,
            id=f"defect_{len(self.defects) + len(self.defect_history)}"
        )
        
        # Add to defects list
        self.defects.append(defect)
        
        # Maintain history
        self.defect_history.append(defect)
        if len(self.defect_history) > self.max_history_size:
            self.defect_history.pop(0)
        
        logger.debug(f"Added {defect_type.value} defect at {position_3d}")
        return defect
    
    def _map_2d_to_3d(
        self,
        position_2d: Tuple[float, float],
        camera_index: int
    ) -> Tuple[float, float, float]:
        """
        Map 2D camera coordinates to 3D world coordinates
        
        Args:
            position_2d: Normalized 2D position (-1 to 1)
            camera_index: Camera index
            
        Returns:
            3D world coordinates (x, y, z)
        """
        if camera_index >= len(self.camera_positions):
            logger.warning(f"Invalid camera index: {camera_index}")
            camera_index = 0
        
        # Camera position
        cam_x, cam_y, cam_z = self.camera_positions[camera_index]
        
        # Normalize 2D coordinates
        norm_x, norm_y = position_2d
        
        # Simple projection model (this would be more complex in reality)
        # Assuming cameras are looking at the furnace center
        target_x = norm_x * self.furnace_length / 2
        target_y = norm_y * self.furnace_width / 2
        target_z = self.furnace_height / 2
        
        # Interpolate between camera and target based on furnace depth
        # This is a simplified model - real implementation would use camera matrices
        interp_factor = 0.7  # How far along the ray to place the defect
        
        world_x = cam_x + (target_x - cam_x) * interp_factor
        world_y = cam_y + (target_y - cam_y) * interp_factor
        world_z = cam_z + (target_z - cam_z) * interp_factor
        
        # Clamp to furnace bounds
        world_x = np.clip(world_x, 0, self.furnace_length)
        world_y = np.clip(world_y, -self.furnace_width/2, self.furnace_width/2)
        world_z = np.clip(world_z, 0, self.furnace_height)
        
        return (float(world_x), float(world_y), float(world_z))
    
    def _estimate_3d_size(
        self,
        size_2d: Tuple[float, float],
        position_3d: Tuple[float, float, float],
        camera_index: int
    ) -> Tuple[float, float, float]:
        """
        Estimate 3D size from 2D size and position
        
        Args:
            size_2d: 2D size in normalized coordinates
            position_3d: 3D position
            camera_index: Camera index
            
        Returns:
            Estimated 3D size (width, height, depth)
        """
        # Distance from camera to defect
        cam_pos = self.camera_positions[camera_index]
        distance = np.sqrt(
            (position_3d[0] - cam_pos[0])**2 +
            (position_3d[1] - cam_pos[1])**2 +
            (position_3d[2] - cam_pos[2])**2
        )
        
        # Simple inverse perspective scaling
        # In reality, this would use camera intrinsic parameters
        scale_factor = max(0.1, distance / 10.0)
        
        width_3d = abs(size_2d[0]) * self.furnace_width * scale_factor
        height_3d = abs(size_2d[1]) * self.furnace_height * scale_factor
        depth_3d = min(width_3d, height_3d) * 0.5  # Assume depth is smaller
        
        return (
            float(width_3d),
            float(height_3d),
            float(depth_3d)
        )
    
    def triangulate_defect(
        self,
        defect_type: Union[DefectType, str],
        positions_2d: List[Tuple[float, float]],
        camera_indices: List[int],
        severities: Optional[List[float]] = None,
        confidences: Optional[List[float]] = None,
        timestamp: float = 0.0
    ) -> Optional[Defect]:
        """
        Triangulate defect position using multiple camera views
        
        Args:
            defect_type: Type of defect
            positions_2d: List of 2D positions from different cameras
            camera_indices: List of camera indices
            severities: List of severity values (optional)
            confidences: List of confidence values (optional)
            timestamp: Detection time
            
        Returns:
            Triangulated Defect object or None if triangulation fails
        """
        if len(positions_2d) < 2:
            logger.warning("Need at least 2 camera views for triangulation")
            return None
        
        if severities is None:
            severities = [0.5] * len(positions_2d)
        if confidences is None:
            confidences = [1.0] * len(positions_2d)
        
        # Convert to enum if needed
        if isinstance(defect_type, str):
            try:
                defect_type = DefectType(defect_type.lower())
            except ValueError:
                defect_type = DefectType.CRACK
        
        # Simple triangulation by averaging projected positions
        positions_3d = []
        weights = []
        
        for i, (pos_2d, cam_idx) in enumerate(zip(positions_2d, camera_indices)):
            pos_3d = self._map_2d_to_3d(pos_2d, cam_idx)
            positions_3d.append(pos_3d)
            # Weight by confidence and severity
            weights.append(confidences[i] * severities[i])
        
        # Weighted average of positions
        total_weight = sum(weights)
        if total_weight > 0:
            avg_x = sum(pos[0] * w for pos, w in zip(positions_3d, weights)) / total_weight
            avg_y = sum(pos[1] * w for pos, w in zip(positions_3d, weights)) / total_weight
            avg_z = sum(pos[2] * w for pos, w in zip(positions_3d, weights)) / total_weight
            
            # Average severity and confidence
            avg_severity = sum(sev * w for sev, w in zip(severities, weights)) / total_weight
            avg_confidence = sum(conf for conf in confidences) / len(confidences)
            
            # Estimate size from first view (simplified)
            size_3d = self._estimate_3d_size(positions_2d[0], (avg_x, avg_y, avg_z), camera_indices[0])
            
            # Create triangulated defect
            defect = Defect(
                defect_type=defect_type,
                position=(avg_x, avg_y, avg_z),
                severity=avg_severity,
                size=size_3d,
                confidence=avg_confidence,
                timestamp=timestamp,
                id=f"triangulated_{len(self.defects) + len(self.defect_history)}"
            )
            
            self.defects.append(defect)
            self.defect_history.append(defect)
            if len(self.defect_history) > self.max_history_size:
                self.defect_history.pop(0)
            
            logger.debug(f"Triangulated {defect_type.value} defect at ({avg_x:.2f}, {avg_y:.2f}, {avg_z:.2f})")
            return defect
        
        return None
    
    def get_defects_in_region(
        self,
        region_min: Tuple[float, float, float],
        region_max: Tuple[float, float, float]
    ) -> List[Defect]:
        """
        Get defects within a 3D region
        
        Args:
            region_min: Minimum coordinates (x, y, z)
            region_max: Maximum coordinates (x, y, z)
            
        Returns:
            List of defects in the region
        """
        defects_in_region = []
        
        for defect in self.defects:
            x, y, z = defect.position
            if (region_min[0] <= x <= region_max[0] and
                region_min[1] <= y <= region_max[1] and
                region_min[2] <= z <= region_max[2]):
                defects_in_region.append(defect)
        
        return defects_in_region
    
    def get_defects_by_type(self, defect_type: Union[DefectType, str]) -> List[Defect]:
        """
        Get all defects of a specific type
        
        Args:
            defect_type: Defect type to filter by
            
        Returns:
            List of defects of the specified type
        """
        if isinstance(defect_type, str):
            try:
                defect_type = DefectType(defect_type.lower())
            except ValueError:
                return []
        
        return [d for d in self.defects if d.defect_type == defect_type]
    
    def clear_defects(self, keep_history: bool = True):
        """
        Clear current defects
        
        Args:
            keep_history: Whether to keep defects in history
        """
        if not keep_history:
            self.defect_history.clear()
        self.defects.clear()
        logger.info("Cleared defects")
    
    def get_defect_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about current defects
        
        Returns:
            Dictionary with defect statistics
        """
        if not self.defects:
            return {
                'total_defects': 0,
                'defects_by_type': {},
                'average_severity': 0.0,
                'average_confidence': 0.0
            }
        
        # Count by type
        type_counts = {}
        total_severity = 0.0
        total_confidence = 0.0
        
        for defect in self.defects:
            defect_type = defect.defect_type.value
            type_counts[defect_type] = type_counts.get(defect_type, 0) + 1
            total_severity += defect.severity
            total_confidence += defect.confidence
        
        return {
            'total_defects': len(self.defects),
            'defects_by_type': type_counts,
            'average_severity': total_severity / len(self.defects),
            'average_confidence': total_confidence / len(self.defects)
        }
    
    def get_defect_visualization_data(self) -> List[Dict]:
        """
        Get data for 3D visualization of defects
        
        Returns:
            List of dictionaries with visualization data
        """
        visualization_data = []
        
        for defect in self.defects:
            # Get color for defect type
            color = self.defect_colors.get(defect.defect_type, (1.0, 1.0, 1.0))  # Default white
            
            # Scale color by severity
            scaled_color = tuple(c * defect.severity for c in color)
            
            # Create visualization object
            viz_object = {
                'id': defect.id,
                'type': defect.defect_type.value,
                'position': defect.position,
                'size': defect.size,
                'color': scaled_color,
                'severity': defect.severity,
                'confidence': defect.confidence,
                'timestamp': defect.timestamp
            }
            
            visualization_data.append(viz_object)
        
        return visualization_data


def create_defect_overlay(**kwargs) -> DefectOverlay:
    """
    Factory function to create a DefectOverlay instance
    
    Args:
        **kwargs: Parameters for DefectOverlay
        
    Returns:
        DefectOverlay instance
    """
    overlay = DefectOverlay(**kwargs)
    logger.info("Created DefectOverlay")
    return overlay


if __name__ == "__main__":
    # Example usage
    print("Testing DefectOverlay...")
    
    # Create overlay
    overlay = create_defect_overlay()
    
    # Add some defects
    print("Adding sample defects...")
    
    # Add single-camera defects
    overlay.add_defect(
        defect_type=DefectType.CRACK,
        position_2d=(0.1, 0.2),
        camera_index=0,
        severity=0.8,
        size_2d=(0.05, 0.02),
        confidence=0.9
    )
    
    overlay.add_defect(
        defect_type=DefectType.BUBBLE,
        position_2d=(-0.3, 0.1),
        camera_index=1,
        severity=0.6,
        size_2d=(0.03, 0.03),
        confidence=0.85
    )
    
    # Add triangulated defect
    overlay.triangulate_defect(
        defect_type=DefectType.CHIP,
        positions_2d=[(0.2, -0.1), (-0.2, -0.15)],
        camera_indices=[0, 1],
        severities=[0.7, 0.65],
        confidences=[0.9, 0.85]
    )
    
    # Get statistics
    stats = overlay.get_defect_statistics()
    print(f"\nDefect Statistics:")
    print(f"  Total defects: {stats['total_defects']}")
    print(f"  By type: {stats['defects_by_type']}")
    print(f"  Average severity: {stats['average_severity']:.2f}")
    print(f"  Average confidence: {stats['average_confidence']:.2f}")
    
    # Get visualization data
    viz_data = overlay.get_defect_visualization_data()
    print(f"\nVisualization Data ({len(viz_data)} objects):")
    for i, obj in enumerate(viz_data[:3]):  # Show first 3
        print(f"  {i+1}. {obj['type']} at {obj['position']} - severity: {obj['severity']:.2f}")
    
    # Test region filtering
    region_defects = overlay.get_defects_in_region(
        region_min=(0, -1, 0),
        region_max=(25, 1, 3)
    )
    print(f"\nDefects in region (0, -1, 0) to (25, 1, 3): {len(region_defects)}")
    
    # Test type filtering
    cracks = overlay.get_defects_by_type(DefectType.CRACK)
    print(f"Crack defects: {len(cracks)}")
    
    print("\nDefectOverlay testing completed!")