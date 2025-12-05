"""
Three.js Renderer for Digital Twin Visualization
Web-based 3D visualization of furnace and glass forming processes
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreeJSRenderer:
    """
    Three.js renderer for creating 3D visualizations of glass manufacturing processes
    Generates JSON scene data that can be consumed by a Three.js frontend
    """
    
    def __init__(
        self,
        container_id: str = "visualization-container",
        width: int = 800,
        height: int = 600
    ):
        """
        Args:
            container_id: HTML container ID for the visualization
            width: Canvas width in pixels
            height: Canvas height in pixels
        """
        self.container_id = container_id
        self.width = width
        self.height = height
        
        # Scene state
        self.scene_objects = []
        self.cameras = []
        self.lights = []
        self.materials = {}
        self.geometries = {}
        
        # Default camera
        self.add_camera(
            camera_type="perspective",
            position=[0, 0, 10],
            target=[0, 0, 0],
            fov=75,
            near=0.1,
            far=1000
        )
        
        # Default lights
        self.add_light(
            light_type="ambient",
            color=0xffffff,
            intensity=0.5
        )
        
        self.add_light(
            light_type="directional",
            color=0xffffff,
            intensity=0.8,
            position=[5, 10, 7]
        )
        
        logger.info(f"Initialized ThreeJSRenderer: {width}x{height}")
    
    def add_camera(
        self,
        camera_type: str = "perspective",
        position: List[float] = None,
        target: List[float] = None,
        **kwargs
    ):
        """
        Add a camera to the scene
        
        Args:
            camera_type: "perspective" or "orthographic"
            position: Camera position [x, y, z]
            target: Camera target [x, y, z]
            **kwargs: Camera-specific parameters
        """
        camera = {
            "type": camera_type,
            "position": position or [0, 0, 10],
            "target": target or [0, 0, 0],
            **kwargs
        }
        
        self.cameras.append(camera)
        logger.debug(f"Added {camera_type} camera at {position}")
    
    def add_light(
        self,
        light_type: str,
        color: int = 0xffffff,
        intensity: float = 1.0,
        position: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Add a light to the scene
        
        Args:
            light_type: "ambient", "directional", "point", "spot"
            color: Light color (hex)
            intensity: Light intensity
            position: Light position [x, y, z] (for point/spot/directional)
            **kwargs: Light-specific parameters
        """
        light = {
            "type": light_type,
            "color": color,
            "intensity": intensity,
            **kwargs
        }
        
        if position is not None:
            light["position"] = position
        
        self.lights.append(light)
        logger.debug(f"Added {light_type} light with intensity {intensity}")
    
    def add_furnace_geometry(
        self,
        dimensions: Tuple[float, float, float] = (50.0, 5.0, 3.0),
        position: List[float] = None,
        temperature_field: Optional[np.ndarray] = None,
        show_temperature_colors: bool = True
    ):
        """
        Add furnace geometry to the scene
        
        Args:
            dimensions: Furnace dimensions (length, width, height) in meters
            position: Furnace position [x, y, z]
            temperature_field: 3D temperature field for color mapping
            show_temperature_colors: Whether to color by temperature
        """
        length, width, height = dimensions
        position = position or [0, 0, 0]
        
        # Create furnace body (rectangular prism)
        furnace_geometry = {
            "type": "box",
            "width": width,
            "height": height,
            "depth": length,
            "widthSegments": 1,
            "heightSegments": 1,
            "depthSegments": 20  # More segments for temperature visualization
        }
        
        # Material based on temperature or default
        if show_temperature_colors and temperature_field is not None:
            # Create temperature-based material with color mapping
            material_id = f"furnace_temp_material_{len(self.materials)}"
            self.materials[material_id] = {
                "type": "shader",
                "vertexShader": self._get_furnace_vertex_shader(),
                "fragmentShader": self._get_furnace_fragment_shader(),
                "uniforms": {
                    "temperatureData": self._encode_temperature_texture(temperature_field),
                    "minTemp": float(np.min(temperature_field)),
                    "maxTemp": float(np.max(temperature_field))
                }
            }
        else:
            # Default furnace material (metallic)
            material_id = "furnace_material"
            if material_id not in self.materials:
                self.materials[material_id] = {
                    "type": "meshStandardMaterial",
                    "color": 0x444444,
                    "metalness": 0.8,
                    "roughness": 0.4
                }
        
        # Add furnace object to scene
        furnace_object = {
            "type": "mesh",
            "geometry": furnace_geometry,
            "material": material_id,
            "position": position,
            "rotation": [0, 0, 0],
            "name": "glass_furnace"
        }
        
        self.scene_objects.append(furnace_object)
        logger.debug(f"Added furnace geometry: {dimensions}")
    
    def _encode_temperature_texture(self, temperature_field: np.ndarray) -> Dict:
        """
        Encode temperature field as texture data
        
        Args:
            temperature_field: 3D temperature field
            
        Returns:
            Texture data dictionary
        """
        # Simplify to 2D for visualization (top surface)
        if len(temperature_field.shape) == 3:
            temp_2d = temperature_field[:, :, -1]  # Top surface
        else:
            temp_2d = temperature_field
        
        # Normalize temperature data
        min_temp = np.min(temp_2d)
        max_temp = np.max(temp_2d)
        normalized = (temp_2d - min_temp) / (max_temp - min_temp) if max_temp > min_temp else temp_2d
        
        # Convert to image data (grayscale)
        image_data = (normalized * 255).astype(np.uint8)
        
        return {
            "data": image_data.tolist(),
            "width": image_data.shape[1],
            "height": image_data.shape[0],
            "format": "grayscale"
        }
    
    def _get_furnace_vertex_shader(self) -> str:
        """Get vertex shader for furnace temperature visualization"""
        return """
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        """
    
    def _get_furnace_fragment_shader(self) -> str:
        """Get fragment shader for furnace temperature visualization"""
        return """
            uniform sampler2D temperatureData;
            uniform float minTemp;
            uniform float maxTemp;
            varying vec2 vUv;
            
            vec3 temperatureColor(float temp) {
                // Simple color mapping: blue (cold) -> red (hot)
                float normTemp = (temp - minTemp) / (maxTemp - minTemp);
                return mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), normTemp);
            }
            
            void main() {
                float temp = texture2D(temperatureData, vUv).r * (maxTemp - minTemp) + minTemp;
                gl_FragColor = vec4(temperatureColor(temp), 1.0);
            }
        """
    
    def add_glass_melt_geometry(
        self,
        position: List[float] = None,
        dimensions: Tuple[float, float, float] = (40.0, 4.0, 1.0),
        temperature: float = 1500.0,
        viscosity: Optional[float] = None
    ):
        """
        Add glass melt visualization
        
        Args:
            position: Position of glass melt
            dimensions: Dimensions of melt pool
            temperature: Melt temperature for color
            viscosity: Viscosity for surface appearance
        """
        position = position or [0, 0, 1.5]  # Middle height of furnace
        length, width, height = dimensions
        
        # Create melt geometry
        melt_geometry = {
            "type": "box",
            "width": width,
            "height": height,
            "depth": length,
            "widthSegments": 10,
            "heightSegments": 1,
            "depthSegments": 40
        }
        
        # Color based on temperature
        temp_normalized = np.clip((temperature - 1000) / 1000, 0, 1)
        red = np.clip(temp_normalized * 2, 0, 1)
        green = np.clip(1 - abs(temp_normalized - 0.5) * 2, 0, 1)
        blue = np.clip((1 - temp_normalized) * 2, 0, 1)
        
        color_hex = int(f"0x{int(red*255):02x}{int(green*255):02x}{int(blue*255):02x}", 16)
        
        # Material properties based on viscosity
        roughness = 0.1 if viscosity and viscosity < 100 else 0.3
        metalness = 0.8 if viscosity and viscosity < 50 else 0.5
        
        material_id = f"melt_material_{len(self.materials)}"
        self.materials[material_id] = {
            "type": "meshStandardMaterial",
            "color": color_hex,
            "roughness": roughness,
            "metalness": metalness,
            "transparent": True,
            "opacity": 0.8
        }
        
        # Add melt object
        melt_object = {
            "type": "mesh",
            "geometry": melt_geometry,
            "material": material_id,
            "position": position,
            "rotation": [0, 0, 0],
            "name": "glass_melt"
        }
        
        self.scene_objects.append(melt_object)
        logger.debug(f"Added glass melt geometry at {position}")
    
    def add_defect_indicators(
        self,
        defects: Dict[str, float],
        furnace_position: List[float] = None
    ):
        """
        Add visual indicators for defects
        
        Args:
            defects: Dictionary of defect types and likelihoods
            furnace_position: Reference position for defect indicators
        """
        furnace_position = furnace_position or [0, 0, 0]
        
        for defect_type, likelihood in defects.items():
            if likelihood > 0.1:  # Only show significant defects
                # Position indicators along furnace length
                x_pos = furnace_position[0] + (np.random.random() - 0.5) * 20
                y_pos = furnace_position[1] + (np.random.random() - 0.5) * 2
                z_pos = furnace_position[2] + (np.random.random() - 0.5) * 1
                
                # Color coding
                if defect_type in ['crack', 'stress']:
                    color = 0xff0000  # Red
                elif defect_type in ['bubble', 'cloudiness']:
                    color = 0xffff00  # Yellow
                elif defect_type in ['chip', 'deformation']:
                    color = 0xff8800  # Orange
                else:
                    color = 0xff00ff  # Magenta
                
                # Size based on likelihood
                size = 0.2 + likelihood * 0.8
                
                # Create indicator geometry
                indicator_geometry = {
                    "type": "sphere",
                    "radius": size,
                    "widthSegments": 16,
                    "heightSegments": 16
                }
                
                material_id = f"defect_{defect_type}_material"
                if material_id not in self.materials:
                    self.materials[material_id] = {
                        "type": "meshStandardMaterial",
                        "color": color,
                        "emissive": color,
                        "emissiveIntensity": likelihood,
                        "transparent": True,
                        "opacity": 0.7
                    }
                
                # Add indicator object
                indicator_object = {
                    "type": "mesh",
                    "geometry": indicator_geometry,
                    "material": material_id,
                    "position": [x_pos, y_pos, z_pos],
                    "name": f"defect_{defect_type}"
                }
                
                self.scene_objects.append(indicator_object)
    
    def add_coordinate_system(
        self,
        size: float = 5.0,
        position: List[float] = None
    ):
        """
        Add coordinate system axes for reference
        
        Args:
            size: Length of axes
            position: Position of coordinate system origin
        """
        position = position or [0, 0, 0]
        
        # X-axis (red)
        self._add_axis_line(
            start=position,
            end=[position[0] + size, position[1], position[2]],
            color=0xff0000,
            name="x_axis"
        )
        
        # Y-axis (green)
        self._add_axis_line(
            start=position,
            end=[position[0], position[1] + size, position[2]],
            color=0x00ff00,
            name="y_axis"
        )
        
        # Z-axis (blue)
        self._add_axis_line(
            start=position,
            end=[position[0], position[1], position[2] + size],
            color=0x0000ff,
            name="z_axis"
        )
    
    def _add_axis_line(
        self,
        start: List[float],
        end: List[float],
        color: int,
        name: str
    ):
        """
        Add a line for coordinate axis
        
        Args:
            start: Start position
            end: End position
            color: Line color
            name: Object name
        """
        # Create line geometry
        line_geometry = {
            "type": "line",
            "points": [start, end]
        }
        
        material_id = f"line_material_{color}"
        if material_id not in self.materials:
            self.materials[material_id] = {
                "type": "lineBasicMaterial",
                "color": color,
                "linewidth": 2
            }
        
        # Add line object
        line_object = {
            "type": "line",
            "geometry": line_geometry,
            "material": material_id,
            "name": name
        }
        
        self.scene_objects.append(line_object)
    
    def generate_scene_json(self) -> str:
        """
        Generate complete scene JSON for Three.js
        
        Returns:
            JSON string representing the 3D scene
        """
        scene_data = {
            "metadata": {
                "version": 1.0,
                "type": "ThreeJSScene",
                "generator": "DigitalTwinThreeJSRenderer",
                "timestamp": datetime.now().isoformat()
            },
            "container": {
                "id": self.container_id,
                "width": self.width,
                "height": self.height
            },
            "scene": {
                "objects": self.scene_objects,
                "cameras": self.cameras,
                "lights": self.lights,
                "materials": self.materials
            }
        }
        
        return json.dumps(scene_data, indent=2)
    
    def update_scene(
        self,
        temperature_field: Optional[np.ndarray] = None,
        defects: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        """
        Update scene with new data and regenerate JSON
        
        Args:
            temperature_field: New temperature field
            defects: Updated defect information
            **kwargs: Other scene updates
            
        Returns:
            Updated scene JSON
        """
        # Clear previous objects (except cameras and lights)
        self.scene_objects = []
        
        # Re-add main geometry with updated data
        if temperature_field is not None:
            self.add_furnace_geometry(
                temperature_field=temperature_field,
                show_temperature_colors=True
            )
            
            # Add glass melt with temperature-based properties
            avg_temp = np.mean(temperature_field)
            self.add_glass_melt_geometry(temperature=avg_temp)
        
        # Add defect indicators if provided
        if defects:
            self.add_defect_indicators(defects)
        
        # Add coordinate system for reference
        self.add_coordinate_system()
        
        # Generate updated scene
        return self.generate_scene_json()


def create_threejs_renderer(**kwargs) -> ThreeJSRenderer:
    """
    Factory function to create a ThreeJSRenderer instance
    
    Args:
        **kwargs: Parameters for ThreeJSRenderer
        
    Returns:
        ThreeJSRenderer instance
    """
    renderer = ThreeJSRenderer(**kwargs)
    logger.info("Created ThreeJSRenderer")
    return renderer


if __name__ == "__main__":
    # Example usage
    print("Testing ThreeJSRenderer...")
    
    # Create renderer
    renderer = create_threejs_renderer(width=1024, height=768)
    
    # Generate test temperature field
    nx, ny, nz = 20, 10, 5
    temperature_field = np.zeros((nx, ny, nz))
    
    # Create temperature gradient (hot in center)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                center_dist = np.sqrt((i - nx/2)**2 + (j - ny/2)**2 + (k - nz/2)**2)
                temperature_field[i, j, k] = 1000 + 800 * np.exp(-center_dist**2 / 50)
    
    # Add some defects
    defects = {
        'crack': 0.7,
        'bubble': 0.3,
        'cloudiness': 0.5
    }
    
    # Update scene with data
    scene_json = renderer.update_scene(
        temperature_field=temperature_field,
        defects=defects
    )
    
    # Save to file for inspection
    with open("sample_scene.json", "w") as f:
        f.write(scene_json)
    
    print(f"Generated scene JSON ({len(scene_json)} characters)")
    print("Sample scene saved to 'sample_scene.json'")
    
    # Print scene summary
    scene_data = json.loads(scene_json)
    print(f"\nScene Summary:")
    print(f"  Objects: {len(scene_data['scene']['objects'])}")
    print(f"  Cameras: {len(scene_data['scene']['cameras'])}")
    print(f"  Lights: {len(scene_data['scene']['lights'])}")
    print(f"  Materials: {len(scene_data['scene']['materials'])}")
    
    print("\nThreeJSRenderer testing completed!")