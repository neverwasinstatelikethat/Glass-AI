"""
MIK-1 Camera Stream Connector for Glass Defect Detection
Real-time video streaming from MIK-1 inspection system with defect detection
"""

import cv2
import asyncio
import logging
import numpy as np
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import base64
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIK1CameraStream:
    """Real-time camera stream connector for MIK-1 inspection system"""
    
    def __init__(
        self,
        camera_source: str = "0",  # "0" for webcam, or IP address for network camera
        resolution: tuple = (1920, 1080),
        fps: int = 30,
        callback: Optional[Callable] = None
    ):
        self.camera_source = camera_source
        self.resolution = resolution
        self.fps = fps
        self.callback = callback
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_count = 0
        self.camera_available = True
        
        # Defect detection parameters
        self.defect_classifier = None
        self.min_defect_area = 100
        self.defect_threshold = 0.7
        
    async def connect(self) -> bool:
        """Connect to the MIK-1 camera stream"""
        try:
            # Convert camera source to appropriate type
            if self.camera_source.isdigit():
                source = int(self.camera_source)
            else:
                source = self.camera_source
            
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                logger.warning(f"⚠️ Cannot open camera source: {self.camera_source}. Will use simulator.")
                self.camera_available = False
                return True  # Still return True to indicate successful initialization
            
            # Set resolution and FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            logger.info(f"✅ Connected to MIK-1 camera: {self.camera_source}")
            logger.info(f"📹 Resolution: {self.resolution}, FPS: {self.fps}")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Error connecting to camera: {e}. Will use simulator.")
            self.camera_available = False
            return True  # Still return True to indicate successful initialization
    
    async def start_streaming(self, process_frames: bool = True):
        """Start streaming frames from the camera"""
        if self.camera_available and (not self.cap or not self.cap.isOpened()):
            logger.error("❌ Camera not connected")
            return
        
        self.running = True
        frame_interval = 1.0 / self.fps
        last_frame_time = 0
        
        logger.info("🔄 Starting MIK-1 camera stream...")
        
        while self.running:
            try:
                current_time = asyncio.get_event_loop().time()
                
                # Control frame rate
                if current_time - last_frame_time < frame_interval:
                    await asyncio.sleep(frame_interval - (current_time - last_frame_time))
                
                last_frame_time = current_time
                
                # If camera is not available, use simulator
                if not self.camera_available:
                    frame_data = await self._generate_simulated_frame()
                    raw_data = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "frame_id": self.frame_count,
                        "camera_id": "MIK1_Camera_01",
                        "frame_data": frame_data,
                        "resolution": self.resolution,
                        "fps": self.fps,
                        "simulated": True
                    }
                    
                    if self.callback:
                        await self.callback(raw_data)
                        
                    self.frame_count += 1
                    await asyncio.sleep(1.0 / self.fps)  # Simulate frame rate
                    continue
                
                # Read frame from actual camera
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame from camera")
                    await asyncio.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Process frame for defect detection if requested
                if process_frames:
                    processed_data = await self._process_frame(frame)
                    
                    if self.callback:
                        await self.callback(processed_data)
                else:
                    # Send raw frame data
                    frame_data = await self._encode_frame(frame)
                    raw_data = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "frame_id": self.frame_count,
                        "camera_id": "MIK1_Camera_01",
                        "frame_data": frame_data,
                        "resolution": self.resolution,
                        "fps": self.fps
                    }
                    
                    if self.callback:
                        await self.callback(raw_data)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
                
            except asyncio.CancelledError:
                logger.info("⏹️ Streaming cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in streaming loop: {e}")
                await asyncio.sleep(1)
    
    async def _generate_simulated_frame(self) -> str:
        """Generate a simulated frame for testing purposes"""
        try:
            # Create a simulated frame with random defects
            frame = np.random.randint(50, 200, (*self.resolution[::-1], 3), dtype=np.uint8)
            
            # Add some random defects
            num_defects = np.random.randint(1, 10)
            
            for _ in range(num_defects):
                # Random position and size
                x = np.random.randint(50, self.resolution[0] - 50)
                y = np.random.randint(50, self.resolution[1] - 50)
                size = np.random.randint(10, 100)
                
                # Draw a dark spot (simulating a defect)
                cv2.circle(frame, (x, y), size, (0, 0, 0), -1)
                
                # Add some noise with proper boundary checking
                x_start = max(0, x - size)
                x_end = min(self.resolution[0], x + size)
                y_start = max(0, y - size)
                y_end = min(self.resolution[1], y + size)
                
                # Get ROI dimensions
                roi_height = y_end - y_start
                roi_width = x_end - x_start
                
                # Generate noise matching ROI size
                noise = np.random.randint(-20, 20, (roi_height, roi_width, 3), dtype=np.int16)
                roi = frame[y_start:y_end, x_start:x_end]
                
                # Add noise to ROI
                noisy_roi = cv2.add(roi.astype(np.int16), noise, dtype=cv2.CV_16S)
                frame[y_start:y_end, x_start:x_end] = cv2.convertScaleAbs(noisy_roi)
            
            # Encode frame as base64 JPEG for transmission
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            return jpg_as_text
        except Exception as e:
            logger.error(f"❌ Error generating simulated frame: {e}")
            return ""
    
    async def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame for defect detection"""
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours (potential defects)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area to identify significant defects
            defects = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_defect_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate defect characteristics
                    aspect_ratio = float(w) / h
                    extent = float(area) / (w * h)
                    
                    defect_info = {
                        "position": {"x": x, "y": y},
                        "size": {"width": w, "height": h},
                        "area": area,
                        "aspect_ratio": aspect_ratio,
                        "extent": extent,
                        "confidence": min(1.0, area / 1000.0)  # Simple confidence heuristic
                    }
                    
                    defects.append(defect_info)
            
            # Encode frame for transmission
            frame_data = await self._encode_frame(frame)
            
            processed_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "frame_id": self.frame_count,
                "camera_id": "MIK1_Camera_01",
                "frame_data": frame_data,
                "defects_detected": len(defects),
                "defects": defects,
                "image_stats": {
                    "mean_brightness": float(np.mean(gray)),
                    "std_brightness": float(np.std(gray))
                }
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error processing frame: {e}")
            # Return basic frame data on error
            frame_data = await self._encode_frame(frame)
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "frame_id": self.frame_count,
                "camera_id": "MIK1_Camera_01",
                "frame_data": frame_data,
                "defects_detected": 0,
                "defects": [],
                "error": str(e)
            }
    
    async def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG for transmission"""
        try:
            # Resize frame if too large
            height, width = frame.shape[:2]
            if width > 1920 or height > 1080:
                scale = min(1920/width, 1080/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            return jpg_as_text
        except Exception as e:
            logger.error(f"❌ Error encoding frame: {e}")
            return ""
    
    async def stop_streaming(self):
        """Stop the camera stream"""
        self.running = False
        logger.info("⏹️ Stopping MIK-1 camera stream...")
    
    async def disconnect(self):
        """Disconnect from the camera"""
        await self.stop_streaming()
        
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info("✅ Disconnected from MIK-1 camera")
    
    def set_defect_parameters(self, min_area: int = 100, threshold: float = 0.7):
        """Set defect detection parameters"""
        self.min_defect_area = min_area
        self.defect_threshold = threshold
        logger.info(f"⚙️ Defect parameters updated: min_area={min_area}, threshold={threshold}")


class MIK1Simulator:
    """Simulator for MIK-1 camera for testing purposes"""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.frame_count = 0
        self.base_image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    async def generate_frame(self) -> np.ndarray:
        """Generate a simulated frame with random defects"""
        # Copy base image
        frame = self.base_image.copy()
        
        # Add some random defects
        num_defects = np.random.randint(1, 10)
        
        for _ in range(num_defects):
            # Random position and size
            x = np.random.randint(50, self.width - 50)
            y = np.random.randint(50, self.height - 50)
            size = np.random.randint(10, 100)
            
            # Draw a dark spot (simulating a defect)
            cv2.circle(frame, (x, y), size, (0, 0, 0), -1)
            
            # Add some noise with proper boundary checking
            x_start = max(0, x - size)
            x_end = min(self.width, x + size)
            y_start = max(0, y - size)
            y_end = min(self.height, y + size)
            
            # Get ROI dimensions
            roi_height = y_end - y_start
            roi_width = x_end - x_start
            
            # Generate noise matching ROI size
            noise = np.random.randint(-20, 20, (roi_height, roi_width, 3), dtype=np.int16)
            roi = frame[y_start:y_end, x_start:x_end]
            
            # Add noise to ROI
            noisy_roi = cv2.add(roi.astype(np.int16), noise, dtype=cv2.CV_16S)
            frame[y_start:y_end, x_start:x_end] = cv2.convertScaleAbs(noisy_roi)
        
        self.frame_count += 1
        return frame


async def main_example():
    """Example usage of MIK-1 camera stream"""
    
    async def frame_callback(data):
        """Callback function to handle incoming frame data"""
        print(f"📡 Frame {data.get('frame_id', 0)} received with {data.get('defects_detected', 0)} defects")
        
        # Print first defect if any
        if data.get('defects'):
            defect = data['defects'][0]
            print(f"🔍 Defect detected: Area={defect['area']:.1f}, Confidence={defect['confidence']:.2f}")
    
    # Create camera stream (using simulator for demo)
    camera = MIK1CameraStream(camera_source="0", callback=frame_callback)
    
    try:
        # For actual implementation, connect to real camera
        success = await camera.connect()
        
        if success:
            print("✅ Camera initialized successfully")
            
            # Start streaming
            streaming_task = asyncio.create_task(camera.start_streaming(process_frames=True))
            
            # Let it run for 10 seconds
            await asyncio.sleep(10)
            
            # Stop streaming
            await camera.stop_streaming()
            streaming_task.cancel()
            
        else:
            print("❌ Camera initialization failed")
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        await camera.stop_streaming()
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        await camera.stop_streaming()
    finally:
        await camera.disconnect()


if __name__ == "__main__":
    asyncio.run(main_example())