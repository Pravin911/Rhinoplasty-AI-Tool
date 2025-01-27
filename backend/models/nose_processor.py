import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from PIL import Image
import cv2
from .nose_gan import NoseGAN
from utils.image_utils import ImageProcessor

class NoseProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = NoseGAN().to(self.device)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.image_processor = ImageProcessor()
        
        # Define nose styles with parameters
        self.nose_styles = {
            "natural": {
                "description": "Maintains natural nose shape with minimal enhancement",
                "params": {"bridge_height": 1.0, "tip_projection": 1.0}
            },
            "refined": {
                "description": "Subtle refinement for a more defined appearance",
                "params": {"bridge_height": 1.1, "tip_projection": 1.2}
            },
            "upturned": {
                "description": "Slightly lifted tip with maintained bridge",
                "params": {"bridge_height": 1.0, "tip_projection": 1.1}
            },
            "straight": {
                "description": "Enhanced bridge with natural tip",
                "params": {"bridge_height": 1.2, "tip_projection": 1.0}
            },
            "reduced": {
                "description": "Subtle reduction in overall prominence",
                "params": {"bridge_height": 0.9, "tip_projection": 0.9}
            }
        }
    
    def get_available_styles(self):
        """Return list of available styles."""
        return list(self.nose_styles.keys())
    
    def get_style_descriptions(self):
        """Return style descriptions."""
        return {style: info["description"] 
                for style, info in self.nose_styles.items()}
    
    def analyze_face(self, image):
        """Analyze face features and orientation."""
        try:
            # Convert to RGB numpy array
            image_np = np.array(image)
            if image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(image_np)
            
            if not results.multi_face_landmarks:
                return {"error": "No face detected in image"}
            
            # Get orientation
            orientation = self._detect_orientation(results.multi_face_landmarks[0])
            
            # Get landmarks
            landmarks = self._get_nose_landmarks(results.multi_face_landmarks[0], image_np)
            
            # Get compatible styles based on orientation
            compatible_styles = self._get_compatible_styles(orientation)
            
            return {
                "orientation": orientation,
                "landmarks": landmarks,
                "compatible_styles": compatible_styles
            }
            
        except Exception as e:
            return {"error": f"Face analysis failed: {str(e)}"}
    
    def process_image(self, image, style):
        """Process image with selected nose style."""
        try:
            # Analyze face
            analysis = self.analyze_face(image)
            if analysis.get("error"):
                return analysis
            
            # Convert image to tensor
            img_tensor = self.image_processor.to_tensor(image).to(self.device)
            
            # Get style parameters
            style_params = self.nose_styles[style]["params"]
            
            # Generate output
            with torch.no_grad():
                output = self.model(img_tensor, torch.tensor([self.get_available_styles().index(style)]).to(self.device))
            
            # Convert output to image
            output_2d = self.image_processor.tensor_to_image(output)
            
            # Generate 3D visualization
            output_3d = self._generate_3d_visualization(image, analysis["landmarks"], style_params)
            
            return {
                "output_2d": output_2d,
                "output_3d": output_3d,
                "analysis": {
                    "orientation": analysis["orientation"],
                    "style_parameters": style_params
                }
            }
            
        except Exception as e:
            return {"error": f"Image processing failed: {str(e)}"}
    
    def _detect_orientation(self, landmarks):
        """Detect face orientation."""
        nose_tip = landmarks.landmark[4]
        left_ear = landmarks.landmark[234]
        right_ear = landmarks.landmark[454]
        
        ear_distance = abs(left_ear.x - right_ear.x)
        nose_to_ear_left = abs(nose_tip.x - left_ear.x)
        nose_to_ear_right = abs(nose_tip.x - right_ear.x)
        
        if abs(nose_to_ear_left - nose_to_ear_right) > ear_distance * 0.4:
            return "side"
        return "front"
    
    def _get_nose_landmarks(self, landmarks, image):
        """Extract nose landmarks."""
        h, w = image.shape[:2]
        nose_indices = [5, 4, 195, 197, 6, 168, 197, 195]
        
        return [(int(landmarks.landmark[idx].x * w),
                 int(landmarks.landmark[idx].y * h))
                for idx in nose_indices]
    
    def _get_compatible_styles(self, orientation):
        """Get compatible styles based on orientation."""
        if orientation == "front":
            return self.get_available_styles()
        return ["natural", "refined", "straight"]  # Limited styles for side view
    
    def _generate_3d_visualization(self, image, landmarks, style_params):
        """Generate 3D visualization."""
        image_np = np.array(image)
        viz_image = image_np.copy()
        
        if landmarks:
            # Draw landmarks
            for point in landmarks:
                cv2.circle(viz_image, point, 2, (0, 255, 0), -1)
            
            # Draw connections
            for i in range(len(landmarks) - 1):
                cv2.line(viz_image, landmarks[i], landmarks[i + 1], (0, 255, 0), 1)
            
            # Add style parameter visualization
            height = int(style_params["bridge_height"] * 20)
            projection = int(style_params["tip_projection"] * 20)
            
            center_x = sum(x for x, _ in landmarks) // len(landmarks)
            center_y = sum(y for _, y in landmarks) // len(landmarks)
            
            cv2.line(viz_image, 
                    (center_x, center_y - height),
                    (center_x, center_y + height),
                    (255, 0, 0), 2)
            cv2.line(viz_image,
                    (center_x - projection, center_y),
                    (center_x + projection, center_y),
                    (0, 0, 255), 2)
        
        return Image.fromarray(viz_image)