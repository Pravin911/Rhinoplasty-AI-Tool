import cv2
import numpy as np
import mediapipe as mp
from PIL import Image

class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def detect_face_orientation(self, image):
        """Detect if the image is front-facing or side profile."""
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        if image_np.shape[2] == 4:  # If RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
        # Process with MediaPipe
        results = self.face_mesh.process(image_np)
        
        if not results.multi_face_landmarks:
            return "unknown"
            
        # Get landmarks
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate face orientation based on landmark positions
        # Get nose tip and ears positions
        nose_tip = landmarks[4]
        left_ear = landmarks[234]
        right_ear = landmarks[454]
        
        # Calculate distances
        ear_distance = abs(left_ear.x - right_ear.x)
        nose_to_ear_left = abs(nose_tip.x - left_ear.x)
        nose_to_ear_right = abs(nose_tip.x - right_ear.x)
        
        # If one ear is much closer to the nose than the other, it's likely a side profile
        if abs(nose_to_ear_left - nose_to_ear_right) > ear_distance * 0.4:
            return "side"
        return "front"
        
    def get_nose_landmarks(self, image):
        """Extract nose-specific landmarks."""
        image_np = np.array(image)
        if image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
        results = self.face_mesh.process(image_np)
        
        if not results.multi_face_landmarks:
            return None
            
        # Nose-specific landmark indices
        nose_indices = [5, 4, 195, 197, 6, 168, 197, 195, 5, 4, 45, 275, 440, 279, 456, 420]
        landmarks = results.multi_face_landmarks[0].landmark
        
        nose_landmarks = []
        h, w = image_np.shape[:2]
        for idx in nose_indices:
            point = landmarks[idx]
            nose_landmarks.append((int(point.x * w), int(point.y * h)))
            
        return nose_landmarks