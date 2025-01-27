# Version1: 2D and 3D outputs

# import base64
# import io
# import traceback
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# import numpy as np
# import cv2
# import mediapipe as mp

# app = Flask(__name__)
# CORS(app)

# class RhinoplastyModel:
#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             static_image_mode=True, 
#             max_num_faces=1, 
#             refine_landmarks=True
#         )

#     def generate_2d_output(self, image):
#         # Simple 2D processing - just return the input image
#         return image

#     def generate_3d_output(self, image):
#         # Convert image to numpy array
#         image_np = np.array(image)
        
#         # Convert to RGB if needed
#         if image_np.shape[2] == 4:  # If RGBA
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
#         # Process with MediaPipe
#         results = self.face_mesh.process(image_np)
        
#         if not results.multi_face_landmarks:
#             # If no face detected, return original image
#             return image
        
#         # Annotate image with landmarks
#         annotated_image = image_np.copy()
#         for face_landmarks in results.multi_face_landmarks:
#             for landmark in face_landmarks.landmark:
#                 h, w, _ = annotated_image.shape
#                 cx, cy = int(landmark.x * w), int(landmark.y * h)
#                 cv2.circle(annotated_image, (cx, cy), 2, (0, 255, 0), -1)
        
#         return Image.fromarray(annotated_image)

# @app.route("/upload", methods=["POST"])
# def upload_image():
#     try:
#         # Check if file is present
#         if "file" not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
        
#         file = request.files["file"]
        
#         # Read and process image
#         image = Image.open(io.BytesIO(file.read()))
        
#         # Create model instance
#         model = RhinoplastyModel()
        
#         # Generate outputs
#         output_2d = model.generate_2d_output(image)
#         output_3d = model.generate_3d_output(image)
        
#         # Convert to base64
#         def image_to_base64(img):
#             buffered = io.BytesIO()
#             img.save(buffered, format="JPEG")
#             return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
#         return jsonify({
#             "output_2d": image_to_base64(output_2d),
#             "output_3d": image_to_base64(output_3d)
#         })
    
#     except Exception as e:
#         # Log full error trace
#         print(traceback.format_exc())
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

# Version2: 2D and 3D outputs:

# import base64
# import io
# import traceback
# import os
# import sys
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from PIL import Image
# import numpy as np
# import cv2
# import mediapipe as mp
# import torch
# import torchvision.transforms as transforms

# app = Flask(__name__)
# CORS(app)

# class NoseReshapingModel:
#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             static_image_mode=True,
#             max_num_faces=1,
#             refine_landmarks=True
#         )
        
#         # Define preset nose styles
#         self.nose_styles = {
#             "natural": 0,
#             "refined": 1,
#             "upturned": 2,
#             "straight": 3,
#             "reduced": 4
#         }
        
#     def process_image(self, image, style_name="natural"):
#         """Process image with selected nose style."""
#         # For now, return the original image with face mesh
#         image_np = np.array(image)
#         if len(image_np.shape) == 2:  # Convert grayscale to RGB
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
#         elif image_np.shape[2] == 4:  # Convert RGBA to RGB
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
#         results = self.face_mesh.process(image_np)
        
#         if not results.multi_face_landmarks:
#             return image
            
#         # Draw landmarks focused on nose area
#         annotated_image = image_np.copy()
#         nose_landmarks = [4, 5, 6, 168, 195, 197]  # Nose-specific landmarks
        
#         for face_landmarks in results.multi_face_landmarks:
#             for idx in nose_landmarks:
#                 pt = face_landmarks.landmark[idx]
#                 x, y = int(pt.x * image_np.shape[1]), int(pt.y * image_np.shape[0])
#                 cv2.circle(annotated_image, (x, y), 2, (0, 255, 0), -1)
                
#         return Image.fromarray(annotated_image)

# class FaceAnalyzer:
#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             static_image_mode=True,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5
#         )
        
#     def detect_face_orientation(self, image):
#         """Detect if image is front-facing or side profile."""
#         image_np = np.array(image)
#         if len(image_np.shape) == 2:
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
#         elif image_np.shape[2] == 4:
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
#         results = self.face_mesh.process(image_np)
        
#         if not results.multi_face_landmarks:
#             return "unknown"
            
#         landmarks = results.multi_face_landmarks[0].landmark
        
#         # Get nose tip and ears positions
#         nose_tip = landmarks[4]
#         left_ear = landmarks[234]
#         right_ear = landmarks[454]
        
#         # Calculate distances
#         ear_distance = abs(left_ear.x - right_ear.x)
#         nose_to_ear_left = abs(nose_tip.x - left_ear.x)
#         nose_to_ear_right = abs(nose_tip.x - right_ear.x)
        
#         if abs(nose_to_ear_left - nose_to_ear_right) > ear_distance * 0.4:
#             return "side"
#         return "front"
        
#     def get_nose_landmarks(self, image):
#         """Extract nose-specific landmarks."""
#         image_np = np.array(image)
#         if len(image_np.shape) == 2:
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
#         elif image_np.shape[2] == 4:
#             image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
#         results = self.face_mesh.process(image_np)
        
#         if not results.multi_face_landmarks:
#             return None
            
#         nose_indices = [5, 4, 195, 197, 6, 168, 197, 195, 5, 4, 45, 275]
#         landmarks = results.multi_face_landmarks[0].landmark
        
#         nose_points = []
#         h, w = image_np.shape[:2]
#         for idx in nose_indices:
#             point = landmarks[idx]
#             nose_points.append([int(point.x * w), int(point.y * h)])
            
#         return nose_points

# # Initialize models
# nose_model = NoseReshapingModel()
# face_analyzer = FaceAnalyzer()

# @app.route("/api/nose-styles", methods=["GET"])
# def get_nose_styles():
#     """Return available nose style options."""
#     return jsonify({
#         "styles": list(nose_model.nose_styles.keys())
#     })

# @app.route("/api/analyze-face", methods=["POST"])
# def analyze_face():
#     """Analyze face orientation and features."""
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
            
#         file = request.files["file"]
#         image = Image.open(io.BytesIO(file.read()))
        
#         # Detect orientation
#         orientation = face_analyzer.detect_face_orientation(image)
        
#         # Get nose landmarks
#         landmarks = face_analyzer.get_nose_landmarks(image)
        
#         return jsonify({
#             "orientation": orientation,
#             "landmarks": landmarks
#         })
#     except Exception as e:
#         print(traceback.format_exc())
#         return jsonify({"error": str(e)}), 500

# @app.route("/api/generate", methods=["POST"])
# def generate_nose():
#     """Generate nose reshaping visualization."""
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
            
#         file = request.files["file"]
#         style = request.form.get("style", "natural")
        
#         # Load and process image
#         image = Image.open(io.BytesIO(file.read()))
        
#         # Generate 2D output with selected style
#         output_2d = nose_model.process_image(image, style)
        
#         # Generate 3D visualization
#         output_3d = nose_model.process_image(image, style)  # Using same processing for now
        
#         # Convert outputs to base64
#         def image_to_base64(img):
#             buffered = io.BytesIO()
#             img.save(buffered, format="JPEG")
#             return base64.b64encode(buffered.getvalue()).decode()
            
#         return jsonify({
#             "output_2d": image_to_base64(output_2d),
#             "output_3d": image_to_base64(output_3d)
#         })
        
#     except Exception as e:
#         print(traceback.format_exc())
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     try:
#         print("Starting Flask application...")
#         app.run(host="0.0.0.0", port=5000, debug=True)
#     except Exception as e:
#         print(f"Error starting the application: {str(e)}")
#         traceback.print_exc()

# Version 3: 2D and 3D outputs:

# backend/app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# import mediapipe as mp
# import base64
# import io
# from PIL import Image

# app = Flask(__name__)
# CORS(app)

# class FaceAnalyzer:
#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
#             static_image_mode=True,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5
#         )
        
#         self.styles = {
#             "natural": {
#                 "description": "Natural look with minimal changes",
#                 "params": {"bridge_height": 1.0, "tip_projection": 1.0},
#                 "color": (0, 255, 0)
#             },
#             "refined": {
#                 "description": "Refined bridge with balanced tip",
#                 "params": {"bridge_height": 1.2, "tip_projection": 1.1},
#                 "color": (255, 0, 0)
#             },
#             "upturned": {
#                 "description": "Slightly upturned tip",
#                 "params": {"bridge_height": 1.0, "tip_projection": 1.2},
#                 "color": (0, 0, 255)
#             }
#         }
        
#     def analyze_face(self, image):
#         """Analyze face orientation and features."""
#         image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         results = self.mp_face_mesh.process(image_np)
        
#         if not results.multi_face_landmarks:
#             return None, "No face detected"
            
#         landmarks = results.multi_face_landmarks[0]
        
#         # Determine orientation
#         orientation = self._get_orientation(landmarks)
        
#         # Get nose landmarks
#         nose_landmarks = self._get_nose_landmarks(landmarks, image_np.shape)
        
#         return {
#             "orientation": orientation,
#             "landmarks": nose_landmarks
#         }, None
        
#     def generate_visualization(self, image, style):
#         """Generate nose reshaping visualization."""
#         image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         results = self.mp_face_mesh.process(image_np)
        
#         if not results.multi_face_landmarks:
#             return None, None, "No face detected"
            
#         # Create 2D visualization
#         output_2d = image_np.copy()
#         landmarks = results.multi_face_landmarks[0]
#         nose_landmarks = self._get_nose_landmarks(landmarks, image_np.shape)
        
#         # Draw nose outline
#         style_color = self.styles[style]["color"]
#         cv2.polylines(output_2d, [np.array(nose_landmarks)], True, style_color, 2)
        
#         # Add style indicators
#         for point in nose_landmarks:
#             cv2.circle(output_2d, point, 2, style_color, -1)
            
#         # Create 3D visualization with depth
#         output_3d = image_np.copy()
#         self._draw_3d_indicators(output_3d, nose_landmarks, self.styles[style])
        
#         return output_2d, output_3d, None
        
#     def _get_orientation(self, landmarks):
#         """Determine face orientation."""
#         nose_tip = landmarks.landmark[4]
#         left_ear = landmarks.landmark[234]
#         right_ear = landmarks.landmark[454]
        
#         ear_distance = abs(left_ear.x - right_ear.x)
#         nose_to_ear_left = abs(nose_tip.x - left_ear.x)
#         nose_to_ear_right = abs(nose_tip.x - right_ear.x)
        
#         if abs(nose_to_ear_left - nose_to_ear_right) > ear_distance * 0.4:
#             return "side"
#         return "front"
        
#     def _get_nose_landmarks(self, landmarks, image_shape):
#         """Extract nose landmarks."""
#         h, w = image_shape[:2]
#         nose_indices = [4, 5, 195, 197, 6]  # Key nose points
        
#         return [(int(landmarks.landmark[idx].x * w), 
#                  int(landmarks.landmark[idx].y * h)) 
#                 for idx in nose_indices]
                
#     def _draw_3d_indicators(self, image, landmarks, style):
#         """Draw 3D visualization indicators."""
#         color = style["color"]
#         params = style["params"]
        
#         # Draw depth lines
#         for i, point in enumerate(landmarks):
#             depth = int(20 * params["bridge_height"]) if i < 2 else int(15 * params["tip_projection"])
#             cv2.line(image, point, (point[0], point[1] - depth), color, 2)
#             cv2.circle(image, (point[0], point[1] - depth), 3, color, -1)
            
#         # Connect points for 3D effect
#         for i in range(len(landmarks) - 1):
#             pt1 = landmarks[i]
#             pt2 = landmarks[i + 1]
#             depth1 = int(20 * params["bridge_height"]) if i < 2 else int(15 * params["tip_projection"])
#             depth2 = int(20 * params["bridge_height"]) if i + 1 < 2 else int(15 * params["tip_projection"])
            
#             cv2.line(image, 
#                     (pt1[0], pt1[1] - depth1),
#                     (pt2[0], pt2[1] - depth2),
#                     color, 2)

# face_analyzer = FaceAnalyzer()

# def image_to_base64(image_array):
#     """Convert image array to base64 string."""
#     _, buffer = cv2.imencode('.jpg', image_array)
#     return base64.b64encode(buffer).decode('utf-8')

# @app.route('/api/nose-styles', methods=['GET'])
# def get_nose_styles():
#     """Return available nose styles."""
#     return jsonify({
#         "styles": list(face_analyzer.styles.keys()),
#         "descriptions": {style: info["description"] 
#                         for style, info in face_analyzer.styles.items()}
#     })

# @app.route('/api/analyze-face', methods=['POST'])
# def analyze_face():
#     """Analyze uploaded face image."""
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
            
#         file = request.files['file']
#         image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
#         result, error = face_analyzer.analyze_face(image)
#         if error:
#             return jsonify({"error": error}), 400
            
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/api/generate', methods=['POST'])
# def generate():
#     """Generate nose reshaping visualization."""
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
            
#         file = request.files['file']
#         style = request.form.get('style', 'natural')
        
#         if style not in face_analyzer.styles:
#             return jsonify({"error": "Invalid style"}), 400
            
#         image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
#         output_2d, output_3d, error = face_analyzer.generate_visualization(image, style)
#         if error:
#             return jsonify({"error": error}), 400
            
#         return jsonify({
#             "output_2d": image_to_base64(output_2d),
#             "output_3d": image_to_base64(output_3d),
#             "analysis": {
#                 "style_parameters": face_analyzer.styles[style]["params"]
#             }
#         })
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     print("Starting server...")
#     print("Available styles:", list(face_analyzer.styles.keys()))
#     app.run(host='0.0.0.0', port=5000, debug=True)

# Version 4 

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import cv2
# import numpy as np
# import mediapipe as mp
# import base64
# import io
# from PIL import Image

# app = Flask(__name__)
# CORS(app)

# class FaceAnalyzer:
#     def __init__(self):
#         self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
#             static_image_mode=True,
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.5
#         )

#         self.styles = {
#             "natural": {
#                 "description": "Natural look with minimal changes",
#                 "params": {"bridge_height": 1.0, "tip_projection": 1.0},
#                 "color": (0, 255, 0)
#             },
#             "refined": {
#                 "description": "Refined bridge with balanced tip",
#                 "params": {"bridge_height": 1.2, "tip_projection": 1.1},
#                 "color": (255, 0, 0)
#             },
#             "upturned": {
#                 "description": "Slightly upturned tip",
#                 "params": {"bridge_height": 1.0, "tip_projection": 1.2},
#                 "color": (0, 0, 255)
#             }
#         }

#     def analyze_face(self, image):
#         """Analyze face orientation and features."""
#         image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         results = self.mp_face_mesh.process(image_np)

#         if not results.multi_face_landmarks:
#             return None, "No face detected"

#         landmarks = results.multi_face_landmarks[0]
#         orientation = self._get_orientation(landmarks)
#         nose_landmarks = self._get_nose_landmarks(landmarks, image_np.shape)

#         return {
#             "orientation": orientation,
#             "landmarks": nose_landmarks
#         }, None

#     def generate_visualization(self, image, style):
#         """Generate nose reshaping visualization."""
#         image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         results = self.mp_face_mesh.process(image_np)

#         if not results.multi_face_landmarks:
#             return None, None, "No face detected"

#         output_2d = image_np.copy()
#         landmarks = results.multi_face_landmarks[0]
#         nose_landmarks = self._get_nose_landmarks(landmarks, image_np.shape)
#         style_color = self.styles[style]["color"]

#         cv2.polylines(output_2d, [np.array(nose_landmarks)], True, style_color, 2)
#         for point in nose_landmarks:
#             cv2.circle(output_2d, point, 2, style_color, -1)

#         output_3d = image_np.copy()
#         self._draw_3d_indicators(output_3d, nose_landmarks, self.styles[style])

#         return output_2d, output_3d, None

#     def _get_orientation(self, landmarks):
#         """Determine face orientation."""
#         nose_tip = landmarks.landmark[4]
#         left_ear = landmarks.landmark[234]
#         right_ear = landmarks.landmark[454]

#         ear_distance = abs(left_ear.x - right_ear.x)
#         nose_to_ear_left = abs(nose_tip.x - left_ear.x)
#         nose_to_ear_right = abs(nose_tip.x - right_ear.x)

#         if abs(nose_to_ear_left - nose_to_ear_right) > ear_distance * 0.4:
#             return "side"
#         return "front"

#     def _get_nose_landmarks(self, landmarks, image_shape):
#         """Extract nose landmarks."""
#         h, w = image_shape[:2]
#         nose_indices = [4, 5, 195, 197, 6]

#         return [(int(landmarks.landmark[idx].x * w), 
#                  int(landmarks.landmark[idx].y * h)) 
#                 for idx in nose_indices]

#     def _draw_3d_indicators(self, image, landmarks, style):
#         """Draw 3D visualization indicators."""
#         color = style["color"]
#         params = style["params"]

#         for i, point in enumerate(landmarks):
#             depth = int(20 * params["bridge_height"]) if i < 2 else int(15 * params["tip_projection"])
#             cv2.line(image, point, (point[0], point[1] - depth), color, 2)
#             cv2.circle(image, (point[0], point[1] - depth), 3, color, -1)

#         for i in range(len(landmarks) - 1):
#             pt1 = landmarks[i]
#             pt2 = landmarks[i + 1]
#             depth1 = int(20 * params["bridge_height"]) if i < 2 else int(15 * params["tip_projection"])
#             depth2 = int(20 * params["bridge_height"]) if i + 1 < 2 else int(15 * params["tip_projection"])

#             cv2.line(image, 
#                     (pt1[0], pt1[1] - depth1),
#                     (pt2[0], pt2[1] - depth2),
#                     color, 2)

# face_analyzer = FaceAnalyzer()

# def image_to_base64(image_array):
#     """Convert image array to base64 string."""
#     _, buffer = cv2.imencode('.jpg', image_array)
#     return base64.b64encode(buffer).decode('utf-8')

# @app.route('/api/nose-styles', methods=['GET'])
# def get_nose_styles():
#     """Return available nose styles."""
#     return jsonify({
#         "styles": list(face_analyzer.styles.keys()),
#         "descriptions": {style: info["description"] \
#                         for style, info in face_analyzer.styles.items()}
#     })

# @app.route('/api/analyze-face', methods=['POST'])
# def analyze_face():
#     """Analyze uploaded face image."""
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400

#         file = request.files['file']
#         image = Image.open(io.BytesIO(file.read())).convert('RGB')

#         result, error = face_analyzer.analyze_face(image)
#         if error:
#             return jsonify({"error": error}), 400

#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/api/generate', methods=['POST'])
# def generate():
#     """Generate nose reshaping visualization."""
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400

#         file = request.files['file']
#         style = request.form.get('style', 'natural')

#         if style not in face_analyzer.styles:
#             return jsonify({"error": "Invalid style"}), 400

#         image = Image.open(io.BytesIO(file.read())).convert('RGB')

#         output_2d, output_3d, error = face_analyzer.generate_visualization(image, style)
#         if error:
#             return jsonify({"error": error}), 400

#         return jsonify({
#             "output_2d": image_to_base64(output_2d),
#             "output_3d": image_to_base64(output_3d),
#             "analysis": {
#                 "style_parameters": face_analyzer.styles[style]["params"]
#             }
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     print("Starting server...")
#     print("Available styles:", list(face_analyzer.styles.keys()))
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )

        self.styles = {
            "natural": {
                "description": "Natural look with minimal changes",
                "params": {
                    "bridge_height": 1.0, 
                    "tip_projection": 1.0, 
                    "nostril_width": 1.0,
                    "dorsal_hump": 0.0,
                    "alar_width": 1.0,
                    "tip_rotation": 0.0,
                },
                "color": (0, 255, 0)
            },
            "refined": {
                "description": "Refined bridge with balanced tip",
                "params": {
                    "bridge_height": 1.2, 
                    "tip_projection": 1.1, 
                    "nostril_width": 0.9,
                    "dorsal_hump": -0.1,
                    "alar_width": 0.9,
                    "tip_rotation": 0.05,
                },
                "color": (255, 0, 0)
            },
            "upturned": {
                "description": "Slightly upturned tip",
                "params": {
                    "bridge_height": 1.0, 
                    "tip_projection": 1.3, 
                    "nostril_width": 1.1,
                    "dorsal_hump": 0.0,
                    "alar_width": 1.0,
                    "tip_rotation": 0.2,
                },
                "color": (0, 0, 255)
            },
            "straight": {
                "description": "Even bridge with no pronounced tip",
                "params": {
                    "bridge_height": 1.1,
                    "tip_projection": 1.0,
                    "nostril_width": 1.0,
                    "dorsal_hump": 0.0,
                    "alar_width": 1.0,
                    "tip_rotation": 0.0,
                },
                "color": (255, 165, 0)
            },
            "snub": {
                "description": "Shorter upturned tip with reduced bridge",
                "params": {
                    "bridge_height": 0.8,
                    "tip_projection": 1.2,
                    "nostril_width": 1.0,
                    "dorsal_hump": 0.0,
                    "alar_width": 1.1,
                    "tip_rotation": 0.15,
                },
                "color": (255, 105, 180)
            },
            "aquiline": {
                "description": "Prominent bridge with refined tip",
                "params": {
                    "bridge_height": 1.4,
                    "tip_projection": 1.0,
                    "nostril_width": 0.9,
                    "dorsal_hump": 0.2,
                    "alar_width": 0.9,
                    "tip_rotation": 0.0,
                },
                "color": (128, 128, 0)
            },
            "button": {
                "description": "Small rounded tip with softer bridge",
                "params": {
                    "bridge_height": 0.9,
                    "tip_projection": 1.1,
                    "nostril_width": 1.1,
                    "dorsal_hump": 0.0,
                    "alar_width": 1.0,
                    "tip_rotation": -0.1,
                },
                "color": (255, 182, 193)
            },
            "sharp": {
                "description": "Sharp bridge with angular tip",
                "params": {
                    "bridge_height": 1.3,
                    "tip_projection": 1.2,
                    "nostril_width": 0.9,
                    "dorsal_hump": -0.1,
                    "alar_width": 0.9,
                    "tip_rotation": 0.1,
                },
                "color": (0, 0, 0)
            },
        }

    def analyze_face(self, image):
        """Analyze face orientation and features."""
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.mp_face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return None, "No face detected"

        landmarks = results.multi_face_landmarks[0]

        # Determine orientation
        orientation = self._get_orientation(landmarks)

        # Get nose landmarks
        nose_landmarks = self._get_nose_landmarks(landmarks, image_np.shape)

        return {
            "orientation": orientation,
            "landmarks": nose_landmarks
        }, None

    def generate_visualization(self, image, style):
        """Generate nose reshaping visualization."""
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.mp_face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return None, None, "No face detected"

        output_2d = image_np.copy()
        landmarks = results.multi_face_landmarks[0]
        nose_landmarks = self._get_nose_landmarks(landmarks, image_np.shape)

        style_params = self.styles[style]["params"]
        style_color = self.styles[style]["color"]

        # Draw reshaped nose outline based on style
        reshaped_nose = self._reshape_nose(nose_landmarks, style_params)
        cv2.polylines(output_2d, [np.array(reshaped_nose)], True, style_color, 2)

        for point in reshaped_nose:
            cv2.circle(output_2d, point, 2, style_color, -1)

        output_3d = image_np.copy()
        self._draw_3d_indicators(output_3d, reshaped_nose, style_params, style_color)

        return output_2d, output_3d, None

    def _get_orientation(self, landmarks):
        """Determine face orientation."""
        nose_tip = landmarks.landmark[4]
        left_ear = landmarks.landmark[234]
        right_ear = landmarks.landmark[454]

        ear_distance = abs(left_ear.x - right_ear.x)
        nose_to_ear_left = abs(nose_tip.x - left_ear.x)
        nose_to_ear_right = abs(nose_tip.x - right_ear.x)

        if abs(nose_to_ear_left - nose_to_ear_right) > ear_distance * 0.4:
            return "side"
        return "front"

    def _get_nose_landmarks(self, landmarks, image_shape):
        """Extract nose landmarks."""
        h, w = image_shape[:2]
        nose_indices = [4, 5, 195, 197, 6]  # Key nose points

        return [(int(landmarks.landmark[idx].x * w),
                 int(landmarks.landmark[idx].y * h))
                for idx in nose_indices]

    def _reshape_nose(self, nose_landmarks, params):
        """Apply reshaping parameters to nose landmarks."""
        reshaped = []
        for i, (x, y) in enumerate(nose_landmarks):
            if i < 2:  # Bridge points
                y -= int(params["bridge_height"] * 5)
            elif i >= 2:  # Tip and nostrils
                x += int((i - 2) * params["tip_projection"] * 3)
                y -= int(params["dorsal_hump"] * 5)  # Adjust for hump
            reshaped.append((x, y))
        return reshaped

    def _draw_3d_indicators(self, image, reshaped_nose, params, color):
        """Draw 3D visualization indicators with dynamic lighting."""
        for i, point in enumerate(reshaped_nose):
            depth = int(20 * params["bridge_height"]) if i < 2 else int(15 * params["tip_projection"])
            shadow = int(depth * 0.3)  # Add shadow effect
            cv2.line(image, point, (point[0], point[1] - depth), color, 2)
            cv2.circle(image, (point[0], point[1] - depth - shadow), 3, color, -1)

face_analyzer = FaceAnalyzer()

def image_to_base64(image_array):
    """Convert image array to base64 string."""
    _, buffer = cv2.imencode('.jpg', image_array)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/api/nose-styles', methods=['GET'])
def get_nose_styles():
    """Return available nose styles."""
    return jsonify({
        "styles": list(face_analyzer.styles.keys()),
        "descriptions": {style: info["description"]
                          for style, info in face_analyzer.styles.items()}
    })

@app.route('/api/analyze-face', methods=['POST'])
def analyze_face():
    """Analyze uploaded face image."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        result, error = face_analyzer.analyze_face(image)
        if error:
            return jsonify({"error": error}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate nose reshaping visualization with advanced tuning."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        style = request.form.get('style', 'natural')

        if style not in face_analyzer.styles:
            return jsonify({"error": "Invalid style"}), 400

        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        output_2d, output_3d, error = face_analyzer.generate_visualization(image, style)
        if error:
            return jsonify({"error": error}), 400

        return jsonify({
            "output_2d": image_to_base64(output_2d),
            "output_3d": image_to_base64(output_3d),
            "analysis": {
                "style_parameters": face_analyzer.styles[style]["params"]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting server...")
    print("Available styles:", list(face_analyzer.styles.keys()))
    app.run(host='0.0.0.0', port=5000, debug=True)



