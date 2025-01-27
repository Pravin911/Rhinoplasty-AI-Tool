import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

def reconstruct_3d_face(image):
    """Reconstruct a 3D face using MediaPipe, with improved error handling and visualization."""
    # Ensure input is NumPy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Convert to RGB if needed
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Process the image
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        refine_landmarks=True, 
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_np)

        # Handle no face detection gracefully
        if not results.multi_face_landmarks:
            print("No face detected. Returning original image.")
            return Image.fromarray(image_np)

        # Annotate image with landmarks
        annotated_image = image_np.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        # Convert back to PIL Image
        return Image.fromarray(annotated_image)