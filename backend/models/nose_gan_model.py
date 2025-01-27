# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import numpy as np
# from PIL import Image

# class NoseGAN(nn.Module):
#     def __init__(self):
#         super(NoseGAN, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 4, 2, 1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, 4, 2, 1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2),
#         )
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 3, 4, 2, 1),
#             nn.Tanh()
#         )
        
#         # Style embedding
#         self.style_embedding = nn.Embedding(num_embeddings=5, embedding_dim=256)
        
#     def forward(self, x, style_idx):
#         # Encode
#         features = self.encoder(x)
        
#         # Apply style
#         style = self.style_embedding(style_idx)
#         style = style.view(style.size(0), -1, 1, 1)
#         features = features * style
        
#         # Decode
#         return self.decoder(features)

# class NoseReshapingModel:
#     def __init__(self, model_path=None):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = NoseGAN().to(self.device)
#         if model_path:
#             self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
        
#         self.transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
        
#         # Define preset nose styles
#         self.nose_styles = {
#             "natural": 0,
#             "refined": 1,
#             "upturned": 2,
#             "straight": 3,
#             "reduced": 4
#         }
    
#     def process_image(self, image, style_name):
#         """Process an image with the selected nose style."""
#         # Convert PIL Image to tensor
#         img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
#         # Get style index
#         style_idx = torch.tensor([self.nose_styles[style_name]]).to(self.device)
        
#         # Generate output
#         with torch.no_grad():
#             output = self.model(img_tensor, style_idx)
        
#         # Convert back to PIL Image
#         output = output.squeeze(0).cpu()
#         output = transforms.ToPILImage()(output * 0.5 + 0.5)
#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from PIL import Image
import cv2

class NoseGAN(nn.Module):
    def __init__(self):
        super(NoseGAN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
        
        # Style embedding
        self.style_embedding = nn.Embedding(num_embeddings=5, embedding_dim=256)
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Define nose styles
        self.nose_styles = {
            "natural": {"bridge_height": 1.0, "tip_projection": 1.0},
            "refined": {"bridge_height": 1.1, "tip_projection": 1.2},
            "upturned": {"bridge_height": 1.0, "tip_projection": 1.1},
            "straight": {"bridge_height": 1.2, "tip_projection": 1.0},
            "reduced": {"bridge_height": 0.9, "tip_projection": 0.9}
        }
        
    def forward(self, x, style_idx):
        # Encode
        features = self.encoder(x)
        
        # Apply style
        style = self.style_embedding(style_idx)
        style = style.view(style.size(0), -1, 1, 1)
        features = features * style
        
        # Decode
        return self.decoder(features)

    def detect_orientation(self, image):
        """Detect if image is front-facing or side profile."""
        results = self.mp_face_mesh.process(image)
        if not results.multi_face_landmarks:
            return "unknown"
            
        landmarks = results.multi_face_landmarks[0].landmark
        nose_tip = landmarks[4]
        left_ear = landmarks[234]
        right_ear = landmarks[454]
        
        ear_distance = abs(left_ear.x - right_ear.x)
        nose_to_ear_left = abs(nose_tip.x - left_ear.x)
        nose_to_ear_right = abs(nose_tip.x - right_ear.x)
        
        if abs(nose_to_ear_left - nose_to_ear_right) > ear_distance * 0.4:
            return "side"
        return "front"

    def get_nose_landmarks(self, image):
        """Extract nose landmarks."""
        results = self.mp_face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None
            
        nose_indices = [5, 4, 195, 197, 6, 168, 197, 195]
        landmarks = results.multi_face_landmarks[0].landmark
        
        h, w = image.shape[:2]
        return [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) 
                for idx in nose_indices]