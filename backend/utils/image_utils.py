# from PIL import Image
# import numpy as np

# def resize_image(image, size=(256, 256)):
#     """Resize the image to the specified size."""
#     return image.resize(size)

# def normalize_image(image):
#     """Normalize the image pixel values to the range [-1, 1]."""
#     image = np.array(image) / 255.0  # Scale to [0, 1]
#     image = (image - 0.5) / 0.5      # Scale to [-1, 1]
#     return image

# def denormalize_image(image):
#     """Denormalize the image pixel values to the range [0, 255]."""
#     image = (image * 0.5) + 0.5  # Scale to [0, 1]
#     image = image * 255.0        # Scale to [0, 255]
#     return image.astype(np.uint8)

import base64
import io
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def preprocess(self, image):
        """Preprocess image for model input."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image.resize((256, 256))
    
    def to_tensor(self, image):
        """Convert PIL image to tensor."""
        return self.transform(image).unsqueeze(0)
    
    def tensor_to_image(self, tensor):
        """Convert tensor to PIL image."""
        tensor = tensor.squeeze(0).cpu()
        tensor = ((tensor * 0.5 + 0.5) * 255).clamp(0, 255).byte()
        return Image.fromarray(tensor.permute(1, 2, 0).numpy())
    
    def to_base64(self, image):
        """Convert PIL image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()