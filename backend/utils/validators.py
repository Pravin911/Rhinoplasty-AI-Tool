from PIL import Image

def validate_image(file):
    """Validate uploaded image file."""
    try:
        image = Image.open(file)
        
        # Check image mode
        if image.mode not in ['RGB', 'RGBA']:
            raise ValueError("Invalid image format. Must be RGB or RGBA")
            
        # Check image size
        if image.size[0] < 64 or image.size[1] < 64:
            raise ValueError("Image too small. Minimum size is 64x64 pixels")
            
        if image.size[0] > 4096 or image.size[1] > 4096:
            raise ValueError("Image too large. Maximum size is 4096x4096 pixels")
        
        return image
        
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

def validate_style(style, available_styles):
    """Validate requested nose style."""
    if style not in available_styles:
        raise ValueError(f"Invalid style. Must be one of: {', '.join(available_styles)}")