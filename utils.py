import os
from PIL import ImageOps

def detect_device(torch):
    """Return best device string given torch availability."""
    try:
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def save_pil_image(image, path="generated_image.png"):
    #  Save a PIL image to disk.
    image.save(path)
    return os.path.abspath(path)

def make_thumbnail(pil_image, size=(320, 240)):
    # maintain aspect ratio
    img = ImageOps.contain(pil_image, size)
    return img
