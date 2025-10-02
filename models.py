"""
Model wrappers demonstrating:
- Encapsulation (protected attributes like _name, _pipeline)
- Polymorphism (BaseModel.run is overridden)
- Lazy loading: heavy HF models are loaded only when load() is called
"""

from abc import ABC, abstractmethod
from transformers import pipeline as hf_pipeline
try:
    from diffusers import StableDiffusionPipeline
except Exception:
    StableDiffusionPipeline = None

import torch
from utils import save_pil_image

# Abstract Base Model Class

class BaseModel(ABC):
    
    """
    Abstract base class for model wrappers.
    
    Features:
    - Encapsulation: protected attributes (_name, _pipeline, etc.)
    - Lazy loading: models are not loaded until `load()` is called
    - Polymorphism: subclasses must implement `load` and `run`
    """
    
    def __init__(self, name: str, category: str, description: str):
        self._name = name                 # protected attribute (encapsulation)
        self._category = category
        self._description = description
        self._pipeline = None             # lazy-loaded
        self._is_loaded = False

    def get_info(self):
        """Return metadata about the model"""
        return {
            "name": self._name,
            "category": self._category,
            "description": self._description,
            "loaded": self._is_loaded
        }

    @abstractmethod
    def load(self, device="cpu"):
        """
        Abstract method: must be implemented by subclasses.
        Responsible for loading the actual model/pipeline onto the given device
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, input_data):
        """Run the model on input_data. Must be overridden.
            Defines how the model processes `input_data`.
        """
        raise NotImplementedError


# Text-to-Image Model
class TextToImageModel(BaseModel):
    """Uses diffusers StableDiffusionPipeline. Lazy-loads on demand."""
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        super().__init__(model_name, "Text-to-Image", "Stable Diffusion text->image generator.")

    def load(self, device="cpu"):
        """Load the Stable Diffusion pipeline onto the given device."""
        if self._is_loaded:
            return
        if StableDiffusionPipeline is None:
            raise RuntimeError("diffusers not installed or unavailable. Run: pip install diffusers accelerate safetensors")
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        # from_pretrained will download the model the first time
        self._pipeline = StableDiffusionPipeline.from_pretrained(self._name, torch_dtype=dtype)
        # move to device
        self._pipeline = self._pipeline.to(device)
        self._is_loaded = True

    def run(self, prompt: str):
        """
        Generate an image from a text prompt.
        Automatically loads the model if not already loaded.
        """
        if not self._is_loaded:
            # default to cpu; caller can call load() with preferred device
            self.load(device="cpu")
        # pipeline returns a PipelineOutput containing .images
        output = self._pipeline(prompt)
        image = output.images[0]
        path = save_pil_image(image, "generated_image.png")
        return {"type": "image", "path": path, "pil_image": image}

# Image Classification Model
class ImageClassificationModel(BaseModel):
    """
    Image classification using HuggingFace transformers pipeline.
    Uses transformers pipeline for image-classification
    """
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__(model_name, "Image Classification", "Classify images using ViT.")

    def load(self, device="cpu"):
        if self._is_loaded:
            return
        # transformers pipelines handle device automatically but we keep this simple
        # device argument: -1 = cpu, 0 = first GPU. We'll set to -1 for safety.
        self._pipeline = hf_pipeline("image-classification", model=self._name, device=-1)
        self._is_loaded = True

    def run(self, image):  # expects a PIL image or path
        """
        Classify a given image.
        Accepts a PIL.Image object or a file path.
        """
        if not self._is_loaded:
            self.load()
        results = self._pipeline(image)
        return {"type": "classifications", "results": results}
