import logging
import re
import ast
from typing import Union

from PIL import Image
import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

# Resolution mode presets
RESOLUTION_MODES = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

# Operation modes that determine output format
OPERATIONS = {
    "grounding": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "return_type": "detections"
    },
    "ocr": {
        "prompt": "<image>\nFree OCR.",
        "return_type": "text"
    },
    "describe": {
        "prompt": "<image>\nParse the figure.",
        "return_type": "text"
    }
}


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DeepSeekOCR(Model, SamplesMixin):
    """FiftyOne model for DeepSeek-OCR vision-language tasks.
    
    Supports three operation modes:
    - grounding: Document to markdown with bounding boxes (returns fo.Detections)
    - ocr: Free text extraction (returns str)
    - describe: Figure/chart description (returns str)
    
    Automatically selects optimal dtype based on hardware:
    - bfloat16 for CUDA devices with compute capability 8.0+ (Ampere and newer)
    - float16 for older CUDA devices
    - float32 for CPU/MPS devices
    
    Args:
        model_path: HuggingFace model ID or local path
        resolution_mode: One of "gundam", "base", "small", "large", "tiny" (default: "gundam")
        operation: Task type - "grounding", "ocr", "describe" (default: "grounding")
        custom_prompt: Custom prompt (overrides operation prompt)
        torch_dtype: Override automatic dtype selection
    """
    
    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        resolution_mode: str = "gundam",
        operation: str = "grounding",
        custom_prompt: str = None,
        torch_dtype: torch.dtype = None,
        **kwargs
    ):
        SamplesMixin.__init__(self) 
        self.model_path = model_path
        self._resolution_mode = resolution_mode
        self._operation = operation
        self._custom_prompt = custom_prompt
        
        # Validate resolution mode
        if resolution_mode not in RESOLUTION_MODES:
            raise ValueError(
                f"resolution_mode must be one of {list(RESOLUTION_MODES.keys())}, "
                f"got '{resolution_mode}'"
            )
        
        # Validate operation
        if operation not in OPERATIONS and custom_prompt is None:
            raise ValueError(
                f"operation must be one of {list(OPERATIONS.keys())}, "
                f"got '{operation}'"
            )
        
        # Device setup
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Dtype selection
        if torch_dtype is not None:
            self.dtype = torch_dtype
        elif self.device == "cuda":
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
            logger.info(f"Using {self.dtype} dtype (compute capability {capability[0]}.{capability[1]})")
        else:
            self.dtype = torch.float32
            logger.info(f"Using float32 dtype for {self.device}")
        
        # Load tokenizer and model
        logger.info(f"Loading DeepSeek-OCR from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        model_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
            "torch_dtype": self.dtype,
            "device_map": self.device
        }
        
        model_kwargs["_attn_implementation"] = "flash_attention_2" if is_flash_attn_2_available() else "eager"
        
        self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
        self.model = self.model.eval()
        
        logger.info("DeepSeek-OCR model loaded successfully")
    
    @property
    def media_type(self):
        """The media type processed by this model."""
        return "image"
    
    @property
    def resolution_mode(self):
        """Current resolution mode."""
        return self._resolution_mode
    
    @resolution_mode.setter
    def resolution_mode(self, value):
        """Change resolution mode at runtime."""
        if value not in RESOLUTION_MODES:
            raise ValueError(
                f"resolution_mode must be one of {list(RESOLUTION_MODES.keys())}, "
                f"got '{value}'"
            )
        self._resolution_mode = value
        logger.info(f"Resolution mode changed to: {value}")
    
    @property
    def operation(self):
        """Current operation mode."""
        return self._operation
    
    @operation.setter
    def operation(self, value):
        """Change operation mode at runtime."""
        if value not in OPERATIONS:
            raise ValueError(
                f"operation must be one of {list(OPERATIONS.keys())}, "
                f"got '{value}'"
            )
        self._operation = value
        logger.info(f"Operation changed to: {value}")
    
    @property
    def prompt(self):
        """Current active prompt."""
        if self._custom_prompt:
            return self._custom_prompt
        return OPERATIONS[self._operation]["prompt"]
    
    @prompt.setter
    def prompt(self, value):
        """Set custom prompt at runtime."""
        self._custom_prompt = value
        logger.info(f"Custom prompt set: {value}")
    
    def _to_detections(self, text: str) -> fo.Detections:
        """Parse DeepSeek-OCR grounding output to FiftyOne Detections.
        
        Parses text containing:
        <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2], ...]<|/det|>
        actual text content
        
        Coordinates are in 0-999 range and need normalization to [0, 1].
        Format is [x1, y1, x2, y2] (top-left, bottom-right corners).
        
        Args:
            text: Model output string containing ref/det tags
        
        Returns:
            fo.Detections: FiftyOne Detections object with all parsed detections
        """
        detections = []
        
        # Pattern: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2], ...]<|/det|>\ntext content
        pattern = r'<\|ref\|>(.*?)<\/\|ref\|><\|det\|>(\[\[.*?\]\])<\/\|det\|>\n(.*?)(?=\n<\|ref\||$)'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            label = match.group(1).strip()
            coords_list = ast.literal_eval(match.group(2))
            text_content = match.group(3).strip()
            
            # Handle multiple boxes for same reference
            for coords in coords_list:
                x1, y1, x2, y2 = coords
                
                # Normalize from 0-999 to 0-1 and convert to [x, y, w, h]
                x_norm = x1 / 999.0
                y_norm = y1 / 999.0
                width_norm = (x2 - x1) / 999.0
                height_norm = (y2 - y1) / 999.0
                
                detection = fo.Detection(
                    label=label,
                    bounding_box=[x_norm, y_norm, width_norm, height_norm],
                    text=text_content
                )
                
                detections.append(detection)
        
        return fo.Detections(detections=detections)
    
    def _predict(self, image: Image.Image, sample) -> Union[fo.Detections, str]:
        """Process image through DeepSeek-OCR.
        
        Args:
            image: PIL Image to process
            sample: FiftyOne sample (has filepath attribute)
        
        Returns:
            - fo.Detections if operation="grounding" (with bounding boxes)
            - str if operation="ocr" or "describe"
        """
        # Get resolution parameters
        mode_params = RESOLUTION_MODES[self.resolution_mode]
        
        # Run inference
        result = self.model.infer(
            self.tokenizer,
            prompt=self.prompt,
            image_file=sample.filepath,
            output_path='',
            base_size=mode_params["base_size"],
            image_size=mode_params["image_size"],
            crop_mode=mode_params["crop_mode"],
            save_results=False,
            test_compress=False,
            eval_mode=True
        )
        
        # Parse output based on operation type
        if OPERATIONS[self.operation]["return_type"] == "detections":
            return self._to_detections(result)
        
        return result
    
    def predict(self, image, sample=None):
        """Process an image with DeepSeek-OCR.
        
        Args:
            image: PIL Image or numpy array to process
            sample: FiftyOne sample containing the image filepath
        
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)

