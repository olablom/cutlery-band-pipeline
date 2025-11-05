# capture.py
"""Image capture and preprocessing."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array (H, W, C) or None if failed
    """
    if not Path(image_path).exists():
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def preprocess_for_model(img: np.ndarray, target_size: Tuple[int, int] = (480, 170)) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        img: Input image (H, W, C)
        target_size: Target (width, height)
        
    Returns:
        Preprocessed image ready for model
    """
    # Resize to target size
    img_resized = cv2.resize(img, target_size)
    
    # Normalize to [0, 1] and convert to float32
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Convert HWC to CHW and add batch dimension
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    
    return img_batch

