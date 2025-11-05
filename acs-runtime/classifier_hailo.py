#!/usr/bin/env python3
# classifier_hailo.py
"""Hailo-based classification (HEF format)."""

import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np

# Try to import HailoRT (may not be available on all systems)
try:
    from hailo_platform import Device, VStream, InferVStreams, HEF
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    Device = None
    VStream = None
    InferVStreams = None
    HEF = None


# Global session and labels (loaded once)
_device: Optional[Any] = None
_input_vstreams: Optional[Any] = None
_output_vstreams: Optional[Any] = None
_network_group: Optional[Any] = None
_labels: Optional[Dict[int, str]] = None


def load_model(hef_path: str, labels_path: str) -> None:
    """
    Load Hailo HEF model and labels.
    
    Args:
        hef_path: Path to HEF file
        labels_path: Path to labels JSON file
    """
    global _device, _input_vstreams, _output_vstreams, _network_group, _labels
    
    if not HAILO_AVAILABLE:
        raise RuntimeError("HailoRT not available. Install Hailo SDK.")
    
    if _device is None:
        # Initialize device
        _device = Device()
        
        # Load HEF
        hef = HEF(hef_path)
        _network_group = _device.configure(hef)
        _network_group_params = _network_group.create_params()
        
        # Create VStreams
        _input_vstreams, _output_vstreams = InferVStreams(_network_group)
        
        print(f"[classifier_hailo] Loaded HEF: {hef_path}")
    
    if _labels is None:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_dict = json.load(f)
        # Convert string keys to int keys
        _labels = {int(k): v for k, v in labels_dict.items()}
        print(f"[classifier_hailo] Loaded labels: {labels_path}")


def classify(image: np.ndarray) -> Tuple[int, float, Dict[str, float], float]:
    """
    Classify preprocessed image using Hailo model.
    
    Args:
        image: Preprocessed image array (1, C, H, W) - already normalized to [0,1]
        
    Returns:
        Tuple of (class_id, confidence, softmax_dict, latency_ms)
        - class_id: Model's class ID (0=FORK, 1=KNIFE, 2=SPOON)
        - confidence: Softmax probability
        - softmax_dict: Dict mapping class names to probabilities
        - latency_ms: Inference latency in milliseconds
    """
    global _device, _input_vstreams, _output_vstreams, _network_group, _labels
    
    if _device is None or _labels is None:
        raise RuntimeError("Model and labels must be loaded first with load_model()")
    
    # Get input name and shape
    input_name = _input_vstreams[0].name
    input_shape = _input_vstreams[0].shape
    
    # Ensure image matches expected shape
    # Hailo expects (1, 3, 170, 480) for NCHW
    if image.shape != input_shape:
        # Reshape if needed
        image = image.reshape(input_shape)
    
    # Convert to correct data type (Hailo may expect uint8 or float32)
    # Check input stream info
    input_info = _input_vstreams[0].info
    if hasattr(input_info, 'dtype'):
        if input_info.dtype == np.uint8:
            # Denormalize if needed
            image = (image * 255.0).astype(np.uint8)
        else:
            image = image.astype(np.float32)
    
    # Run inference with timing
    t0 = time.time()
    
    # Send input
    _input_vstreams[0].send(image)
    
    # Receive output
    output = _output_vstreams[0].recv()
    
    lat_ms = (time.time() - t0) * 1000
    
    # Get logits (first output, first batch)
    logits = output[0]  # [num_classes]
    
    # Apply softmax
    probs = _softmax(logits)
    
    # Get predicted class
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    
    # Build softmax dict with class names
    softmax_dict = {}
    for idx, prob in enumerate(probs):
        class_name = _labels.get(idx, f"CLASS_{idx}")
        softmax_dict[class_name] = float(prob)
    
    print(f"[inference] {lat_ms:.2f} ms (Hailo)")
    
    return class_id, confidence, softmax_dict, lat_ms


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    x = x - np.max(x)  # Numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

