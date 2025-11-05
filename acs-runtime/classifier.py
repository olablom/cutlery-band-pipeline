#!/usr/bin/env python3
# classifier.py
"""ONNX-based classification."""

import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import onnxruntime as ort


# Global session and labels (loaded once)
_session: Optional[ort.InferenceSession] = None
_labels: Optional[Dict[int, str]] = None


def load_model(model_path: str, labels_path: str) -> None:
    """
    Load ONNX model and labels.
    
    Args:
        model_path: Path to ONNX model file
        labels_path: Path to labels JSON file
    """
    global _session, _labels
    
    if _session is None:
        _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"[classifier] Loaded model: {model_path}")
    
    if _labels is None:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_dict = json.load(f)
        # Convert string keys to int keys
        _labels = {int(k): v for k, v in labels_dict.items()}
        print(f"[classifier] Loaded labels: {labels_path}")


def classify(image: np.ndarray) -> Tuple[int, float, Dict[str, float], float]:
    """
    Classify preprocessed image using ONNX model.
    
    Args:
        image: Preprocessed image array (1, C, H, W) - already normalized to [0,1]
        
    Returns:
        Tuple of (class_id, confidence, softmax_dict, latency_ms)
        - class_id: Model's class ID (0=FORK, 1=KNIFE, 2=SPOON)
        - confidence: Softmax probability
        - softmax_dict: Dict mapping class names to probabilities
        - latency_ms: Inference latency in milliseconds
    """
    global _session, _labels
    
    if _session is None or _labels is None:
        raise RuntimeError("Model and labels must be loaded first with load_model()")
    
    # Apply ImageNet normalization (mean/std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    
    # Image is already in [0, 1] range from preprocess_for_model
    image_normalized = (image - mean) / std
    
    # Get input name
    input_name = _session.get_inputs()[0].name
    
    # Run inference with timing
    t0 = time.time()
    outputs = _session.run(None, {input_name: image_normalized})
    lat_ms = (time.time() - t0) * 1000
    
    # Get logits (first output, first batch)
    logits = outputs[0][0]  # [num_classes]
    
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
    
    print(f"[inference] {lat_ms:.2f} ms")
    
    return class_id, confidence, softmax_dict, lat_ms


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities."""
    x = x - np.max(x)  # Numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

