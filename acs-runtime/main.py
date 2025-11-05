#!/usr/bin/env python3
# main.py
"""ACS runtime main entry point."""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any

from utils import load_config, ensure_dir
from capture import load_image, preprocess_for_model
from classifier import classify as classify_onnx, load_model as load_model_onnx
from decision_engine import load_thresholds, load_plc_actions, make_decision, resolve_plc_action, load_registry
from plc_packet import create_plc_packet, packet_to_hex

# Try to import Hailo classifier (may not be available on all systems)
try:
    from classifier_hailo import classify as classify_hailo, load_model as load_model_hailo
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    classify_hailo = None
    load_model_hailo = None


def load_labels(labels_path: str) -> Dict[int, str]:
    """
    Load class labels from JSON file.
    
    Args:
        labels_path: Path to type_labels.json
        
    Returns:
        Dict mapping class IDs to class names
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)
    
    # Convert string keys to int keys
    return {int(k): v for k, v in labels_dict.items()}


def now_ms() -> int:
    """Get current timestamp in milliseconds."""
    return int(time.time() * 1000)


def log_inference(
    log_path: str,
    image_path: str,
    decision_obj: Dict[str, Any],
    plc_action_resolved: str,
    plc_frame_hex: str,
    latency_ms: float,
) -> None:
    """
    Log inference result to JSONL file.
    
    Args:
        log_path: Path to log file
        image_path: Input image path
        decision_obj: Decision object
        plc_action_resolved: Resolved PLC action string
        plc_frame_hex: PLC frame as hex string
        latency_ms: Inference latency in milliseconds
    """
    log_entry = {
        "ts_ms": now_ms(),
        "input_file": str(image_path),
        "pred_label": decision_obj.get("pred_type"),
        "conf": decision_obj.get("conf"),
        "latency_ms": round(latency_ms, 2),
        "decision": decision_obj,
        "plc_action_resolved": plc_action_resolved,
        "plc_frame_hex": plc_frame_hex,
    }
    
    ensure_dir(Path(log_path).parent)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ACS runtime inference")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--config",
        default="runtime_config.yaml",
        help="Path to runtime config (default: runtime_config.yaml)",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load labels
    labels = load_labels(config["labels_path"])
    
    # Load thresholds and PLC actions
    thresholds = load_thresholds(config["thresholds_path"])
    plc_actions = load_plc_actions(config["plc_actions_path"])
    
    # Load registry
    registry_path = config.get("registry_path", "registry")
    registry = load_registry(registry_path)
    
    # Determine backend and load model
    backend = config.get("inference_backend", "onnx")
    
    if backend == "onnx":
        load_model_onnx(config["model_path"], config["labels_path"])
        classify_fn = classify_onnx
    elif backend == "hailo":
        if not HAILO_AVAILABLE:
            print(f"Error: Hailo backend requested but classifier_hailo not available", file=sys.stderr)
            return 1
        if "hef_path" not in config:
            print(f"Error: Hailo backend requires 'hef_path' in config", file=sys.stderr)
            return 1
        load_model_hailo(config["hef_path"], config["labels_path"])
        classify_fn = classify_hailo
    else:
        print(f"Error: Unknown inference_backend: {backend}", file=sys.stderr)
        return 1
    
    print(f"[main] Using backend: {backend}")
    
    # Load and preprocess image
    img = load_image(args.image_path)
    if img is None:
        print(f"Error: Could not load image from {args.image_path}", file=sys.stderr)
        return 1
    
    img_preprocessed = preprocess_for_model(img)
    
    # Classify
    class_id, confidence, softmax_dict, latency_ms = classify_fn(img_preprocessed)
    
    # Get class name
    class_name = labels.get(class_id, "BACKGROUND")
    
    # Make decision (with registry lookup)
    # Note: features=None for now - will be added when variant classifier is ready
    decision_obj = make_decision(
        class_id=class_id,
        confidence=confidence,
        softmax_dict=softmax_dict,
        class_name=class_name,
        thresholds=thresholds,
        plc_actions=plc_actions,
        registry=registry,
        registry_path=registry_path,
        features=None,  # TODO: Extract features from variant classifier when available
    )
    
    # Resolve PLC action string
    manufacturer = decision_obj.get("manufacturer")
    plc_action_resolved = resolve_plc_action(
        decision_class=decision_obj["decision_class"],
        plc_actions=plc_actions,
        manufacturer=manufacturer,
    )
    
    # Create PLC packet (use same timestamp as log)
    current_ts = now_ms()
    packet = create_plc_packet(decision_obj, ts_ms=current_ts)
    frame_hex = packet_to_hex(packet)
    
    # Log inference
    log_inference(
        log_path=config["log_path"],
        image_path=args.image_path,
        decision_obj=decision_obj,
        plc_action_resolved=plc_action_resolved,
        plc_frame_hex=frame_hex,
        latency_ms=latency_ms,
    )
    
    # Output PLC packet as hex
    print(frame_hex)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

