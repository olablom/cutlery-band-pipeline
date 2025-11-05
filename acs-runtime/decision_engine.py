# decision_engine.py
"""Decision engine for classification results."""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import yaml
import numpy as np

from registry_utils import (
    load_registry as load_registry_utils,
    find_variant_match,
)


def load_thresholds(thresholds_path: str) -> Dict[str, Dict[str, float]]:
    """
    Load thresholds from YAML file.
    
    Args:
        thresholds_path: Path to thresholds.yaml
        
    Returns:
        Dict mapping class names to threshold configs
    """
    with open(thresholds_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_plc_actions(actions_path: str) -> Dict[str, str]:
    """
    Load PLC action templates from YAML file.
    
    Args:
        actions_path: Path to plc_actions.yaml
        
    Returns:
        Dict mapping decision types to action templates
    """
    with open(actions_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_decision(
    class_id: int,
    confidence: float,
    softmax_dict: Dict[str, float],
    class_name: str,
    thresholds: Dict[str, Dict[str, float]],
    plc_actions: Dict[str, str],
    registry: Dict[str, Dict[str, Any]],
    registry_path: str,
    features: Optional[np.ndarray] = None,
    manufacturer: Optional[str] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Make decision based on classification result.
    
    System class_id mapping:
    - BACKGROUND → 9999
    - Unknown variant → 0
    - Known variants → 2000/3000/4000 series (from registry)
    
    Args:
        class_id: Model's predicted class ID (from ONNX, not used directly)
        confidence: Softmax confidence
        softmax_dict: Full softmax probabilities
        class_name: Predicted class name
        thresholds: Threshold configuration
        plc_actions: PLC action templates
        registry: Registry dict with manufacturer mappings
        registry_path: Path to registry directory (for loading prototypes)
        features: Optional feature vector for variant matching
        manufacturer: Optional manufacturer code (if already known)
        variant: Optional variant code (if already known)
        
    Returns:
        Decision object with keys:
            - pred_type: Predicted class name
            - conf: Confidence value
            - decision_class: Decision class (e.g., "BACKGROUND_TRASH", "UNKNOWN_VARIANT", "HIGH_CONFIDENCE_SORT")
            - class_id: System class ID (9999 for BACKGROUND, 0 for unknown, 2000-4999 for known variants)
            - target_bin: Target bin number (0 for now)
            - manufacturer: Variant name (if matched)
            - variant_score: Cosine similarity score if variant matched
    """
    bg_threshold = thresholds.get("BACKGROUND", {}).get("softmax_threshold", 0.50)
    
    # Pseudo-background rule: if confidence is too low, treat as BACKGROUND
    if confidence < bg_threshold:
        return {
            "pred_type": "BACKGROUND",
            "conf": confidence,
            "decision_class": "BACKGROUND_TRASH",
            "class_id": 9999,
            "target_bin": 0,
            "manufacturer": None,
        }
    
    # Explicit BACKGROUND class (if model had it)
    if class_name == "BACKGROUND":
        return {
            "pred_type": "BACKGROUND",
            "conf": confidence,
            "decision_class": "BACKGROUND_TRASH",
            "class_id": 9999,
            "target_bin": 0,
            "manufacturer": None,
        }
    
    # Cutlery classes (KNIFE, FORK, SPOON)
    # Check if prediction is actually a known cutlery type
    if class_name not in ("FORK", "KNIFE", "SPOON"):
        # Unknown type → treat as BACKGROUND
        return {
            "pred_type": "BACKGROUND",
            "conf": confidence,
            "decision_class": "BACKGROUND_TRASH",
            "class_id": 9999,
            "target_bin": 0,
            "manufacturer": None,
        }
    
    # Known cutlery type - check confidence threshold
    threshold = thresholds.get(class_name, {}).get("softmax_threshold", 0.85)
    
    # Try to match variant using features (if available)
    variant_name = None
    registry_class_id = None
    variant_score = 0.0
    
    if features is not None:
        # Use cosine matching against prototypes
        variant_name, registry_class_id, variant_score = find_variant_match(
            features=features,
            type_name=class_name,
            registry_path=registry_path,
        )
        
        if variant_name:
            manufacturer = variant_name  # Use variant name as manufacturer identifier
    
    if confidence >= threshold:
        # High confidence - normal sort
        if variant_name and registry_class_id and manufacturer:
            # Found variant match - use variant class_id
            return {
                "pred_type": class_name,
                "conf": confidence,
                "decision_class": "HIGH_CONFIDENCE_SORT",
                "class_id": registry_class_id,  # 2000/3000/4000 series
                "target_bin": 0,
                "manufacturer": variant_name,
                "variant_score": variant_score,
            }
        elif registry.get(class_name.upper()):
            # Registry exists but no match - use unknown variant path
            return {
                "pred_type": class_name,
                "conf": confidence,
                "decision_class": "UNKNOWN_VARIANT",
                "class_id": 0,  # Unknown variant → 0
                "target_bin": 0,
                "manufacturer": None,
            }
        else:
            # No registry file exists - use unknown variant path
            return {
                "pred_type": class_name,
                "conf": confidence,
                "decision_class": "UNKNOWN_VARIANT",
                "class_id": 0,  # Unknown variant → 0
                "target_bin": 0,
                "manufacturer": None,
            }
    else:
        # Low confidence - embedding rescue (if variant match available)
        if variant_name and registry_class_id:
            return {
                "pred_type": class_name,
                "conf": confidence,
                "decision_class": "EMBEDDING_RESCUE",
                "class_id": registry_class_id,  # 2000/3000/4000 series
                "target_bin": 0,
                "manufacturer": variant_name,
                "variant_score": variant_score,
            }
        else:
            return {
                "pred_type": class_name,
                "conf": confidence,
                "decision_class": "UNKNOWN_VARIANT",
                "class_id": 0,  # Unknown variant → 0
                "target_bin": 0,
                "manufacturer": None,
            }


def load_registry(registry_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load registry files for all cutlery types.
    
    This is a wrapper that calls registry_utils.load_registry for consistency.
    
    Args:
        registry_path: Path to registry directory
        
    Returns:
        Dict mapping type names to registry data
    """
    return load_registry_utils(registry_path)


# Note: lookup_manufacturer is deprecated - use find_variant_match from registry_utils instead


def resolve_plc_action(decision_class: str, plc_actions: Dict[str, str], manufacturer: Optional[str] = None) -> str:
    """
    Resolve decision_class to actual PLC action string.
    
    Args:
        decision_class: Decision class name (e.g., "BACKGROUND_TRASH")
        plc_actions: PLC action templates from YAML
        manufacturer: Optional manufacturer code for template substitution
        
    Returns:
        Resolved PLC action string (e.g., "REJECT_TO_TRASH_LANE")
    """
    template = plc_actions.get(decision_class, "")
    if not template:
        return ""
    
    if manufacturer and "{manufacturer}" in template:
        return template.format(manufacturer=manufacturer)
    
    return template

