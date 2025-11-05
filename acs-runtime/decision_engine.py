# decision_engine.py
"""Decision engine for classification results."""

from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import yaml
import numpy as np


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
    manufacturer: Optional[str] = None,
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Make decision based on classification result.
    
    System class_id mapping:
    - BACKGROUND → 9999
    - Unknown variant → 0
    - Future: real cutlery → 2000/3000/4000 series
    
    Args:
        class_id: Model's predicted class ID (from ONNX, not used directly)
        confidence: Softmax confidence
        softmax_dict: Full softmax probabilities
        class_name: Predicted class name
        thresholds: Threshold configuration
        plc_actions: PLC action templates
        registry: Registry dict with manufacturer mappings
        manufacturer: Optional manufacturer code (if already known)
        variant: Optional variant code (for future use)
        
    Returns:
        Decision object with keys:
            - pred_type: Predicted class name
            - conf: Confidence value
            - decision_class: Decision class (e.g., "BACKGROUND_TRASH", "UNKNOWN_VARIANT", "HIGH_CONFIDENCE_SORT")
            - class_id: System class ID (9999 for BACKGROUND, 0 for unknown, 2000-4999 for known manufacturers)
            - target_bin: Target bin number (0 for now)
            - manufacturer: Manufacturer code (if matched)
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
    
    # Try to lookup manufacturer from registry
    if not manufacturer:
        manufacturer, registry_class_id = lookup_manufacturer(class_name, registry)
    else:
        registry_class_id = None
    
    if confidence >= threshold:
        # High confidence - normal sort
        if manufacturer and registry_class_id:
            # Found in registry - use manufacturer class_id
            return {
                "pred_type": class_name,
                "conf": confidence,
                "decision_class": "HIGH_CONFIDENCE_SORT",
                "class_id": registry_class_id,  # 2000/3000/4000 series
                "target_bin": 0,  # Will be set by registry lookup later
                "manufacturer": manufacturer,
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
        # Low confidence - embedding rescue (if manufacturer available)
        if manufacturer and registry_class_id:
            return {
                "pred_type": class_name,
                "conf": confidence,
                "decision_class": "EMBEDDING_RESCUE",
                "class_id": registry_class_id,  # 2000/3000/4000 series
                "target_bin": 0,  # Will be set by registry lookup later
                "manufacturer": manufacturer,
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
    
    Args:
        registry_path: Path to registry directory
        
    Returns:
        Dict mapping type names to registry data
    """
    registry = {}
    registry_dir = Path(registry_path)
    
    if not registry_dir.exists():
        return registry
    
    for type_name in ["fork", "knife", "spoon"]:
        registry_file = registry_dir / f"{type_name}.json"
        if registry_file.exists():
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    registry[type_name.upper()] = json.load(f)
            except Exception as e:
                print(f"[decision_engine] Warning: Could not load {registry_file}: {e}")
    
    return registry


def lookup_manufacturer(
    class_name: str,
    registry: Dict[str, Dict[str, Any]],
    embedding: Optional[np.ndarray] = None,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Look up manufacturer from registry (placeholder for future embedding matching).
    
    Args:
        class_name: Predicted class name (FORK, KNIFE, SPOON)
        registry: Registry dict loaded from JSON files
        embedding: Optional embedding vector (for future use)
        
    Returns:
        Tuple of (manufacturer_code, class_id) or (None, None) if not found
    """
    type_key = class_name.upper()
    if type_key not in registry:
        return None, None
    
    type_registry = registry[type_key]
    manufacturers = type_registry.get("manufacturers", [])
    
    if not manufacturers:
        return None, None
    
    # TODO: Future: Use embedding to match manufacturer
    # For now, we return None to indicate no match
    # This will be implemented when variant classifier is ready
    # 
    # When implemented, this should:
    # 1. Compute embedding from image (using variant classifier)
    # 2. Compare with embeddings in registry
    # 3. Return best match if similarity > threshold
    
    return None, None


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

