# registry_utils.py
"""Utilities for loading and matching against manufacturer/variant registry."""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import numpy as np


def load_registry(registry_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load registry files for all cutlery types.
    
    Args:
        registry_path: Path to registry directory
        
    Returns:
        Dict mapping type names (FORK, KNIFE, SPOON) to registry data
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
                print(f"[registry_utils] Warning: Could not load {registry_file}: {e}")
    
    return registry


def load_prototypes(registry_path: str, type_name: str) -> Dict[str, np.ndarray]:
    """
    Load prototype embeddings for a specific cutlery type.
    
    Args:
        registry_path: Path to registry directory
        type_name: Cutlery type (fork, knife, spoon)
        
    Returns:
        Dict mapping variant names to prototype embedding vectors
    """
    prototypes_file = Path(registry_path) / f"{type_name}_prototypes.json"
    
    if not prototypes_file.exists():
        return {}
    
    try:
        with open(prototypes_file, "r", encoding="utf-8") as f:
            prototypes_dict = json.load(f)
        
        # Convert lists to numpy arrays
        prototypes = {}
        for name, embedding_list in prototypes_dict.items():
            prototypes[name] = np.array(embedding_list, dtype=np.float32)
        
        return prototypes
    except Exception as e:
        print(f"[registry_utils] Warning: Could not load {prototypes_file}: {e}")
        return {}


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity score in [0, 1] (1.0 = identical, 0.0 = orthogonal)
    """
    # Normalize vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    # Cosine similarity = dot product / (norm1 * norm2)
    similarity = np.dot(v1, v2) / (norm_v1 * norm_v2)
    
    # Clamp to [0, 1] (should already be in range, but safety check)
    return float(np.clip(similarity, -1.0, 1.0))


def cosine_match(
    features: np.ndarray,
    prototypes: Dict[str, np.ndarray],
) -> Tuple[Optional[str], float]:
    """
    Match features against prototypes using cosine similarity.
    
    Args:
        features: Feature vector from model (e.g., 512-dim embedding)
        prototypes: Dict mapping variant names to prototype vectors
        
    Returns:
        Tuple of (best_match_name, best_score) or (None, 0.0) if no match
    """
    if not prototypes or features is None:
        return None, 0.0
    
    best_name = None
    best_score = -1.0
    
    for name, prototype in prototypes.items():
        score = cosine_similarity(features, prototype)
        if score > best_score:
            best_score = score
            best_name = name
    
    return best_name, best_score


def find_variant_match(
    features: np.ndarray,
    type_name: str,
    registry_path: str,
    threshold: Optional[float] = None,
) -> Tuple[Optional[str], Optional[int], float]:
    """
    Find matching variant for given features.
    
    Args:
        features: Feature vector from model
        type_name: Cutlery type (FORK, KNIFE, SPOON)
        registry_path: Path to registry directory
        threshold: Optional threshold override (uses registry default if None)
        
    Returns:
        Tuple of (variant_name, class_id, similarity_score)
        Returns (None, None, 0.0) if no match above threshold
    """
    # Load registry for type
    registry = load_registry(registry_path)
    type_key = type_name.upper()
    
    if type_key not in registry:
        return None, None, 0.0
    
    type_reg = registry[type_key]
    
    # Get threshold from registry if not provided
    if threshold is None:
        threshold = type_reg.get("manufacturer_threshold", 0.85)
    
    # Load prototypes
    prototypes = load_prototypes(registry_path, type_name.lower())
    
    if not prototypes:
        return None, None, 0.0
    
    # Find best match
    best_name, best_score = cosine_match(features, prototypes)
    
    if best_score < threshold:
        return None, None, best_score
    
    # Find class_id from registry
    variants = type_reg.get("variants", [])
    class_id = None
    
    for variant in variants:
        if variant.get("name") == best_name:
            class_id = variant.get("id")
            break
    
    return best_name, class_id, best_score

