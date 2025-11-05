# utils.py
"""Utility functions for ACS runtime."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

