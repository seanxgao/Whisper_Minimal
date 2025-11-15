"""Configuration loading utilities."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML file

    Returns:
        Dictionary containing the configuration
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path_obj, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config if config is not None else {}

