"""Configuration helpers for the VAD distillation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent.parent


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load the default YAML configuration and optionally merge overrides.

    Args:
        config_path: Optional override path. Defaults to default_config.yaml.

    Returns:
        Parsed configuration dictionary with defaults applied.
    """
    default_path = CONFIG_DIR / "default_config.yaml"
    config = _read_yaml(default_path)
    if config_path:
        override_path = Path(config_path)
        overrides = _read_yaml(override_path)
        if overrides:
            config = _deep_update(config, overrides)
    return config


def resolve_path(path_str: str, base: Path | None = None) -> Path:
    """
    Resolve a filesystem path relative to project root.

    Args:
        path_str: Raw string from config (absolute or relative).
        base: Optional base directory override.

    Returns:
        Absolute Path object.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    base_dir = base or PROJECT_ROOT
    return (base_dir / path).resolve()


__all__ = ["CONFIG_DIR", "PROJECT_ROOT", "load_config", "resolve_path"]
