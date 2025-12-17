"""Centralized data path utilities derived from the YAML configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

from vad_distill.config import load_config, resolve_path

PathMapping = Dict[str, Path]


def resolve_paths(paths_cfg: Dict[str, str]) -> PathMapping:
    resolved: PathMapping = {}
    for key, value in paths_cfg.items():
        resolved[key] = resolve_path(value)
    return resolved


@lru_cache(maxsize=1)
def _default_paths() -> PathMapping:
    config = load_config()
    return resolve_paths(config.get("paths", {}))


def ensure_dirs(paths: PathMapping | None = None) -> None:
    """
    Ensure that critical external directories exist.
    """
    resolved = paths or _default_paths()
    for key in (
        "chunks_dir",
        "teacher_prob_dir",
        "features_dir",
        "metadata_dir",
        "checkpoint_dir",
        "logs_dir",
        "onnx_dir",
    ):
        path = resolved.get(key)
        if path:
            path.mkdir(parents=True, exist_ok=True)


# Backwards-compatible constants for legacy scripts.
PATHS = _default_paths()
DATA_ROOT = PATHS.get("data_root", resolve_path("."))
RAW_DIR = DATA_ROOT / "raw"
CHUNKS_DIR = PATHS.get("chunks_dir", DATA_ROOT / "chunks")
TEACHER_PROBS_DIR = PATHS.get("teacher_prob_dir", DATA_ROOT / "teacher_probs")
CHECKPOINTS_DIR = PATHS.get("checkpoint_dir", DATA_ROOT / "checkpoints")
LOGS_DIR = PATHS.get("logs_dir", DATA_ROOT / "logs")
EXPORT_DIR = PATHS.get("onnx_dir", DATA_ROOT / "export")


__all__ = [
    "DATA_ROOT",
    "RAW_DIR",
    "CHUNKS_DIR",
    "TEACHER_PROBS_DIR",
    "CHECKPOINTS_DIR",
    "LOGS_DIR",
    "EXPORT_DIR",
    "PATHS",
    "resolve_paths",
    "ensure_dirs",
]
