"""
Script to train Student A: tiny frame-level VAD model.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vad_distill.utils.config import load_yaml
from vad_distill.utils.logging_utils import setup_logging
from vad_distill.distill.tiny_vad.train import train_tiny_vad


def main():
    """Main entry point for training tiny VAD model."""
    setup_logging()
    
    # Load configuration
    config_path = Path("vad_distill/configs/student_tiny_vad.yaml")
    config = load_yaml(str(config_path))
    
    # Train model
    train_tiny_vad(config)


if __name__ == "__main__":
    main()

