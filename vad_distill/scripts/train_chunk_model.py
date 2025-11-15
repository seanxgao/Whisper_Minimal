"""
Script to train Student B: chunk trigger model.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vad_distill.utils.config import load_yaml
from vad_distill.utils.logging_utils import setup_logging
from vad_distill.distill.chunk_trigger.train import train_chunk_trigger


def main():
    """Main entry point for training chunk trigger model."""
    setup_logging()
    
    # Load configuration
    config_path = Path("vad_distill/configs/student_chunk.yaml")
    config = load_yaml(str(config_path))
    
    # Train model
    train_chunk_trigger(config)


if __name__ == "__main__":
    main()

