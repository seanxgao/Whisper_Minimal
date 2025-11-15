"""
Script to export both student models to ONNX format.
"""

from __future__ import annotations

import logging
from pathlib import Path

from vad_distill.utils.config import load_yaml
from vad_distill.utils.logging_utils import setup_logging
from vad_distill.distill.tiny_vad.export_onnx import export_tiny_vad_onnx
from vad_distill.distill.chunk_trigger.export_onnx import export_chunk_trigger_onnx


def main():
    """Main entry point for exporting all models to ONNX."""
    setup_logging()
    
    logger = logging.getLogger(__name__)
    
    # Export Student A
    logger.info("Exporting Student A (tiny VAD) to ONNX...")
    tiny_vad_config = load_yaml("vad_distill/configs/student_tiny_vad.yaml")
    export_tiny_vad_onnx(tiny_vad_config)
    
    # Export Student B
    logger.info("Exporting Student B (chunk trigger) to ONNX...")
    chunk_config = load_yaml("vad_distill/configs/student_chunk.yaml")
    export_chunk_trigger_onnx(chunk_config)
    
    logger.info("All ONNX exports completed")


if __name__ == "__main__":
    main()

