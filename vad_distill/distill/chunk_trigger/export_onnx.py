"""ONNX export for chunk trigger model."""

from __future__ import annotations

import torch
import torch.onnx
from pathlib import Path
from typing import Dict, Any
import logging

from vad_distill.distill.chunk_trigger.model import build_chunk_trigger_model

logger = logging.getLogger(__name__)


def export_chunk_trigger_onnx(config: Dict[str, Any], checkpoint_path: str | None = None) -> None:
    """
    Export chunk trigger model to ONNX format.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint file. If None, uses best_model.pt from checkpoint_dir
    """
    device = torch.device("cpu")  # ONNX export on CPU
    
    # Build model
    model = build_chunk_trigger_model(config)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'models/chunk_trigger/checkpoints'))
        checkpoint_path = checkpoint_dir / "best_model.pt"
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    window_size = config.get('model', {}).get('window_size', 50)
    dummy_input = torch.randn(1, window_size, 1)  # (batch=1, window_size, 1)
    
    # Export to ONNX
    onnx_dir = Path(config.get('paths', {}).get('onnx_dir', 'models/chunk_trigger/onnx'))
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "chunk_trigger.onnx"
    
    logger.info(f"Exporting to ONNX: {onnx_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['vad_window'],
        output_names=['trigger_logit'],
        opset_version=11,
    )
    
    logger.info(f"ONNX export completed: {onnx_path}")

