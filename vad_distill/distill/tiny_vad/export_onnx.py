"""ONNX export for tiny VAD model."""

from __future__ import annotations

import torch
import torch.onnx
from pathlib import Path
from typing import Dict, Any
import logging

from vad_distill.distill.tiny_vad.model import build_tiny_vad_model

logger = logging.getLogger(__name__)


def export_tiny_vad_onnx(config: Dict[str, Any], checkpoint_path: str | None = None) -> None:
    """
    Export tiny VAD model to ONNX format.

    Args:
        config: Configuration dictionary
        checkpoint_path: Path to checkpoint file. If None, uses best_model.pt from checkpoint_dir
    """
    device = torch.device("cpu")  # ONNX export on CPU
    
    # Build model
    model = build_tiny_vad_model(config)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'models/tiny_vad/checkpoints'))
        checkpoint_path = checkpoint_dir / "best_model.pt"
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    n_mels = config.get('model', {}).get('n_mels', 80)
    # Dynamic time dimension
    dummy_input = torch.randn(1, 100, n_mels)  # (batch=1, time=100, freq=n_mels)
    
    # Export to ONNX
    onnx_dir = Path(config.get('paths', {}).get('onnx_dir', 'models/tiny_vad/onnx'))
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "tiny_vad.onnx"
    
    logger.info(f"Exporting to ONNX: {onnx_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['mel_features'],
        output_names=['vad_logits'],
        dynamic_axes={
            'mel_features': {1: 'time'},  # Dynamic time dimension
            'vad_logits': {1: 'time'},
        },
        opset_version=11,  # Adjust if needed
    )
    
    logger.info(f"ONNX export completed: {onnx_path}")

