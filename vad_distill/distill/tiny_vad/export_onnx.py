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
    
    # Create dummy input with fixed size: (batch=1, time=100, freq=80)
    n_mels = config.get('model', {}).get('n_mels', 80)
    dummy_input = torch.randn(1, 100, n_mels)  # Fixed-size chunk input
    
    # Export to ONNX
    onnx_dir = Path(config.get('paths', {}).get('onnx_dir', 'models/tiny_vad/onnx'))
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "tiny_vad.onnx"
    
    logger.info(f"Exporting to ONNX: {onnx_path}")
    logger.info(f"Input shape: {dummy_input.shape} (fixed-size chunks)")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['mel_features'],
        output_names=['vad_logits'],
        # Fixed-size input: no dynamic axes
        opset_version=11,  # Adjust if needed
    )
    
    logger.info(f"ONNX export completed: {onnx_path}")
    
    # Validate ONNX model with ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np
        
        logger.info("Validating ONNX model with ONNX Runtime...")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(str(onnx_path))
        
        # Run inference with dummy input
        dummy_input_np = dummy_input.numpy().astype(np.float32)
        onnx_outputs = session.run(None, {'mel_features': dummy_input_np})
        onnx_output = onnx_outputs[0]
        
        # Run PyTorch inference for comparison
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()
        
        # Compare outputs
        max_diff = np.abs(onnx_output - pytorch_output).max()
        mean_diff = np.abs(onnx_output - pytorch_output).mean()
        
        logger.info(f"ONNX vs PyTorch comparison:")
        logger.info(f"  Max difference: {max_diff:.2e}")
        logger.info(f"  Mean difference: {mean_diff:.2e}")
        
        tolerance = 1e-5
        if max_diff > tolerance:
            logger.warning(
                f"ONNX output differs from PyTorch by {max_diff:.2e} "
                f"(tolerance: {tolerance:.2e})"
            )
        else:
            logger.info(f"ONNX validation passed (tolerance: {tolerance:.2e})")
        
    except ImportError:
        logger.warning("onnxruntime not available, skipping ONNX validation")
    except Exception as e:
        logger.warning(f"ONNX validation failed: {e}")

