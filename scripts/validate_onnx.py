"""Validate ONNX model against PyTorch model."""

from __future__ import annotations

import argparse
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
import logging

from vad_distill.distill.tiny_vad.model import build_tiny_vad_model
from vad_distill.utils.config import load_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_onnx(
    onnx_path: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    num_tests: int = 10,
    tolerance: float = 1e-5,
) -> bool:
    """
    Validate ONNX model against PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        config_path: Path to config YAML
        checkpoint_path: Path to PyTorch checkpoint
        num_tests: Number of random inputs to test
        tolerance: Maximum allowed difference
    
    Returns:
        True if validation passes
    """
    onnx_path = Path(onnx_path)
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load config
    config = load_yaml(str(config_path))
    n_mels = config.get('model', {}).get('n_mels', 80)
    
    # Build PyTorch model
    logger.info("Loading PyTorch model...")
    pytorch_model = build_tiny_vad_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])
    pytorch_model.eval()
    
    # Load ONNX model
    logger.info("Loading ONNX model...")
    session = ort.InferenceSession(str(onnx_path))
    
    # Run tests
    logger.info(f"Running {num_tests} validation tests...")
    max_diff = 0.0
    mean_diff = 0.0
    
    for i in range(num_tests):
        # Generate random input
        dummy_input = torch.randn(1, 100, n_mels)
        dummy_input_np = dummy_input.numpy().astype(np.float32)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input).numpy()
        
        # ONNX inference
        onnx_outputs = session.run(None, {'mel_features': dummy_input_np})
        onnx_output = onnx_outputs[0]
        
        # Compare
        diff = np.abs(onnx_output - pytorch_output)
        max_diff = max(max_diff, diff.max())
        mean_diff += diff.mean()
    
    mean_diff /= num_tests
    
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Max difference: {max_diff:.2e}")
    logger.info(f"Mean difference: {mean_diff:.2e}")
    logger.info(f"Tolerance: {tolerance:.2e}")
    
    if max_diff <= tolerance:
        logger.info("PASS: ONNX model matches PyTorch model")
        return True
    else:
        logger.error(f"FAIL: ONNX model differs by {max_diff:.2e} > {tolerance:.2e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate ONNX model")
    parser.add_argument('onnx_path', type=str, help='Path to ONNX model')
    parser.add_argument('config_path', type=str, help='Path to config YAML')
    parser.add_argument('checkpoint_path', type=str, help='Path to PyTorch checkpoint')
    parser.add_argument('--num_tests', type=int, default=10, help='Number of test inputs')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='Maximum allowed difference')
    
    args = parser.parse_args()
    
    success = validate_onnx(
        onnx_path=args.onnx_path,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        num_tests=args.num_tests,
        tolerance=args.tolerance,
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

