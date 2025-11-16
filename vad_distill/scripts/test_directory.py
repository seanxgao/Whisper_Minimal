"""Test VAD model on a directory of WAV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import logging

from vad_distill.utils.logging_utils import setup_logging
from vad_distill.scripts.test_single_wav import test_single_wav

logger = logging.getLogger(__name__)


def test_directory(
    wav_dir: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    threshold: float = 0.5,
    use_onnx: bool = False,
) -> None:
    """Test VAD model on all WAV files in a directory."""
    wav_dir = Path(wav_dir)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    
    if not wav_dir.exists():
        raise FileNotFoundError(f"Directory not found: {wav_dir}")
    
    # Find all WAV files
    wav_files = list(wav_dir.glob("*.wav")) + list(wav_dir.glob("*.WAV"))
    logger.info(f"Found {len(wav_files)} WAV files")
    
    # Process each file
    for wav_file in wav_files:
        try:
            test_single_wav(
                wav_path=wav_file,
                model_path=model_path,
                output_dir=output_dir,
                config_path=config_path,
                threshold=threshold,
                use_onnx=use_onnx,
            )
        except Exception as e:
            logger.error(f"Failed to process {wav_file}: {e}", exc_info=True)
            continue


def main():
    """Main entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Test VAD model on directory of WAV files")
    parser.add_argument('wav_dir', type=str, help='Directory containing WAV files')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint or ONNX file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--config', type=str, help='Path to config file (required for PyTorch)')
    parser.add_argument('--threshold', type=float, default=0.5, help='VAD threshold')
    parser.add_argument('--onnx', action='store_true', help='Use ONNX model')
    
    args = parser.parse_args()
    
    test_directory(
        wav_dir=args.wav_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        config_path=args.config,
        threshold=args.threshold,
        use_onnx=args.onnx,
    )


if __name__ == "__main__":
    main()

