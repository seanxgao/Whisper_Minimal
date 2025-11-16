"""Test VAD model on a single WAV file."""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
import torch
import logging

from vad_distill.utils.logging_utils import setup_logging
from vad_distill.utils.audio_io import load_wav
from vad_distill.utils.features import log_mel
from vad_distill.utils.chunking import chunk_fbank_features, reassemble_chunk_predictions
from vad_distill.utils.postprocessing import postprocess_vad_scores
from vad_distill.distill.tiny_vad.model import build_tiny_vad_model
from preprocessing.chunk_config import (
    CHUNK_SIZE, FRAME_LEN, FRAME_HOP, SAMPLE_RATE, N_MELS
)

logger = logging.getLogger(__name__)




def test_single_wav(
    wav_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    threshold: float = 0.5,
    use_onnx: bool = False,
) -> None:
    """
    Test VAD model on a single WAV file.
    
    Args:
        wav_path: Path to input WAV file
        model_path: Path to model checkpoint or ONNX file
        output_dir: Directory to save outputs
        config_path: Path to config file (required for PyTorch model)
        threshold: VAD threshold for segment extraction
        use_onnx: Whether to use ONNX model
    """
    wav_path = Path(wav_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Processing {wav_path}")
    
    # Load audio
    wav = load_wav(str(wav_path), target_sr=SAMPLE_RATE)
    
    # Extract fbank (using unified config from chunk_config)
    fbank = log_mel(
        wav,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        frame_len=FRAME_LEN,
        frame_hop=FRAME_HOP,
    )
    logger.info(f"Extracted fbank: {fbank.shape}")
    
    # Chunk audio (using unified chunking logic)
    chunks = chunk_fbank_features(fbank, pad_incomplete=True)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if use_onnx:
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        logger.info("Loaded ONNX model")
    else:
        if config_path is None:
            raise ValueError("config_path is required for PyTorch model")
        
        from vad_distill.utils.config import load_yaml
        config = load_yaml(str(config_path))
        model = build_tiny_vad_model(config)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)
        logger.info("Loaded PyTorch model")
    
    # Run inference
    predictions = []
    
    with torch.no_grad() if not use_onnx else torch.no_grad():
        for start_frame, chunk in chunks:
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)  # (1, 100, 80)
            
            if use_onnx:
                outputs = session.run(
                    None,
                    {'mel_features': chunk.numpy().astype(np.float32)}
                )
                pred = torch.sigmoid(torch.from_numpy(outputs[0])).squeeze().numpy()
            else:
                chunk_tensor = chunk_tensor.to(device)
                logits = model(chunk_tensor)  # (1, 100, 1)
                pred = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()  # (100,)
            
            predictions.append(pred)
    
    # Reassemble predictions (using unified reassembly logic)
    frame_scores = reassemble_chunk_predictions(chunks, predictions)
    logger.info(f"Reassembled predictions: {frame_scores.shape}")
    
    # Post-process and extract segments
    smoothed_scores, segments = postprocess_vad_scores(
        frame_scores,
        smooth_method="median",
        threshold=threshold,
        use_hysteresis=False,
        use_hangover=False,
    )
    logger.info(f"Extracted {len(segments)} segments")
    
    # Save outputs
    output_name = wav_path.stem
    
    # Save frame scores (smoothed)
    scores_path = output_dir / f"{output_name}_scores.npy"
    np.save(scores_path, smoothed_scores)
    logger.info(f"Saved frame scores to {scores_path}")
    
    # Save segments
    segments_path = output_dir / f"{output_name}_segments.json"
    with open(segments_path, 'w') as f:
        json.dump(segments, f, indent=2)
    logger.info(f"Saved segments to {segments_path}")
    
    # Print summary
    print(f"\nResults for {wav_path.name}:")
    print(f"  Total duration: {len(frame_scores) * FRAME_HOP:.2f} seconds")
    print(f"  Speech segments: {len(segments)}")
    print(f"  Total speech time: {sum(end - start for start, end in segments):.2f} seconds")
    print(f"  Speech ratio: {sum(end - start for start, end in segments) / (len(frame_scores) * FRAME_HOP):.2%}")


def main():
    """Main entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Test VAD model on single WAV file")
    parser.add_argument('wav_path', type=str, help='Path to input WAV file')
    parser.add_argument('model_path', type=str, help='Path to model checkpoint or ONNX file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--config', type=str, help='Path to config file (required for PyTorch)')
    parser.add_argument('--threshold', type=float, default=0.5, help='VAD threshold')
    parser.add_argument('--onnx', action='store_true', help='Use ONNX model')
    
    args = parser.parse_args()
    
    test_single_wav(
        wav_path=args.wav_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        config_path=args.config,
        threshold=args.threshold,
        use_onnx=args.onnx,
    )


if __name__ == "__main__":
    main()

