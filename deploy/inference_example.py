"""Example inference script for TinyVAD ONNX model."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from pathlib import Path
import json
from typing import Optional

from preprocessing.chunk_config import SAMPLE_RATE, N_MELS, FRAME_LEN, FRAME_HOP, CHUNK_SIZE


class TinyVADInference:
    """Simple inference wrapper for TinyVAD ONNX model."""
    
    def __init__(self, onnx_path: str | Path, config_path: str | Path | None = None):
        """
        Initialize TinyVAD inference.
        
        Args:
            onnx_path: Path to ONNX model file
            config_path: Optional path to config.json (for metadata)
        """
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        # Load ONNX model
        self.session = ort.InferenceSession(str(self.onnx_path))
        
        # Load config if provided
        self.config = {}
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
    
    def extract_features(self, wav: np.ndarray) -> np.ndarray:
        """
        Extract log-mel features from waveform.
        
        Args:
            wav: 1-D float32 waveform array at SAMPLE_RATE
        
        Returns:
            Log-mel spectrogram of shape (T, N_MELS)
        """
        try:
            import librosa
            
            n_fft = int(FRAME_LEN * SAMPLE_RATE)
            hop_length = int(FRAME_HOP * SAMPLE_RATE)
            
            mel_spec = librosa.feature.melspectrogram(
                y=wav,
                sr=SAMPLE_RATE,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=N_MELS,
                fmin=0,
                fmax=SAMPLE_RATE // 2,
            )
            
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            return log_mel_spec.T.astype(np.float32)
        except ImportError:
            raise ImportError("librosa is required. Install with: pip install librosa")
    
    def chunk_features(self, fbank: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """
        Chunk features into fixed-size chunks.
        
        Args:
            fbank: Features of shape (T, N_MELS)
        
        Returns:
            List of (start_frame, chunk) tuples
        """
        chunks = []
        T = fbank.shape[0]
        CHUNK_STRIDE = 50  # 50% overlap
        
        for start_idx in range(0, T, CHUNK_STRIDE):
            end_idx = start_idx + CHUNK_SIZE
            if end_idx > T:
                # Pad incomplete chunk
                chunk = np.zeros((CHUNK_SIZE, N_MELS), dtype=fbank.dtype)
                chunk[:T - start_idx, :] = fbank[start_idx:, :]
                chunks.append((start_idx, chunk))
                break
            else:
                chunk = fbank[start_idx:end_idx, :]
                chunks.append((start_idx, chunk))
        
        return chunks
    
    def infer(self, wav: np.ndarray) -> np.ndarray:
        """
        Run VAD inference on waveform.
        
        Args:
            wav: 1-D float32 waveform array at SAMPLE_RATE
        
        Returns:
            Per-frame VAD scores of shape (T,)
        """
        # Extract features
        fbank = self.extract_features(wav)
        
        # Chunk features
        chunks = self.chunk_features(fbank)
        
        # Run inference on each chunk
        predictions = []
        for start_frame, chunk in chunks:
            # ONNX inference
            outputs = self.session.run(None, {'mel_features': chunk[np.newaxis, :, :].astype(np.float32)})
            logits = outputs[0][0, :, 0]  # (100,)
            pred = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
            predictions.append(pred)
        
        # Reassemble predictions
        max_frame = max(start + CHUNK_SIZE for start, _ in chunks)
        frame_scores = np.zeros(max_frame, dtype=np.float32)
        frame_counts = np.zeros(max_frame, dtype=np.int32)
        
        for (start_frame, _), pred in zip(chunks, predictions):
            end_frame = start_frame + CHUNK_SIZE
            frame_scores[start_frame:end_frame] += pred
            frame_counts[start_frame:end_frame] += 1
        
        frame_scores = frame_scores / np.maximum(frame_counts, 1)
        
        return frame_scores
    
    def infer_streaming(self, wav_chunk: np.ndarray, state: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """
        Run streaming inference on a chunk of audio.
        
        Args:
            wav_chunk: Audio chunk (should be ~1 second at SAMPLE_RATE)
            state: Optional state dict for maintaining context
        
        Returns:
            Tuple of (frame_scores, updated_state)
        """
        # For simplicity, this processes each chunk independently
        # In a real streaming scenario, you'd maintain overlap buffers
        fbank = self.extract_features(wav_chunk)
        
        # Ensure we have at least one chunk
        if fbank.shape[0] < CHUNK_SIZE:
            # Pad to chunk size
            padded = np.zeros((CHUNK_SIZE, N_MELS), dtype=fbank.dtype)
            padded[:fbank.shape[0], :] = fbank
            fbank = padded
        
        # Take first chunk
        chunk = fbank[:CHUNK_SIZE, :]
        
        # Run inference
        outputs = self.session.run(None, {'mel_features': chunk[np.newaxis, :, :].astype(np.float32)})
        logits = outputs[0][0, :, 0]
        pred = 1.0 / (1.0 + np.exp(-logits))
        
        # Update state (for future: maintain overlap buffer)
        if state is None:
            state = {}
        
        return pred, state


def main():
    """Example usage."""
    import argparse
    from vad_distill.utils.audio_io import load_wav
    
    parser = argparse.ArgumentParser(description="TinyVAD inference example")
    parser.add_argument('wav_path', type=str, help='Path to input WAV file')
    parser.add_argument('--onnx', type=str, default='tiny_vad.onnx', help='Path to ONNX model')
    parser.add_argument('--config', type=str, help='Path to config.json')
    parser.add_argument('--output', type=str, help='Path to save output scores')
    
    args = parser.parse_args()
    
    # Load audio
    wav = load_wav(args.wav_path, target_sr=SAMPLE_RATE)
    
    # Initialize inference
    inference = TinyVADInference(args.onnx, args.config)
    
    # Run inference
    frame_scores = inference.infer(wav)
    
    # Save output
    if args.output:
        np.save(args.output, frame_scores)
        print(f"Saved scores to {args.output}")
    
    # Print summary
    print(f"Processed {len(wav) / SAMPLE_RATE:.2f} seconds of audio")
    print(f"Generated {len(frame_scores)} frame scores")
    print(f"Mean VAD score: {frame_scores.mean():.3f}")


if __name__ == "__main__":
    main()

