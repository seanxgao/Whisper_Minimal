"""Audio I/O utilities for loading and resampling waveforms."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from typing import Optional


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load a WAV file and resample to target sample rate if needed.

    Args:
        path: Path to the WAV file
        target_sr: Target sample rate (default: 16000)

    Returns:
        Waveform as 1-D float32 numpy array, mono, at target_sr Hz
    """
    try:
        wav, sr = sf.read(path, dtype='float32')
        
        # Convert to mono if stereo
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            except ImportError:
                # Fallback to scipy if librosa not available
                from scipy import signal
                num_samples = int(len(wav) * target_sr / sr)
                wav = signal.resample(wav, num_samples)
        
        return wav.astype(np.float32)
    except Exception as e:
        raise IOError(f"Failed to load audio from {path}: {e}")

