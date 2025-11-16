"""Feature extraction utilities for audio processing."""

from __future__ import annotations

import numpy as np
from typing import Optional

# Import default config from preprocessing (single source of truth)
try:
    from preprocessing.chunk_config import (
        SAMPLE_RATE, N_MELS, FRAME_LEN, FRAME_HOP
    )
    DEFAULT_SR = SAMPLE_RATE
    DEFAULT_N_MELS = N_MELS
    DEFAULT_FRAME_LEN = FRAME_LEN
    DEFAULT_FRAME_HOP = FRAME_HOP
except ImportError:
    # Fallback defaults if preprocessing not available
    DEFAULT_SR = 16000
    DEFAULT_N_MELS = 80
    DEFAULT_FRAME_LEN = 0.025
    DEFAULT_FRAME_HOP = 0.01


def log_mel(
    wav: np.ndarray,
    sr: int | None = None,
    n_mels: int | None = None,
    frame_len: float | None = None,
    frame_hop: float | None = None,
) -> np.ndarray:
    """
    Compute log-mel spectrogram from waveform.

    Args:
        wav: 1-D float32 waveform array
        sr: Sample rate in Hz (defaults to chunk_config.SAMPLE_RATE)
        n_mels: Number of mel filter banks (defaults to chunk_config.N_MELS)
        frame_len: Frame length in seconds (defaults to chunk_config.FRAME_LEN)
        frame_hop: Frame hop in seconds (defaults to chunk_config.FRAME_HOP)

    Returns:
        Log-mel spectrogram of shape (time, n_mels)
    """
    # Use defaults from chunk_config if not provided
    if sr is None:
        sr = DEFAULT_SR
    if n_mels is None:
        n_mels = DEFAULT_N_MELS
    if frame_len is None:
        frame_len = DEFAULT_FRAME_LEN
    if frame_hop is None:
        frame_hop = DEFAULT_FRAME_HOP
    
    try:
        import librosa
        
        # Compute mel spectrogram
        n_fft = int(frame_len * sr)
        hop_length = int(frame_hop * sr)
        
        mel_spec = librosa.feature.melspectrogram(
            y=wav,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=0,
            fmax=sr // 2,
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (time, freq)
        return log_mel_spec.T.astype(np.float32)
    except ImportError:
        # Fallback: TODO implement with scipy if librosa not available
        raise ImportError("librosa is required for log-mel computation. Install with: pip install librosa")

