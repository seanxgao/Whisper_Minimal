"""Audio I/O utilities for loading and resampling waveforms.

This module provides a unified interface for loading audio files in various formats
(WAV, OPUS, FLAC, MP3, etc.) and converting them to a standard format for training
and inference. The resampling uses fast methods suitable for speech (not music),
prioritizing speed over extreme precision.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from typing import Optional


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load an audio file and convert to standard format for training/inference.
    
    This function provides a unified interface that works for both training (OPUS files)
    and inference (WAV files). It automatically handles format conversion, resampling,
    and mono conversion to ensure consistent input format.
    
    Args:
        path: Path to the audio file (supports WAV, OPUS, FLAC, MP3, etc.)
        target_sr: Target sample rate in Hz (default: 16000)
    
    Returns:
        Waveform as 1-D float32 numpy array, mono, at target_sr Hz.
        This format is consistent for both training and inference.
    
    Note:
        - Resampling uses fast methods suitable for speech (not high-fidelity music)
        - Stereo files are automatically converted to mono by averaging channels
        - The output format matches what the model expects during inference
    """
    try:
        # Try librosa first for better format support (OPUS, MP3, etc.)
        # This handles training data (OPUS) and inference data (WAV) uniformly
        try:
            import librosa
            # Load without resampling first to get original sample rate
            wav, sr = librosa.load(path, sr=None, mono=False, dtype='float32')
        except ImportError:
            # Fallback to soundfile for WAV/FLAC (common for inference)
            wav, sr = sf.read(path, dtype='float32')
            # soundfile returns (samples,) for mono, (samples, channels) for stereo
            if len(wav.shape) == 1:
                wav = wav.reshape(-1, 1)  # Make it (samples, 1) for consistency
        
        # Convert to mono if stereo (average channels)
        # This ensures consistent format regardless of input channels
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)
        
        # Resample if needed (fast resampling suitable for speech)
        if sr != target_sr:
            try:
                import librosa
                # Use librosa's resampling (fast, suitable for speech)
                # For speech, we don't need ultra-high precision resampling
                wav = librosa.resample(
                    wav,
                    orig_sr=sr,
                    target_sr=target_sr,
                    res_type='kaiser_fast'  # Fast resampling, good for speech
                )
            except ImportError:
                # Fallback to scipy if librosa not available
                from scipy import signal
                num_samples = int(len(wav) * target_sr / sr)
                wav = signal.resample(wav, num_samples)
        
        # Ensure float32 and 1-D array
        wav = wav.astype(np.float32)
        if len(wav.shape) > 1:
            wav = wav.flatten()
        
        return wav
    except Exception as e:
        raise IOError(f"Failed to load audio from {path}: {e}")

