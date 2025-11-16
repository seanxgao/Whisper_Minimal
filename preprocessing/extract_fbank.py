"""Extract log-mel filterbank features from audio files."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import logging

from vad_distill.utils.audio_io import load_wav
from vad_distill.utils.features import log_mel
from preprocessing.chunk_config import FRAME_LEN, FRAME_HOP, N_MELS, SAMPLE_RATE

logger = logging.getLogger(__name__)


def extract_fbank(wav_path: str | Path, output_path: str | Path) -> int:
    """
    Extract log-mel filterbank features from audio file.
    
    Args:
        wav_path: Path to input audio file
        output_path: Path to save fbank features (.npy file)
    
    Returns:
        Number of frames extracted
    """
    wav_path = Path(wav_path)
    output_path = Path(output_path)
    
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load audio
    logger.debug(f"Loading audio from {wav_path}")
    wav = load_wav(str(wav_path), target_sr=SAMPLE_RATE)
    
    # Extract log-mel features
    fbank = log_mel(
        wav,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        frame_len=FRAME_LEN,
        frame_hop=FRAME_HOP,
    )
    
    # Validate shape: (T, N_MELS)
    if len(fbank.shape) != 2 or fbank.shape[1] != N_MELS:
        raise ValueError(
            f"Invalid fbank shape: {fbank.shape}, expected (T, {N_MELS})"
        )
    
    # Save to file
    np.save(output_path, fbank)
    num_frames = fbank.shape[0]
    
    logger.debug(f"Extracted {num_frames} frames, saved to {output_path}")
    
    return num_frames

