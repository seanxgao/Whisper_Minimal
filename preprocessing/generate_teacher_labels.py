"""Generate teacher model frame-level VAD labels."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import logging

from vad_distill.teacher import TeacherFSMNVAD
from vad_distill.utils.audio_io import load_wav
from preprocessing.chunk_config import SAMPLE_RATE, FRAME_LEN, FRAME_HOP

logger = logging.getLogger(__name__)


def generate_teacher_labels(
    wav_path: str | Path,
    uid: str,
    output_dir: str | Path,
    teacher_model: TeacherFSMNVAD | None = None,
) -> int:
    """
    Generate teacher model frame-level VAD probabilities.
    
    Args:
        wav_path: Path to input audio file
        uid: Unique identifier for this audio file
        output_dir: Directory to save frame_probs.npy
        teacher_model: Optional pre-initialized teacher model (for efficiency)
    
    Returns:
        Number of frames in frame_probs
    """
    wav_path = Path(wav_path)
    output_dir = Path(output_dir)
    
    if not wav_path.exists():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_probs_path = output_dir / "frame_probs.npy"
    
    # Load audio
    logger.debug(f"Loading audio from {wav_path}")
    wav = load_wav(str(wav_path), target_sr=SAMPLE_RATE)
    
    # Initialize teacher model if not provided
    if teacher_model is None:
        logger.debug("Initializing teacher model")
        teacher_model = TeacherFSMNVAD(model_dir="teacher", device="cpu")
    
    # Run teacher inference
    # All output suppression is handled inside teacher_model.infer()
    logger.debug(f"Running teacher inference for {uid}")
    frame_probs, segments = teacher_model.infer(wav, sr=SAMPLE_RATE)
    
    # Ensure frame_probs is 1-D array
    if len(frame_probs.shape) > 1:
        frame_probs = frame_probs.flatten()
    
    # Calculate expected frame count to match librosa melspectrogram output
    # librosa uses: num_frames = (len(wav) - n_fft) // hop_length + 1
    n_fft = int(FRAME_LEN * SAMPLE_RATE)  # 400 at 16kHz
    hop_length = int(FRAME_HOP * SAMPLE_RATE)  # 160 at 16kHz
    expected_frames = (len(wav) - n_fft) // hop_length + 1
    expected_frames = max(1, expected_frames)  # Ensure non-negative
    
    # Force-align frame_probs to expected fbank length by truncating to minimum
    # Length mismatch (1-4 frames) is normal and expected due to teacher model's
    # internal context padding. This is a known issue in all VAD/ASR pipelines.
    if len(frame_probs) != expected_frames:
        frame_diff = abs(len(frame_probs) - expected_frames)
        
        # Log warning if mismatch is significant (>5 frames)
        if frame_diff > 5:
            logger.warning(
                f"Frame count mismatch for {uid}: "
                f"frame_probs has {len(frame_probs)} frames, "
                f"expected {expected_frames} frames (difference: {frame_diff}). "
                f"Truncating to minimum length. This is normal for teacher models."
            )
        else:
            # Log debug for small mismatches (1-5 frames, which is expected)
            logger.debug(
                f"Frame count mismatch for {uid}: "
                f"frame_probs has {len(frame_probs)} frames, "
                f"expected {expected_frames} frames (difference: {frame_diff}). "
                f"Truncating to minimum length."
            )
        
        # Always truncate to minimum length (never pad, never skip, never raise exception)
        min_len = min(len(frame_probs), expected_frames)
        frame_probs = frame_probs[:min_len]
    
    # Validate: frame_probs should be float32 array
    frame_probs = frame_probs.astype(np.float32)
    
    # Save to file
    np.save(frame_probs_path, frame_probs)
    num_frames = len(frame_probs)
    
    logger.debug(
        f"Generated {num_frames} frame probabilities, saved to {frame_probs_path}"
    )
    
    return num_frames

