"""Post-processing utilities for VAD predictions."""

from __future__ import annotations

import numpy as np
from typing import Optional
from scipy import ndimage
from preprocessing.chunk_config import FRAME_HOP


def median_filter(
    frame_scores: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Apply median filter to smooth frame-level VAD scores.
    
    Args:
        frame_scores: Per-frame VAD scores of shape (T,)
        kernel_size: Size of median filter kernel (must be odd)
    
    Returns:
        Smoothed frame scores
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd
    
    return ndimage.median_filter(frame_scores, size=kernel_size)


def hangover_scheme(
    frame_scores: np.ndarray,
    threshold: float = 0.5,
    hangover_frames: int = 5,
) -> np.ndarray:
    """
    Apply hangover scheme to extend speech segments.
    
    After a speech segment ends, extend it by hangover_frames frames
    if the scores are still above a lower threshold.
    
    Args:
        frame_scores: Per-frame VAD scores of shape (T,)
        threshold: Primary threshold for speech detection
        hangover_frames: Number of frames to extend segments
    
    Returns:
        Binary speech/non-speech array
    """
    binary = (frame_scores > threshold).astype(np.float32)
    
    # Extend segments by hangover_frames
    for i in range(len(binary) - 1):
        if binary[i] == 1.0 and binary[i + 1] == 0.0:
            # End of segment, check if we should extend
            end_idx = min(i + 1 + hangover_frames, len(binary))
            for j in range(i + 1, end_idx):
                if frame_scores[j] > threshold * 0.7:  # Lower threshold for hangover
                    binary[j] = 1.0
    
    return binary


def hysteresis_threshold(
    frame_scores: np.ndarray,
    high_threshold: float = 0.6,
    low_threshold: float = 0.4,
) -> np.ndarray:
    """
    Apply hysteresis thresholding to reduce flickering.
    
    Uses two thresholds:
    - High threshold: Start speech segment
    - Low threshold: End speech segment
    
    Args:
        frame_scores: Per-frame VAD scores of shape (T,)
        high_threshold: Threshold to start speech segment
        low_threshold: Threshold to end speech segment
    
    Returns:
        Binary speech/non-speech array
    """
    binary = np.zeros_like(frame_scores, dtype=np.float32)
    in_speech = False
    
    for i, score in enumerate(frame_scores):
        if not in_speech:
            if score >= high_threshold:
                in_speech = True
                binary[i] = 1.0
        else:
            if score < low_threshold:
                in_speech = False
                binary[i] = 0.0
            else:
                binary[i] = 1.0
    
    return binary


def smooth_frame_probs(
    frame_scores: np.ndarray,
    method: str = "median",
    **kwargs,
) -> np.ndarray:
    """
    Smooth frame-level probabilities using various methods.
    
    Args:
        frame_scores: Per-frame VAD scores of shape (T,)
        method: Smoothing method ('median', 'gaussian', 'moving_average')
        **kwargs: Additional arguments for smoothing method
    
    Returns:
        Smoothed frame scores
    """
    if method == "median":
        kernel_size = kwargs.get("kernel_size", 5)
        return median_filter(frame_scores, kernel_size=kernel_size)
    
    elif method == "gaussian":
        sigma = kwargs.get("sigma", 1.0)
        return ndimage.gaussian_filter1d(frame_scores, sigma=sigma)
    
    elif method == "moving_average":
        window_size = kwargs.get("window_size", 5)
        kernel = np.ones(window_size) / window_size
        return np.convolve(frame_scores, kernel, mode='same')
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def extract_segments(
    frame_scores: np.ndarray,
    threshold: float = 0.5,
    min_duration: float = 0.1,
    use_hysteresis: bool = False,
    high_threshold: float = 0.6,
    low_threshold: float = 0.4,
    use_hangover: bool = False,
    hangover_frames: int = 5,
) -> list[tuple[float, float]]:
    """
    Extract speech segments from frame-level scores with post-processing.
    
    Args:
        frame_scores: Per-frame VAD scores of shape (T,)
        threshold: Threshold for speech detection
        min_duration: Minimum segment duration in seconds
        use_hysteresis: Whether to use hysteresis thresholding
        high_threshold: High threshold for hysteresis (if enabled)
        low_threshold: Low threshold for hysteresis (if enabled)
        use_hangover: Whether to use hangover scheme
        hangover_frames: Number of frames for hangover (if enabled)
    
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    # Apply post-processing
    if use_hysteresis:
        binary = hysteresis_threshold(
            frame_scores,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
        )
    else:
        binary = (frame_scores > threshold).astype(np.float32)
    
    if use_hangover:
        binary = hangover_scheme(
            frame_scores,
            threshold=threshold,
            hangover_frames=hangover_frames,
        )
    
    # Extract segments
    segments = []
    in_segment = False
    start_frame = 0
    
    for i, speech in enumerate(binary):
        if speech > 0.5 and not in_segment:
            start_frame = i
            in_segment = True
        elif speech <= 0.5 and in_segment:
            end_frame = i
            duration = (end_frame - start_frame) * FRAME_HOP
            if duration >= min_duration:
                start_time = start_frame * FRAME_HOP
                end_time = end_frame * FRAME_HOP
                segments.append((start_time, end_time))
            in_segment = False
    
    # Handle segment at end
    if in_segment:
        end_frame = len(binary)
        duration = (end_frame - start_frame) * FRAME_HOP
        if duration >= min_duration:
            start_time = start_frame * FRAME_HOP
            end_time = end_frame * FRAME_HOP
            segments.append((start_time, end_time))
    
    return segments


def postprocess_vad_scores(
    frame_scores: np.ndarray,
    smooth_method: Optional[str] = "median",
    smooth_kwargs: Optional[dict] = None,
    threshold: float = 0.5,
    use_hysteresis: bool = False,
    high_threshold: float = 0.6,
    low_threshold: float = 0.4,
    use_hangover: bool = False,
    hangover_frames: int = 5,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Complete post-processing pipeline for VAD scores.
    
    Args:
        frame_scores: Raw per-frame VAD scores of shape (T,)
        smooth_method: Smoothing method ('median', 'gaussian', 'moving_average', None)
        smooth_kwargs: Additional arguments for smoothing
        threshold: Threshold for speech detection
        use_hysteresis: Whether to use hysteresis thresholding
        high_threshold: High threshold for hysteresis
        low_threshold: Low threshold for hysteresis
        use_hangover: Whether to use hangover scheme
        hangover_frames: Number of frames for hangover
    
    Returns:
        Tuple of (smoothed_scores, segments)
    """
    # Smooth scores if requested
    if smooth_method is not None:
        smooth_kwargs = smooth_kwargs or {}
        frame_scores = smooth_frame_probs(frame_scores, method=smooth_method, **smooth_kwargs)
    
    # Extract segments with post-processing
    segments = extract_segments(
        frame_scores,
        threshold=threshold,
        use_hysteresis=use_hysteresis,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        use_hangover=use_hangover,
        hangover_frames=hangover_frames,
    )
    
    return frame_scores, segments

