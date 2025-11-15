"""Utilities for aligning frame indices with time and converting segment boundaries."""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


def time_to_frame(time: float, frame_hop: float) -> int:
    """
    Convert time in seconds to frame index.

    Args:
        time: Time in seconds
        frame_hop: Frame hop in seconds

    Returns:
        Frame index (integer)
    """
    return int(time / frame_hop)


def frame_to_time(frame: int, frame_hop: float) -> float:
    """
    Convert frame index to time in seconds.

    Args:
        frame: Frame index
        frame_hop: Frame hop in seconds

    Returns:
        Time in seconds
    """
    return frame * frame_hop


def segments_to_frame_indices(
    segments: List[Tuple[float, float]],
    num_frames: int,
    frame_hop: float,
) -> List[Tuple[int, int]]:
    """
    Convert segment boundaries from seconds to frame indices.

    Args:
        segments: List of (start_time, end_time) tuples in seconds
        num_frames: Total number of frames
        frame_hop: Frame hop in seconds

    Returns:
        List of (start_frame, end_frame) tuples
    """
    frame_segments = []
    for start_time, end_time in segments:
        start_frame = max(0, time_to_frame(start_time, frame_hop))
        end_frame = min(num_frames, time_to_frame(end_time, frame_hop))
        if end_frame > start_frame:
            frame_segments.append((start_frame, end_frame))
    return frame_segments


def frame_indices_to_segments(
    frame_segments: List[Tuple[int, int]],
    frame_hop: float,
) -> List[Tuple[float, float]]:
    """
    Convert frame indices to time segments in seconds.

    Args:
        frame_segments: List of (start_frame, end_frame) tuples
        frame_hop: Frame hop in seconds

    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    return [
        (frame_to_time(start, frame_hop), frame_to_time(end, frame_hop))
        for start, end in frame_segments
    ]


def create_frame_labels(
    segments: List[Tuple[int, int]],
    num_frames: int,
) -> np.ndarray:
    """
    Create binary frame-level labels from segment boundaries.

    Args:
        segments: List of (start_frame, end_frame) tuples
        num_frames: Total number of frames

    Returns:
        Binary label array of shape (num_frames,), 1 for speech, 0 for non-speech
    """
    labels = np.zeros(num_frames, dtype=np.float32)
    for start_frame, end_frame in segments:
        labels[start_frame:end_frame] = 1.0
    return labels

