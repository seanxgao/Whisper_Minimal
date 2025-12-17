"""Shared chunk configuration for preprocessing and inference."""

from __future__ import annotations

FRAME_LEN = 0.025  # 25 ms
FRAME_HOP = 0.01  # 10 ms

CHUNK_SIZE = 100
CHUNK_STRIDE = 50

N_MELS = 80
SAMPLE_RATE = 16_000

FEATURE_SHAPE = (CHUNK_SIZE, N_MELS)
LABEL_SHAPE = (CHUNK_SIZE,)

__all__ = [
    "FRAME_LEN",
    "FRAME_HOP",
    "CHUNK_SIZE",
    "CHUNK_STRIDE",
    "N_MELS",
    "SAMPLE_RATE",
    "FEATURE_SHAPE",
    "LABEL_SHAPE",
]
