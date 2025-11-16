"""Chunking utilities for VAD inference."""

from __future__ import annotations

import numpy as np
from preprocessing.chunk_config import CHUNK_SIZE, CHUNK_STRIDE, N_MELS


def chunk_fbank_features(
    fbank: np.ndarray,
    pad_incomplete: bool = True,
) -> list[tuple[int, np.ndarray]]:
    """
    Chunk fbank features into fixed-size chunks (same logic as create_chunks.py).
    
    Args:
        fbank: Fbank features of shape (T, N_MELS)
        pad_incomplete: Whether to pad incomplete chunks at the end
    
    Returns:
        List of (start_frame, chunk) tuples where chunk is (CHUNK_SIZE, N_MELS)
    """
    chunks = []
    T = fbank.shape[0]
    
    if len(fbank.shape) != 2 or fbank.shape[1] != N_MELS:
        raise ValueError(
            f"Invalid fbank shape: {fbank.shape}, expected (T, {N_MELS})"
        )
    
    for start_idx in range(0, T, CHUNK_STRIDE):
        end_idx = start_idx + CHUNK_SIZE
        
        if end_idx > T:
            if pad_incomplete:
                # Pad incomplete chunk (for inference)
                chunk = np.zeros((CHUNK_SIZE, N_MELS), dtype=fbank.dtype)
                chunk[:T - start_idx, :] = fbank[start_idx:, :]
                chunks.append((start_idx, chunk))
            # Skip incomplete chunks (for training, matches create_chunks.py)
            break
        else:
            chunk = fbank[start_idx:end_idx, :]
            # Validate chunk shape
            if chunk.shape != (CHUNK_SIZE, N_MELS):
                raise ValueError(
                    f"Invalid chunk shape: {chunk.shape}, "
                    f"expected ({CHUNK_SIZE}, {N_MELS})"
                )
            chunks.append((start_idx, chunk))
    
    return chunks


def reassemble_chunk_predictions(
    chunks: list[tuple[int, np.ndarray]],
    predictions: list[np.ndarray],
) -> np.ndarray:
    """
    Reassemble chunk predictions into per-frame VAD scores.
    
    Handles overlapping chunks by averaging predictions.
    
    Args:
        chunks: List of (start_frame, chunk) tuples
        predictions: List of prediction arrays, each of shape (CHUNK_SIZE,)
    
    Returns:
        Per-frame VAD scores of shape (T,)
    """
    if len(chunks) != len(predictions):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(predictions)} predictions"
        )
    
    # Find total length
    max_frame = max(start + CHUNK_SIZE for start, _ in chunks)
    frame_scores = np.zeros(max_frame, dtype=np.float32)
    frame_counts = np.zeros(max_frame, dtype=np.int32)
    
    # Accumulate predictions (handle overlap by averaging)
    for (start_frame, _), pred in zip(chunks, predictions):
        if len(pred.shape) > 1:
            pred = pred.squeeze()
        if len(pred) != CHUNK_SIZE:
            raise ValueError(
                f"Invalid prediction length: {len(pred)}, expected {CHUNK_SIZE}"
            )
        
        end_frame = start_frame + CHUNK_SIZE
        frame_scores[start_frame:end_frame] += pred
        frame_counts[start_frame:end_frame] += 1
    
    # Average overlapping regions
    frame_scores = frame_scores / np.maximum(frame_counts, 1)
    
    return frame_scores

