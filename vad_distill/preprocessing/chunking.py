"""Chunk creation and aggregation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, cast

import numpy as np

from vad_distill.config.chunk_config import (
    CHUNK_SIZE,
    CHUNK_STRIDE,
    N_MELS,
)

Chunk = Tuple[int, np.ndarray]


def chunk_fbank_features(
    fbank: np.ndarray,
    pad_incomplete: bool = False,
    chunk_size: int = CHUNK_SIZE,
    stride: int = CHUNK_STRIDE,
) -> List[Chunk]:
    """
    Convert a (T, n_mels) log-mel matrix into overlapping fixed-size chunks.
    """
    if fbank.ndim != 2 or fbank.shape[1] != N_MELS:
        raise ValueError(f"Expected (T, {N_MELS}) fbank, got {fbank.shape}")

    chunks: List[Chunk] = []
    total_frames = fbank.shape[0]

    for start in range(0, total_frames, stride):
        end = start + chunk_size
        if end > total_frames and not pad_incomplete:
            break

        if end <= total_frames:
            chunk = fbank[start:end]
        else:
            chunk = np.zeros((chunk_size, N_MELS), dtype=fbank.dtype)
            chunk[: total_frames - start] = fbank[start:]
        chunks.append((start, chunk))

    return chunks


def create_chunk_dataset(
    uid: str,
    fbank: np.ndarray,
    teacher_probs: np.ndarray,
    output_dir: str | Path,
    start_chunk_id: int = 0,
    chunk_size: int = CHUNK_SIZE,
    stride: int = CHUNK_STRIDE,
) -> Tuple[int, List[dict]]:
    """
    Save chunk files ready for training.
    """
    if teacher_probs.ndim != 1:
        raise ValueError("teacher_probs must be 1-D")

    total_frames = min(fbank.shape[0], teacher_probs.shape[0])
    if total_frames < chunk_size:
        return start_chunk_id, []

    fbank = fbank[:total_frames]
    teacher_probs = teacher_probs[:total_frames]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: List[dict] = []
    chunk_id = start_chunk_id

    for start in range(0, total_frames - chunk_size + 1, stride):
        end = start + chunk_size
        features = fbank[start:end]
        labels = teacher_probs[start:end]

        chunk_path = output_dir / f"chunk_{chunk_id:06d}.npy"
        payload: Dict[str, Any] = {
            "uid": uid,
            "chunk_id": chunk_id,
            "features": features.astype(np.float32),
            "teacher_probs": labels.astype(np.float32),
            "hard_labels": (labels >= 0.5).astype(np.float32),
            "frame_start": start,
            "frame_end": end,
        }
        np.save(chunk_path, cast(Any, payload), allow_pickle=True)
        metadata.append(
            {
                "chunk_id": chunk_id,
                "chunk_filename": chunk_path.name,
                "uid": uid,
                "start_frame": start,
                "end_frame": end,
            }
        )
        chunk_id += 1

    return chunk_id, metadata


def reassemble_chunk_predictions(
    chunks: Sequence[Chunk],
    predictions: Sequence[np.ndarray],
    chunk_size: int = CHUNK_SIZE,
) -> np.ndarray:
    """
    Average overlapping chunk predictions back to per-frame probabilities.
    """
    if len(chunks) != len(predictions):
        raise ValueError("chunks and predictions must have same length")

    max_len = max(start + chunk_size for start, _ in chunks)
    scores = np.zeros(max_len, dtype=np.float32)
    counts = np.zeros(max_len, dtype=np.int32)

    for (start, _), probs in zip(chunks, predictions):
        flat = probs.reshape(-1)
        if flat.shape[0] != chunk_size:
            raise ValueError(f"Prediction length {flat.shape[0]} != chunk_size {chunk_size}")
        end = start + chunk_size
        scores[start:end] += flat
        counts[start:end] += 1

    np.maximum(counts, 1, out=counts)
    return scores / counts


def write_metadata(metadata: Iterable[dict], path: str | Path) -> None:
    """
    Persist chunk metadata to JSON.
    """
    import json

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(list(metadata), handle, indent=2)


__all__ = [
    "Chunk",
    "chunk_fbank_features",
    "create_chunk_dataset",
    "reassemble_chunk_predictions",
    "write_metadata",
]
