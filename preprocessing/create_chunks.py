"""Create fixed-size chunks from frame-level features and labels."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import logging

from preprocessing.chunk_config import CHUNK_SIZE, CHUNK_STRIDE

logger = logging.getLogger(__name__)


def create_chunks(
    uid: str,
    fbank_path: str | Path,
    frame_probs_path: str | Path,
    output_dir: str | Path,
    start_chunk_id: int = 0,
) -> tuple[int, list[dict]]:
    """
    Create fixed-size chunks from frame-level features and labels.
    
    Args:
        uid: Unique identifier for the audio file
        fbank_path: Path to fbank.npy file (shape: T, 80)
        frame_probs_path: Path to frame_probs.npy file (shape: T,)
        output_dir: Directory to save chunk files
        start_chunk_id: Starting chunk ID for global numbering
    
    Returns:
        Tuple of (next_chunk_id, list of chunk metadata dicts)
    """
    fbank_path = Path(fbank_path)
    frame_probs_path = Path(frame_probs_path)
    output_dir = Path(output_dir)
    
    if not fbank_path.exists():
        raise FileNotFoundError(f"Fbank file not found: {fbank_path}")
    if not frame_probs_path.exists():
        raise FileNotFoundError(f"Frame probs file not found: {frame_probs_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load fbank and frame_probs
    fbank = np.load(fbank_path)  # (T, 80)
    frame_probs = np.load(frame_probs_path)  # (T,)
    
    # Validate shapes
    if len(fbank.shape) != 2 or fbank.shape[1] != 80:
        raise ValueError(f"Invalid fbank shape: {fbank.shape}, expected (T, 80)")
    if len(frame_probs.shape) != 1:
        raise ValueError(
            f"Invalid frame_probs shape: {frame_probs.shape}, expected (T,)"
        )
    
    # Force-align fbank and frame_probs by truncating to minimum length
    # Length mismatch (1-4 frames) is normal and expected due to teacher model's
    # internal context padding. This is a known issue in all VAD/ASR pipelines.
    T_fbank = fbank.shape[0]
    T_labels = len(frame_probs)
    
    if T_fbank != T_labels:
        frame_diff = abs(T_fbank - T_labels)
        
        # Log warning if mismatch is significant (>5 frames)
        if frame_diff > 5:
            logger.warning(
                f"Length mismatch for {uid}: fbank has {T_fbank} frames, "
                f"frame_probs has {T_labels} frames (difference: {frame_diff}). "
                f"Truncating to minimum length. This is normal for teacher models."
            )
        else:
            # Log debug for small mismatches (1-5 frames, which is expected)
            logger.debug(
                f"Length mismatch for {uid}: fbank has {T_fbank} frames, "
                f"frame_probs has {T_labels} frames (difference: {frame_diff}). "
                f"Truncating to minimum length."
            )
        
        # Always truncate both to minimum length (never skip, never raise exception)
        min_len = min(T_fbank, T_labels)
        fbank = fbank[:min_len, :]
        frame_probs = frame_probs[:min_len]
        T_fbank = min_len
        T_labels = min_len
    
    # Slide window to create chunks
    chunks_metadata = []
    chunk_id = start_chunk_id
    
    for start_idx in range(0, T_fbank, CHUNK_STRIDE):
        end_idx = start_idx + CHUNK_SIZE
        
        # Skip incomplete chunks at the end
        if end_idx > T_fbank:
            break
        
        # Extract chunk
        features = fbank[start_idx:end_idx, :]  # (100, 80)
        labels = frame_probs[start_idx:end_idx]  # (100,)
        
        # Validate chunk shapes
        assert features.shape == (CHUNK_SIZE, 80), (
            f"Invalid features shape: {features.shape}"
        )
        assert labels.shape == (CHUNK_SIZE,), (
            f"Invalid labels shape: {labels.shape}"
        )
        
        # Save chunk file
        chunk_filename = f"chunk_{chunk_id:06d}.npy"
        chunk_path = output_dir / chunk_filename
        
        chunk_data = {
            "features": features,
            "labels": labels,
            "uid": uid,
            "chunk_idx": start_idx // CHUNK_STRIDE,  # Index within audio file
        }
        
        np.save(chunk_path, chunk_data)
        
        # Store metadata
        chunks_metadata.append({
            "chunk_id": chunk_id,
            "chunk_filename": chunk_filename,
            "uid": uid,
            "chunk_idx": start_idx // CHUNK_STRIDE,
            "start_frame": start_idx,
            "end_frame": end_idx,
        })
        
        chunk_id += 1
    
    num_chunks = len(chunks_metadata)
    logger.debug(
        f"Created {num_chunks} chunks for {uid}, "
        f"chunk IDs {start_chunk_id} to {chunk_id - 1}"
    )
    
    return chunk_id, chunks_metadata


def create_metadata_json(
    all_chunks_metadata: list[dict],
    output_path: str | Path,
) -> None:
    """
    Create metadata.json file with chunk statistics.
    
    Args:
        all_chunks_metadata: List of all chunk metadata dicts
        output_path: Path to save metadata.json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Group chunks by uid
    uid_to_chunks = {}
    for chunk_meta in all_chunks_metadata:
        uid = chunk_meta['uid']
        if uid not in uid_to_chunks:
            uid_to_chunks[uid] = []
        uid_to_chunks[uid].append(chunk_meta)
    
    # Create metadata structure
    metadata = {
        "total_chunks": len(all_chunks_metadata),
        "uid_to_chunk_ranges": {}
    }
    
    for uid, chunks in uid_to_chunks.items():
        chunk_ids = sorted([c['chunk_id'] for c in chunks])
        metadata["uid_to_chunk_ranges"][uid] = {
            "num_chunks": len(chunks),
            "chunk_id_range": [min(chunk_ids), max(chunk_ids)],
            "chunk_ids": chunk_ids
        }
    
    # Save metadata
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved metadata to {output_path}")

