"""Unit tests for chunk alignment correctness."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from preprocessing.create_chunks import create_chunks
from preprocessing.chunk_config import (
    CHUNK_SIZE, CHUNK_STRIDE, N_MELS, FRAME_HOP
)
from vad_distill.utils.chunking import chunk_fbank_features, reassemble_chunk_predictions


def test_chunk_creation_alignment():
    """Test that chunk creation aligns correctly with frame boundaries."""
    # Create dummy fbank and frame_probs
    T = 500  # 5 seconds at 100 fps
    fbank = np.random.randn(T, N_MELS).astype(np.float32)
    frame_probs = np.random.rand(T).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fbank_path = tmpdir / "fbank.npy"
        frame_probs_path = tmpdir / "frame_probs.npy"
        output_dir = tmpdir / "chunks"
        
        np.save(fbank_path, fbank)
        np.save(frame_probs_path, frame_probs)
        
        # Create chunks
        next_id, metadata = create_chunks(
            uid="test_audio",
            fbank_path=fbank_path,
            frame_probs_path=frame_probs_path,
            output_dir=output_dir,
        )
        
        # Verify chunk alignment
        expected_num_chunks = (T - CHUNK_SIZE) // CHUNK_STRIDE + 1
        if (T - CHUNK_SIZE) % CHUNK_STRIDE == 0:
            expected_num_chunks += 1
        
        assert len(metadata) > 0, "No chunks created"
        
        # Check that chunks are aligned correctly
        for i, chunk_meta in enumerate(metadata):
            start_frame = chunk_meta['start_frame']
            end_frame = chunk_meta['end_frame']
            
            # Verify chunk indices
            assert start_frame % CHUNK_STRIDE == 0, (
                f"Chunk {i} start_frame {start_frame} not aligned to CHUNK_STRIDE"
            )
            assert end_frame - start_frame == CHUNK_SIZE, (
                f"Chunk {i} size mismatch: {end_frame - start_frame} != {CHUNK_SIZE}"
            )
            assert end_frame <= T, (
                f"Chunk {i} extends beyond fbank length: {end_frame} > {T}"
            )


def test_fbank_frame_probs_length_match():
    """Test that fbank and frame_probs have matching lengths."""
    T = 300
    fbank = np.random.randn(T, N_MELS).astype(np.float32)
    frame_probs = np.random.rand(T).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fbank_path = tmpdir / "fbank.npy"
        frame_probs_path = tmpdir / "frame_probs.npy"
        output_dir = tmpdir / "chunks"
        
        np.save(fbank_path, fbank)
        np.save(frame_probs_path, frame_probs)
        
        # Should succeed
        next_id, metadata = create_chunks(
            uid="test",
            fbank_path=fbank_path,
            frame_probs_path=frame_probs_path,
            output_dir=output_dir,
        )
        assert len(metadata) > 0
    
    # Test mismatch - should raise error
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fbank_path = tmpdir / "fbank.npy"
        frame_probs_path = tmpdir / "frame_probs.npy"
        output_dir = tmpdir / "chunks"
        
        np.save(fbank_path, fbank)
        np.save(frame_probs_path, frame_probs[:T-10])  # Mismatch
        
        with pytest.raises(ValueError, match="Length mismatch"):
            create_chunks(
                uid="test",
                fbank_path=fbank_path,
                frame_probs_path=frame_probs_path,
                output_dir=output_dir,
            )


def test_chunk_slicing_accuracy():
    """Test that chunk slicing extracts correct frames."""
    T = 400
    fbank = np.random.randn(T, N_MELS).astype(np.float32)
    frame_probs = np.random.rand(T).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fbank_path = tmpdir / "fbank.npy"
        frame_probs_path = tmpdir / "frame_probs.npy"
        output_dir = tmpdir / "chunks"
        
        np.save(fbank_path, fbank)
        np.save(frame_probs_path, frame_probs)
        
        next_id, metadata = create_chunks(
            uid="test",
            fbank_path=fbank_path,
            frame_probs_path=frame_probs_path,
            output_dir=output_dir,
        )
        
        # Verify that chunks match original data
        for chunk_meta in metadata:
            chunk_file = output_dir / chunk_meta['chunk_filename']
            chunk_data = np.load(chunk_file, allow_pickle=True).item()
            
            features = chunk_data['features']
            labels = chunk_data['labels']
            start_frame = chunk_meta['start_frame']
            end_frame = chunk_meta['end_frame']
            
            # Verify features match
            expected_features = fbank[start_frame:end_frame, :]
            np.testing.assert_array_equal(
                features, expected_features,
                err_msg=f"Chunk features mismatch at frames {start_frame}:{end_frame}"
            )
            
            # Verify labels match
            expected_labels = frame_probs[start_frame:end_frame]
            np.testing.assert_array_equal(
                labels, expected_labels,
                err_msg=f"Chunk labels mismatch at frames {start_frame}:{end_frame}"
            )


def test_reconstruction_from_chunks():
    """Test that reconstruction from chunks returns exact frame count."""
    T = 350
    fbank = np.random.randn(T, N_MELS).astype(np.float32)
    frame_probs = np.random.rand(T).astype(np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        fbank_path = tmpdir / "fbank.npy"
        frame_probs_path = tmpdir / "frame_probs.npy"
        output_dir = tmpdir / "chunks"
        
        np.save(fbank_path, fbank)
        np.save(frame_probs_path, frame_probs)
        
        next_id, metadata = create_chunks(
            uid="test",
            fbank_path=fbank_path,
            frame_probs_path=frame_probs_path,
            output_dir=output_dir,
        )
        
        # Reconstruct fbank from chunks
        reconstructed_fbank = np.zeros((T, N_MELS), dtype=np.float32)
        frame_counts = np.zeros(T, dtype=np.int32)
        
        for chunk_meta in metadata:
            chunk_file = output_dir / chunk_meta['chunk_filename']
            chunk_data = np.load(chunk_file, allow_pickle=True).item()
            
            features = chunk_data['features']
            start_frame = chunk_meta['start_frame']
            end_frame = chunk_meta['end_frame']
            
            # Accumulate (handle overlap)
            reconstructed_fbank[start_frame:end_frame, :] += features
            frame_counts[start_frame:end_frame] += 1
        
        # Average overlapping regions
        frame_counts = np.maximum(frame_counts, 1)
        reconstructed_fbank = reconstructed_fbank / frame_counts[:, np.newaxis]
        
        # Verify reconstruction matches original (within numerical precision)
        # Note: overlapping regions will be averaged, so exact match only for non-overlapping
        # For this test, we check that non-overlapping regions match exactly
        for chunk_meta in metadata:
            start_frame = chunk_meta['start_frame']
            end_frame = chunk_meta['end_frame']
            
            # Check first chunk (no overlap from left)
            if start_frame == 0:
                np.testing.assert_allclose(
                    reconstructed_fbank[start_frame:end_frame, :],
                    fbank[start_frame:end_frame, :],
                    rtol=1e-5,
                    err_msg=f"Reconstruction mismatch in first chunk"
                )


def test_inference_chunking_consistency():
    """Test that inference chunking matches training chunking logic."""
    T = 450
    fbank = np.random.randn(T, N_MELS).astype(np.float32)
    
    # Use inference chunking
    inference_chunks = chunk_fbank_features(fbank, pad_incomplete=True)
    
    # Verify chunk alignment matches training logic
    for start_frame, chunk in inference_chunks:
        # Check alignment
        assert start_frame % CHUNK_STRIDE == 0, (
            f"Inference chunk start_frame {start_frame} not aligned"
        )
        
        # Check shape
        assert chunk.shape == (CHUNK_SIZE, N_MELS), (
            f"Inference chunk shape {chunk.shape} != ({CHUNK_SIZE}, {N_MELS})"
        )
        
        # Verify data matches (for non-padded chunks)
        end_frame = start_frame + CHUNK_SIZE
        if end_frame <= T:
            expected_chunk = fbank[start_frame:end_frame, :]
            np.testing.assert_array_equal(
                chunk, expected_chunk,
                err_msg=f"Inference chunk data mismatch at {start_frame}:{end_frame}"
            )


def test_reassemble_predictions():
    """Test that reassembly produces correct frame count."""
    T = 380
    fbank = np.random.randn(T, N_MELS).astype(np.float32)
    
    chunks = chunk_fbank_features(fbank, pad_incomplete=True)
    
    # Create dummy predictions
    predictions = []
    for start_frame, chunk in chunks:
        pred = np.random.rand(CHUNK_SIZE).astype(np.float32)
        predictions.append(pred)
    
    # Reassemble
    frame_scores = reassemble_chunk_predictions(chunks, predictions)
    
    # Verify length
    expected_length = max(start + CHUNK_SIZE for start, _ in chunks)
    assert len(frame_scores) == expected_length, (
        f"Reassembled length {len(frame_scores)} != expected {expected_length}"
    )
    
    # Verify all frames have scores (non-zero count)
    frame_counts = np.zeros(len(frame_scores), dtype=np.int32)
    for start_frame, _ in chunks:
        end_frame = start_frame + CHUNK_SIZE
        frame_counts[start_frame:end_frame] += 1
    
    assert np.all(frame_counts > 0), "Some frames have no scores after reassembly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

