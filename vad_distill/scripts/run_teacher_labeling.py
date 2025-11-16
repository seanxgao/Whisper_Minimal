"""
Script to generate teacher model frame-level VAD labels.

This script processes audio files and generates frame-level VAD probabilities
using the teacher FSMN-VAD model. Output is saved in the new directory structure:
data/teacher_labels/{uid}/frame_probs.npy
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

from vad_distill.utils.logging_utils import setup_logging
from vad_distill.teacher import TeacherFSMNVAD
from vad_distill.utils.audio_io import load_wav
from preprocessing.extract_fbank import extract_fbank
from preprocessing.chunk_config import SAMPLE_RATE

logger = logging.getLogger(__name__)


def generate_labels_for_manifest(
    manifest_path: str | Path,
    output_root: str | Path = "data",
    teacher_model_dir: str = "teacher",
    device: str = "cpu",
    skip_existing: bool = True,
) -> dict:
    """
    Generate teacher labels for all audio files in manifest.
    
    Args:
        manifest_path: Path to manifest file (JSONL format)
        output_root: Root directory for output (default: data)
        teacher_model_dir: Directory containing teacher model
        device: Device for teacher inference ("cpu" or "cuda")
        skip_existing: If True, skip files that already have labels
    
    Returns:
        Dictionary with statistics
    """
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    # Setup versioned output directory
    version = "v1"  # Can be made configurable
    teacher_labels_dir = output_root / "teacher_labels" / version
    teacher_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata file
    metadata_path = teacher_labels_dir / "metadata.json"
    if not metadata_path.exists():
        from datetime import datetime
        from preprocessing.chunk_config import SAMPLE_RATE, N_MELS, FRAME_LEN, FRAME_HOP
        
        metadata = {
            "version": version,
            "teacher_model": "FSMN-VAD",
            "teacher_model_dir": str(teacher_model_dir),
            "feature_params": {
                "sample_rate": SAMPLE_RATE,
                "n_mels": N_MELS,
                "frame_len": FRAME_LEN,
                "frame_hop": FRAME_HOP,
            },
            "generation_date": datetime.now().isoformat(),
        }
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Created metadata file: {metadata_path}")
    
    # Load manifest
    logger.info(f"Loading manifest from {manifest_path}")
    items = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    
    logger.info(f"Loaded {len(items)} items from manifest")
    
    # Initialize teacher model
    logger.info(f"Initializing teacher model from {teacher_model_dir}")
    teacher_model = TeacherFSMNVAD(model_dir=teacher_model_dir, device=device)
    
    # Process each audio file
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    logger.info("Starting teacher labeling...")
    
    for item in tqdm(items, desc="Processing audio files"):
        uid = item['utt_id']
        wav_path = item.get('wav_path')
        
        if not wav_path:
            logger.warning(f"Missing wav_path for {uid}, skipping")
            failed_count += 1
            continue
        
        wav_path = Path(wav_path)
        if not wav_path.exists():
            logger.warning(f"Audio file not found: {wav_path}, skipping")
            failed_count += 1
            continue
        
        # Setup per-uid directory
        uid_dir = teacher_labels_dir / uid
        uid_dir.mkdir(parents=True, exist_ok=True)
        
        frame_probs_path = uid_dir / "frame_probs.npy"
        fbank_path = uid_dir / "fbank.npy"
        
        # Skip if already exists
        if skip_existing and frame_probs_path.exists() and fbank_path.exists():
            skipped_count += 1
            continue
        
        try:
            # Extract fbank features (for alignment validation)
            if not fbank_path.exists():
                logger.debug(f"Extracting fbank for {uid}")
                extract_fbank(wav_path, fbank_path)
            
            # Load fbank to get frame count
            fbank = np.load(fbank_path)
            expected_frames = fbank.shape[0]
            
            # Generate teacher labels
            if not frame_probs_path.exists():
                logger.debug(f"Generating teacher labels for {uid}")
                
                # Load audio
                wav = load_wav(str(wav_path), target_sr=SAMPLE_RATE)
                
                # Run teacher inference
                frame_probs, segments = teacher_model.infer(wav, sr=SAMPLE_RATE)
                
                # Ensure frame_probs is 1-D array
                if len(frame_probs.shape) > 1:
                    frame_probs = frame_probs.flatten()
                
                # Validate length matches fbank
                if len(frame_probs) != expected_frames:
                    logger.warning(
                        f"Frame count mismatch for {uid}: "
                        f"fbank has {expected_frames} frames, "
                        f"frame_probs has {len(frame_probs)} frames. "
                        f"Truncating/padding to match."
                    )
                    # Align lengths
                    if len(frame_probs) < expected_frames:
                        # Pad with zeros
                        frame_probs = np.pad(
                            frame_probs,
                            (0, expected_frames - len(frame_probs)),
                            'constant'
                        )
                    else:
                        # Truncate
                        frame_probs = frame_probs[:expected_frames]
                
                # Save frame probabilities
                frame_probs = frame_probs.astype(np.float32)
                np.save(frame_probs_path, frame_probs)
                
                logger.debug(
                    f"Generated {len(frame_probs)} frame probabilities for {uid}"
                )
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {uid}: {e}", exc_info=True)
            failed_count += 1
            continue
    
    # Statistics
    stats = {
        "total_audio_files": len(items),
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "teacher_labels_dir": str(teacher_labels_dir),
    }
    
    logger.info("=" * 60)
    logger.info("TEACHER LABELING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total audio files: {stats['total_audio_files']}")
    logger.info(f"Successfully processed: {stats['success_count']}")
    logger.info(f"Skipped (already exists): {stats['skipped_count']}")
    logger.info(f"Failed: {stats['failed_count']}")
    logger.info(f"Output directory: {stats['teacher_labels_dir']}")
    
    return stats


def main():
    """Main entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Generate teacher model frame-level VAD labels"
    )
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to manifest file (JSONL format)'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='data',
        help='Root directory for output (default: data)'
    )
    parser.add_argument(
        '--teacher_model_dir',
        type=str,
        default='teacher',
        help='Directory containing teacher model (default: teacher)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for teacher inference (default: cpu)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Recompute labels even if they already exist'
    )
    
    args = parser.parse_args()
    
    stats = generate_labels_for_manifest(
        manifest_path=args.manifest,
        output_root=args.output_root,
        teacher_model_dir=args.teacher_model_dir,
        device=args.device,
        skip_existing=not args.no_skip_existing,
    )
    
    logger.info("Teacher labeling completed successfully")


if __name__ == "__main__":
    main()
