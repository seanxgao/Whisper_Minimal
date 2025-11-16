"""Main preprocessing pipeline: extract features, generate labels, create chunks."""

from __future__ import annotations

import json
import argparse
import os
import random
import sys
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, Any, List, Optional
from contextlib import redirect_stdout, redirect_stderr

from vad_distill.utils.logging_utils import setup_logging
from vad_distill.utils.config import load_yaml
from vad_distill.teacher import TeacherFSMNVAD
from preprocessing.extract_fbank import extract_fbank
from preprocessing.generate_teacher_labels import generate_teacher_labels
from preprocessing.create_chunks import create_chunks, create_metadata_json

logger = logging.getLogger(__name__)


def discover_wav_files_from_manifest(manifest_path: Path) -> List[Path]:
    """
    Discover all WAV files from manifest.
    
    Args:
        manifest_path: Path to manifest file (JSONL format)
    
    Returns:
        List of WAV file paths
    """
    all_wavs = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                wav_path = item.get('wav_path')
                if wav_path:
                    wav_path = Path(wav_path)
                    if wav_path.exists():
                        all_wavs.append(wav_path)
    return all_wavs


def apply_deterministic_sampling(
    all_wavs: List[Path],
    config: Dict[str, Any],
    output_root: Path,
) -> List[Path]:
    """
    Apply deterministic sampling with persistence.
    
    If sample_list exists, load it.
    If not, sample once and save it.
    
    Args:
        all_wavs: List of all discovered WAV file paths
        config: Configuration dictionary
        output_root: Root directory for output
    
    Returns:
        List of sampled WAV file paths
    """
    data_config = config.get('data', {})
    sample_list_path = data_config.get('sample_list', 'data/sample_list.txt')
    num_samples = data_config.get('num_samples', 2000)
    
    # Resolve sample_list path
    # If absolute, use as-is; if relative, resolve relative to output_root's parent (project root)
    if os.path.isabs(sample_list_path):
        sample_list_path = Path(sample_list_path)
    else:
        # Resolve relative to output_root's parent (assuming project root)
        # If output_root is "data", parent is project root
        # If output_root is absolute like "/path/to/data", parent is "/path/to"
        sample_list_path = output_root.resolve().parent / sample_list_path
    
    # Case A: sample_list exists - load it
    if sample_list_path.exists():
        with open(sample_list_path, 'r', encoding='utf-8') as f:
            wavs = [Path(line.strip()) for line in f.readlines() if line.strip()]
        print(f"Loaded fixed sample list from: {sample_list_path}")
        print(f"Using fixed subset: {len(wavs)} files")
        logger.info(f"Loaded fixed sample list from: {sample_list_path}")
        logger.info(f"Using fixed subset: {len(wavs)} files")
        return wavs
    
    # Case B: sample_list does NOT exist - sample once and save
    random.seed(42)
    sampled_wavs = random.sample(all_wavs, min(num_samples, len(all_wavs)))
    
    # Ensure directory exists
    os.makedirs(sample_list_path.parent, exist_ok=True)
    
    # Save sampled list
    with open(sample_list_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(str(wav) for wav in sampled_wavs))
    
    print(f"Created fixed sample list ({len(sampled_wavs)} files) at: {sample_list_path}")
    print(f"Using fixed subset: {len(sampled_wavs)} files")
    logger.info(f"Created fixed sample list ({len(sampled_wavs)} files) at: {sample_list_path}")
    logger.info(f"Using fixed subset: {len(sampled_wavs)} files")
    
    return sampled_wavs


def run_preprocessing_pipeline(
    manifest_path: str | Path,
    output_root: str | Path = "data",
    teacher_model_dir: str = "teacher",
    device: str = "cpu",
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str | Path] = None,
) -> dict:
    """
    Run complete preprocessing pipeline.
    
    Pipeline:
    1. Load manifest file
    2. Discover all WAV files and apply deterministic sampling (if configured)
    3. For each audio file:
       - Extract fbank → save to teacher_labels/{uid}/fbank.npy
       - Generate teacher labels → save to teacher_labels/{uid}/frame_probs.npy
       - Create chunks → save to chunks/chunk_*.npy
    4. Generate chunk index file
    
    Args:
        manifest_path: Path to manifest file (JSONL format)
        output_root: Root directory for output (default: data)
        teacher_model_dir: Directory containing teacher model
        device: Device for teacher inference ("cpu" or "cuda")
        config: Optional configuration dictionary. If None, will try to load from config_path
        config_path: Optional path to config file. Used if config is None
    
    Returns:
        Dictionary with statistics
    """
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    # Load config if not provided
    if config is None:
        if config_path is None:
            # Try to find default config
            default_config_path = Path(__file__).parent.parent.parent / "configs" / "student_tiny_vad.yaml"
            if default_config_path.exists():
                config_path = default_config_path
            else:
                config = {}
        else:
            config_path = Path(config_path)
            if config_path.exists():
                config = load_yaml(str(config_path))
            else:
                config = {}
    else:
        config = config.copy()
    
    # Setup output directories
    teacher_labels_dir = output_root / "teacher_labels"
    chunks_dir = output_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest and discover all WAV files
    logger.info(f"Loading manifest from {manifest_path}")
    items = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    
    logger.info(f"Loaded {len(items)} items from manifest")
    
    # Discover all WAV files from manifest
    all_wavs = discover_wav_files_from_manifest(manifest_path)
    logger.info(f"Discovered {len(all_wavs)} WAV files from manifest")
    
    # Apply deterministic sampling if configured
    data_config = config.get('data', {})
    if data_config.get('sample_list') is not None:
        sampled_wavs = apply_deterministic_sampling(all_wavs, config, output_root)
        # Create a set for fast lookup
        sampled_wavs_set = set(sampled_wavs)
        
        # Filter items to only include sampled WAV files
        items = [
            item for item in items
            if item.get('wav_path') and Path(item['wav_path']) in sampled_wavs_set
        ]
        logger.info(f"Filtered to {len(items)} items after sampling")
    else:
        logger.info("No sampling configuration found, using all files from manifest")
    
    # Initialize teacher model once (reuse for all files)
    logger.info(f"Initializing teacher model from {teacher_model_dir}")
    teacher_model = TeacherFSMNVAD(model_dir=teacher_model_dir, device=device)
    
    # Process each audio file
    all_chunks_metadata = []
    chunk_id = 0
    success_count = 0
    failed_count = 0
    
    logger.info("Starting preprocessing pipeline...")
    
    # Disable tqdm globally to suppress FunASR internal progress bars
    # Save original state
    tqdm_disable_state = getattr(tqdm, 'disable', False)
    tqdm.disable = True
    
    # Also set environment variable
    os.environ['TQDM_DISABLE'] = '1'
    
    try:
        for item in tqdm(items, desc="Processing audio files", disable=False):
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
            
            try:
                # Setup per-uid directory
                uid_dir = teacher_labels_dir / uid
                uid_dir.mkdir(parents=True, exist_ok=True)
                
                fbank_path = uid_dir / "fbank.npy"
                frame_probs_path = uid_dir / "frame_probs.npy"
                
                # Step 1: Extract fbank features
                if not fbank_path.exists():
                    logger.debug(f"Extracting fbank for {uid}")
                    extract_fbank(wav_path, fbank_path)
                else:
                    logger.debug(f"Fbank already exists for {uid}, skipping")
                
                # Step 2: Generate teacher labels
                # All output suppression is handled inside teacher_model.infer()
                if not frame_probs_path.exists():
                    logger.debug(f"Generating teacher labels for {uid}")
                    generate_teacher_labels(
                        wav_path, uid, uid_dir, teacher_model=teacher_model
                    )
                else:
                    logger.debug(f"Teacher labels already exist for {uid}, skipping")
                
                # Step 3: Create chunks
                logger.debug(f"Creating chunks for {uid}")
                chunk_id, chunks_metadata = create_chunks(
                    uid, fbank_path, frame_probs_path, chunks_dir, start_chunk_id=chunk_id
                )
                
                all_chunks_metadata.extend(chunks_metadata)
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process {uid}: {e}", exc_info=True)
                failed_count += 1
                continue
    finally:
        # Restore tqdm state
        tqdm.disable = tqdm_disable_state
        if 'TQDM_DISABLE' in os.environ:
            del os.environ['TQDM_DISABLE']
    
    # Save chunk index
    index_path = chunks_dir / "index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks_metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved chunk index to {index_path}")
    
    # Create metadata.json
    metadata_path = chunks_dir / "metadata.json"
    create_metadata_json(all_chunks_metadata, metadata_path)
    
    # Statistics
    stats = {
        "total_audio_files": len(items),
        "success_count": success_count,
        "failed_count": failed_count,
        "total_chunks": len(all_chunks_metadata),
        "chunks_dir": str(chunks_dir),
        "index_path": str(index_path),
    }
    
    # Print final summary (concise)
    print(f"\nPreprocessing complete. Total chunks: {stats['total_chunks']:,}")
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total audio files: {stats['total_audio_files']}")
    logger.info(f"Successfully processed: {stats['success_count']}")
    logger.info(f"Failed: {stats['failed_count']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Chunks directory: {stats['chunks_dir']}")
    logger.info(f"Index file: {stats['index_path']}")
    
    return stats


def main():
    """Main entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Run complete preprocessing pipeline"
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
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: vad_distill/configs/student_tiny_vad.yaml)'
    )
    
    args = parser.parse_args()
    
    stats = run_preprocessing_pipeline(
        manifest_path=args.manifest,
        output_root=args.output_root,
        teacher_model_dir=args.teacher_model_dir,
        device=args.device,
        config_path=args.config,
    )
    
    logger.info("Preprocessing pipeline completed successfully")


if __name__ == "__main__":
    main()

