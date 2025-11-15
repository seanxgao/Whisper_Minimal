"""
Script to run teacher labeling over a manifest of audio files.

This script loads the teacher FSMN-VAD model and processes all audio files
listed in a manifest, saving frame-level probabilities and segment boundaries.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from vad_distill.utils.config import load_yaml
from vad_distill.utils.audio_io import load_wav
from vad_distill.utils.logging_utils import setup_logging
from vad_distill.teacher import TeacherFSMNVAD

logger = logging.getLogger(__name__)


def main():
    """Main entry point for teacher labeling script."""
    setup_logging()
    
    # Load configurations
    dataset_config = load_yaml("vad_distill/configs/dataset.yaml")
    teacher_config = load_yaml("vad_distill/configs/teacher_fsmn.yaml")
    
    # Initialize teacher model
    teacher_model_dir = teacher_config.get('teacher', {}).get('model_dir')
    teacher_device = teacher_config.get('teacher', {}).get('device', 'cpu')
    
    logger.info(f"Loading teacher model from {teacher_model_dir}")
    teacher = TeacherFSMNVAD(model_dir=teacher_model_dir, device=teacher_device)
    
    # Setup paths
    manifest_path = Path(dataset_config.get('dataset', {}).get('manifest_path'))
    sample_rate = dataset_config.get('dataset', {}).get('sample_rate', 16000)
    
    # Setup output directories (can be configured in dataset.yaml if needed)
    data_root = Path(dataset_config.get('dataset', {}).get('root', 'data/raw')).parent
    frame_probs_dir = data_root / "teacher_labels" / "frame_probs"
    segments_dir = data_root / "teacher_labels" / "segments"
    
    frame_probs_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    logger.info(f"Loading manifest from {manifest_path}")
    items = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        if manifest_path.suffix == '.jsonl':
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        elif manifest_path.suffix == '.csv':
            import csv
            reader = csv.DictReader(f)
            items = list(reader)
        else:
            raise ValueError(f"Unsupported manifest format: {manifest_path.suffix}")
    
    logger.info(f"Found {len(items)} items in manifest")
    
    # Process each item
    for item in tqdm(items, desc="Processing audio files"):
        utt_id = item['utt_id']
        wav_path = item['wav_path']
        
        try:
            # Load audio
            wav = load_wav(wav_path, target_sr=sample_rate)
            
            # Run teacher inference
            frame_prob, segments = teacher.infer(wav, sr=sample_rate)
            
            # Save frame probabilities
            frame_prob_path = frame_probs_dir / f"{utt_id}.npy"
            np.save(frame_prob_path, frame_prob)
            
            # Save segments
            segments_path = segments_dir / f"{utt_id}.json"
            with open(segments_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to process {utt_id} ({wav_path}): {e}")
            continue
    
    logger.info("Teacher labeling completed")


if __name__ == "__main__":
    main()

