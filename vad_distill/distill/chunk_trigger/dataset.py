"""Dataset for training chunk trigger model."""

from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class ChunkTriggerDataset(Dataset):
    """
    Dataset for training chunk trigger model.
    
    Builds training samples by sliding a window over VAD probability sequences
    and labeling window centers based on proximity to segment boundaries.
    """
    
    def __init__(
        self,
        frame_probs_dir: str,
        segments_dir: str,
        frame_hop: float = 0.01,
        window_size: int = 50,
        positive_margin: float = 0.05,
        negative_margin: float = 0.2,
    ):
        """
        Initialize the dataset.

        Args:
            frame_probs_dir: Directory containing teacher frame probabilities (.npy files)
            segments_dir: Directory containing segment boundaries (.json files)
            frame_hop: Frame hop in seconds
            window_size: Number of frames in the input window
            positive_margin: Seconds around segment end to label as positive (trigger)
            negative_margin: Minimum distance from segment end to label as negative
        """
        self.frame_probs_dir = Path(frame_probs_dir)
        self.segments_dir = Path(segments_dir)
        self.frame_hop = frame_hop
        self.window_size = window_size
        self.positive_margin = positive_margin
        self.negative_margin = negative_margin
        
        # Find all utterance IDs
        self.utt_ids = self._find_utt_ids()
        logger.info(f"Found {len(self.utt_ids)} utterances")
        
        # Build samples by sliding windows
        self.samples = self._build_samples()
        logger.info(f"Built {len(self.samples)} training samples")
    
    def _find_utt_ids(self) -> List[str]:
        """Find all utterance IDs from frame_probs directory."""
        utt_ids = []
        for npy_file in self.frame_probs_dir.glob("*.npy"):
            utt_id = npy_file.stem
            # Check if corresponding segment file exists
            seg_file = self.segments_dir / f"{utt_id}.json"
            if seg_file.exists():
                utt_ids.append(utt_id)
        return sorted(utt_ids)
    
    def _build_samples(self) -> List[Tuple[str, int, int]]:
        """
        Build training samples by sliding windows.
        
        Returns:
            List of (utt_id, window_start_frame, label) tuples
        """
        samples = []
        
        for utt_id in self.utt_ids:
            # Load frame probabilities
            frame_probs_path = self.frame_probs_dir / f"{utt_id}.npy"
            frame_probs = np.load(frame_probs_path)
            if len(frame_probs.shape) > 1:
                frame_probs = frame_probs.flatten()
            
            num_frames = len(frame_probs)
            
            # Load segment boundaries
            segments_path = self.segments_dir / f"{utt_id}.json"
            with open(segments_path, 'r', encoding='utf-8') as f:
                segments_data = json.load(f)
            
            # Parse segments (assume format: list of [start_time, end_time] or dict with 'segments' key)
            if isinstance(segments_data, list):
                segments = segments_data
            elif isinstance(segments_data, dict):
                segments = segments_data.get('segments', [])
            else:
                segments = []
            
            # Convert segment times to frame indices
            segment_end_frames = [
                int(end_time / self.frame_hop)
                for start_time, end_time in segments
            ]
            
            # Slide window and create samples
            for window_start in range(num_frames - self.window_size + 1):
                window_center = window_start + self.window_size // 2
                
                # Check if window center is near a segment end
                is_positive = False
                for seg_end_frame in segment_end_frames:
                    center_time = window_center * self.frame_hop
                    seg_end_time = seg_end_frame * self.frame_hop
                    distance = abs(center_time - seg_end_time)
                    
                    if distance <= self.positive_margin:
                        is_positive = True
                        break
                    elif distance < self.negative_margin:
                        # Skip ambiguous samples
                        break
                
                label = 1 if is_positive else 0
                samples.append((utt_id, window_start, label))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (vad_window, label) as tensors
        """
        utt_id, window_start, label = self.samples[idx]
        
        # Load frame probabilities
        frame_probs_path = self.frame_probs_dir / f"{utt_id}.npy"
        frame_probs = np.load(frame_probs_path)
        if len(frame_probs.shape) > 1:
            frame_probs = frame_probs.flatten()
        
        # Extract window
        window_end = window_start + self.window_size
        vad_window = frame_probs[window_start:window_end]
        
        # Pad if needed (shouldn't happen, but safety check)
        if len(vad_window) < self.window_size:
            padding = np.zeros(self.window_size - len(vad_window), dtype=vad_window.dtype)
            vad_window = np.concatenate([vad_window, padding])
        
        # Reshape to (window_size, 1)
        vad_window = vad_window.reshape(-1, 1)
        
        # Convert to tensors
        vad_tensor = torch.from_numpy(vad_window).float()
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return vad_tensor, label_tensor

