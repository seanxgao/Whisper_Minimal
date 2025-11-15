"""Dataset for training tiny VAD model."""

from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

from vad_distill.utils.audio_io import load_wav
from vad_distill.utils.features import log_mel

logger = logging.getLogger(__name__)


class TinyVADDataset(Dataset):
    """
    Dataset for training tiny frame-level VAD model.
    
    Loads wav files, computes log-mel features, and pairs them with
    teacher frame-level labels.
    """
    
    def __init__(
        self,
        manifest_path: str,
        frame_probs_dir: str,
        frame_hop: float = 0.01,
        frame_len: float = 0.025,
        n_mels: int = 80,
        cache_features: bool = False,
        cache_dir: str | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            manifest_path: Path to manifest file (JSONL or CSV)
            frame_probs_dir: Directory containing teacher frame probabilities (.npy files)
            frame_hop: Frame hop in seconds
            frame_len: Frame length in seconds
            n_mels: Number of mel filter banks
            cache_features: Whether to cache log-mel features
            cache_dir: Directory to cache features (if cache_features=True)
        """
        self.manifest_path = Path(manifest_path)
        self.frame_probs_dir = Path(frame_probs_dir)
        self.frame_hop = frame_hop
        self.frame_len = frame_len
        self.n_mels = n_mels
        self.cache_features = cache_features
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load manifest
        self.items = self._load_manifest()
        logger.info(f"Loaded {len(self.items)} items from manifest")
    
    def _load_manifest(self) -> list[Dict[str, str]]:
        """Load manifest file (JSONL or CSV format)."""
        items = []
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        
        with open(self.manifest_path, 'r', encoding='utf-8') as f:
            if self.manifest_path.suffix == '.jsonl':
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            elif self.manifest_path.suffix == '.csv':
                import csv
                reader = csv.DictReader(f)
                items = list(reader)
            else:
                raise ValueError(f"Unsupported manifest format: {self.manifest_path.suffix}")
        
        return items
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (mel_features, frame_labels) as tensors
        """
        item = self.items[idx]
        utt_id = item['utt_id']
        wav_path = item['wav_path']
        
        # Load or compute log-mel features
        mel_features = self._get_mel_features(utt_id, wav_path)
        
        # Load teacher frame labels
        frame_labels = self._load_frame_labels(utt_id)
        
        # Align lengths (truncate or pad if needed)
        min_len = min(len(mel_features), len(frame_labels))
        mel_features = mel_features[:min_len]
        frame_labels = frame_labels[:min_len]
        
        # Convert to tensors
        mel_tensor = torch.from_numpy(mel_features).float()
        label_tensor = torch.from_numpy(frame_labels).float()
        
        return mel_tensor, label_tensor
    
    def _get_mel_features(self, utt_id: str, wav_path: str) -> np.ndarray:
        """Load or compute log-mel features."""
        # Check cache
        if self.cache_features and self.cache_dir:
            cache_path = self.cache_dir / f"{utt_id}.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Load wav and compute features
        wav = load_wav(wav_path, target_sr=16000)
        mel_features = log_mel(
            wav,
            sr=16000,
            n_mels=self.n_mels,
            frame_len=self.frame_len,
            frame_hop=self.frame_hop,
        )
        
        # Cache if enabled
        if self.cache_features and self.cache_dir:
            cache_path = self.cache_dir / f"{utt_id}.npy"
            np.save(cache_path, mel_features)
        
        return mel_features
    
    def _load_frame_labels(self, utt_id: str) -> np.ndarray:
        """Load teacher frame-level labels."""
        label_path = self.frame_probs_dir / f"{utt_id}.npy"
        
        if not label_path.exists():
            raise FileNotFoundError(f"Frame labels not found: {label_path}")
        
        labels = np.load(label_path)
        
        # Ensure 1-D array
        if len(labels.shape) > 1:
            labels = labels.flatten()
        
        return labels.astype(np.float32)

