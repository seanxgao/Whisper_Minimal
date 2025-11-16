"""Dataset for training tiny VAD model on fixed-size chunks."""

from __future__ import annotations

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple
import logging
import random

# Import chunk config - handle both direct import and relative import
try:
    from preprocessing.chunk_config import CHUNK_SIZE, N_MELS
except ImportError:
    # Fallback: define constants directly
    CHUNK_SIZE = 100
    N_MELS = 80

logger = logging.getLogger(__name__)


class TinyVADChunkDataset(Dataset):
    """
    Dataset for training tiny VAD model on preprocessed fixed-size chunks.
    
    This dataset loads preprocessed chunk files directly, with no audio loading,
    feature extraction, or teacher inference during training.
    
    Each chunk file contains:
    - features: np.ndarray of shape (100, 80)
    - labels: np.ndarray of shape (100,)
    """
    
    def __init__(
        self,
        chunks_dir: str | Path,
        index_file: str | Path | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        """
        Initialize the chunk dataset.
        
        Args:
            chunks_dir: Directory containing chunk_*.npy files
            index_file: Optional path to index.json file. If None, will scan
                       chunks_dir for all chunk_*.npy files
            shuffle: Whether to shuffle chunk order (default: True)
            seed: Random seed for shuffling (default: None)
        """
        self.chunks_dir = Path(chunks_dir)
        self.shuffle = shuffle
        
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
        
        # Load chunk index
        if index_file is not None:
            index_path = Path(index_file)
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            logger.info(f"Loading chunk index from {index_path}")
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Extract chunk filenames from index
            self.chunk_files = [
                self.chunks_dir / item['chunk_filename']
                for item in index_data
            ]
        else:
            # Scan directory for chunk files
            logger.info(f"Scanning {chunks_dir} for chunk files")
            self.chunk_files = sorted(
                self.chunks_dir.glob("chunk_*.npy")
            )
        
        # Validate all chunk files exist
        missing = [f for f in self.chunk_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} chunk files. First few: {missing[:5]}"
            )
        
        # Shuffle if requested
        if self.shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(self.chunk_files)
        
        logger.info(f"Loaded {len(self.chunk_files)} chunk files")
        if self.shuffle:
            logger.info("Chunk order is randomized")
    
    def __len__(self) -> int:
        """Return total number of chunks."""
        return len(self.chunk_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample (chunk).
        
        Args:
            idx: Chunk index
        
        Returns:
            Tuple of (features, labels) as tensors
            - features: torch.Tensor of shape (100, 80)
            - labels: torch.Tensor of shape (100,)
        
        Raises:
            ValueError: If chunk shape is invalid
            FileNotFoundError: If chunk file is missing
        """
        chunk_path = self.chunk_files[idx]
        
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
        
        try:
            # Load chunk file
            chunk_data = np.load(chunk_path, allow_pickle=True).item()
            
            # Extract features and labels
            features = chunk_data['features']  # (100, 80)
            labels = chunk_data['labels']      # (100,)
            
            # STRICT shape validation - fail loudly on mismatch
            if not isinstance(features, np.ndarray):
                raise ValueError(
                    f"Invalid features type in {chunk_path}: "
                    f"expected np.ndarray, got {type(features)}"
                )
            if not isinstance(labels, np.ndarray):
                raise ValueError(
                    f"Invalid labels type in {chunk_path}: "
                    f"expected np.ndarray, got {type(labels)}"
                )
            
            if features.shape != (CHUNK_SIZE, N_MELS):
                raise ValueError(
                    f"Invalid features shape in {chunk_path}: "
                    f"got {features.shape}, expected ({CHUNK_SIZE}, {N_MELS})"
                )
            if labels.shape != (CHUNK_SIZE,):
                raise ValueError(
                    f"Invalid labels shape in {chunk_path}: "
                    f"got {labels.shape}, expected ({CHUNK_SIZE},)"
                )
            
            # Check for NaN or Inf
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                raise ValueError(
                    f"NaN or Inf values found in features in {chunk_path}"
                )
            if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
                raise ValueError(
                    f"NaN or Inf values found in labels in {chunk_path}"
                )
            
            # Convert to tensors
            features_tensor = torch.from_numpy(features).float()
            labels_tensor = torch.from_numpy(labels).float()
            
            return features_tensor, labels_tensor
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load chunk from {chunk_path}: {e}"
            ) from e
