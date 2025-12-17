"""Dataset and loader utilities for chunk-based Tiny VAD training."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from vad_distill.config.chunk_config import CHUNK_SIZE, N_MELS
from vad_distill.config.data_paths import PATHS, resolve_paths

LOGGER = logging.getLogger(__name__)


def _default_chunks_dir() -> Path:
    path = PATHS.get("chunks_dir")
    if path is None:
        raise ValueError("chunks_dir must be defined in configuration paths.")
    return path


def _default_teacher_dir() -> Path | None:
    return PATHS.get("teacher_prob_dir")


class TinyVADChunkDataset(Dataset):
    """
    Loads fixed-length feature chunks with soft teacher labels.
    """

    def __init__(
        self,
        chunks_dir: str | Path | None = None,
        teacher_prob_dir: str | Path | None = None,
        max_samples: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        resolved_chunks = Path(chunks_dir) if chunks_dir else _default_chunks_dir()
        self.chunks_dir = resolved_chunks
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {resolved_chunks}")
        resolved_teacher = (
            teacher_prob_dir if teacher_prob_dir is not None else _default_teacher_dir()
        )
        self.teacher_prob_dir = Path(resolved_teacher) if resolved_teacher else None
        self._use_external_teacher = (
            self.teacher_prob_dir is not None and self.teacher_prob_dir != self.chunks_dir
        )
        self.chunk_files: List[Path] = sorted(self.chunks_dir.glob("chunk_*.npy"))
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.chunk_files)
        if max_samples is not None:
            self.chunk_files = self.chunk_files[:max_samples]
        LOGGER.info("Loaded %d chunk files from %s", len(self.chunk_files), self.chunks_dir)

    def __len__(self) -> int:
        return len(self.chunk_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_path = self.chunk_files[index]
        data = np.load(chunk_path, allow_pickle=True).item()
        features = data["features"]
        teacher_probs = data.get("teacher_probs")
        if teacher_probs is None and self._use_external_teacher:
            teacher_dir = self.teacher_prob_dir
            if teacher_dir is None:
                raise RuntimeError("teacher_prob_dir became None despite external flag.")
            alt_path = teacher_dir / chunk_path.name
            if alt_path.exists():
                teacher_probs = np.load(alt_path).astype(np.float32)
        if teacher_probs is None:
            teacher_probs = data.get("hard_labels")
            if teacher_probs is None:
                raise ValueError(f"No teacher labels available for {chunk_path}")
            LOGGER.warning(
                "Chunk %s missing soft labels, using hard labels fallback.", chunk_path.name
            )

        features = np.asarray(features, dtype=np.float32)
        teacher_probs = np.asarray(teacher_probs, dtype=np.float32)
        if features.shape != (CHUNK_SIZE, N_MELS):
            raise ValueError(f"Invalid feature shape {features.shape} in {chunk_path}")
        if teacher_probs.shape != (CHUNK_SIZE,):
            raise ValueError(f"Invalid label shape {teacher_probs.shape} in {chunk_path}")

        if np.isnan(features).any() or np.isnan(teacher_probs).any():
            raise ValueError(f"NaN detected in {chunk_path}")
        if np.isinf(features).any() or np.isinf(teacher_probs).any():
            raise ValueError(f"Inf detected in {chunk_path}")

        return torch.from_numpy(features), torch.from_numpy(teacher_probs)


def _limit_subset(subset: Subset, limit: int | None) -> Subset:
    if limit is None or limit >= len(subset):
        return subset
    indices = subset.indices[:limit]
    return Subset(subset.dataset, indices)


def build_datasets(
    config: Dict[str, Any],
    seed: int,
    paths: Dict[str, Path] | None = None,
) -> Tuple[Subset, Subset]:
    training_cfg = config.get("training", {})
    resolved_paths = resolve_paths(config.get("paths", {})) if paths is None else paths

    dataset = TinyVADChunkDataset(
        chunks_dir=resolved_paths.get("chunks_dir"),
        teacher_prob_dir=resolved_paths.get("teacher_prob_dir"),
        shuffle=True,
        seed=seed,
    )

    total_chunks = len(dataset)
    if total_chunks < 2:
        raise ValueError(
            "Dataset must contain at least two chunks to form train/val splits. "
            f"Found: {total_chunks}"
        )
    val_fraction = float(training_cfg.get("validation_split", 0.1))
    val_size = max(1, int(total_chunks * val_fraction))
    if val_size >= total_chunks:
        val_size = total_chunks - 1
    train_size = total_chunks - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_ds = _limit_subset(train_ds, training_cfg.get("limit_train_chunks"))
    val_ds = _limit_subset(val_ds, training_cfg.get("limit_val_chunks"))
    return train_ds, val_ds


def build_dataloader(
    dataset: Dataset,
    training_cfg: Dict[str, Any],
    *,
    shuffle: bool,
    sampler: torch.utils.data.Sampler[int] | None = None,
) -> DataLoader:
    batch_size = int(training_cfg.get("batch_size", 32))
    num_workers = int(training_cfg.get("num_workers", 12))
    prefetch_factor = int(training_cfg.get("prefetch_factor", 4))
    pin_memory = bool(training_cfg.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(training_cfg.get("persistent_workers", True)) and num_workers > 0

    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = shuffle
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)
