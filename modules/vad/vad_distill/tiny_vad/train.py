"""Training script for the Tiny VAD student."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm

from vad_distill.config import load_config
from vad_distill.config.data_paths import ensure_dirs, resolve_paths
from vad_distill.tiny_vad.dataset import build_dataloader, build_datasets
from vad_distill.tiny_vad.model import build_tiny_vad_model

LOGGER = logging.getLogger("vad_distill.train")


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
    log_interval: int,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    total_grad_norm = 0.0
    batches = 0

    progress = tqdm(loader, desc="train", leave=False)
    for batch_idx, (features, teacher_probs) in enumerate(progress):
        features = features.float().to(device, non_blocking=True)
        teacher_probs = teacher_probs.float().to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(features)
        student_prob = torch.sigmoid(logits)
        loss = criterion(student_prob, teacher_probs)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_grad_norm += float(grad_norm)
        batches += 1
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
        if batch_idx % max(1, log_interval) == 0:
            print(f"[epoch {epoch}] batch={batch_idx} loss={loss.item():.6f}")

    avg_loss = total_loss / max(batches, 1)
    avg_grad = total_grad_norm / max(batches, 1)
    return avg_loss, avg_grad


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for features, teacher_probs in tqdm(loader, desc="val", leave=False):
            features = features.float().to(device, non_blocking=True)
            teacher_probs = teacher_probs.float().to(device, non_blocking=True)
            logits = model(features)
            student_prob = torch.sigmoid(logits)
            loss = criterion(student_prob, teacher_probs)
            total_loss += loss.item()
            batches += 1
    return total_loss / max(batches, 1)


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best: bool,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, checkpoint_dir / "last.pt")
    if best:
        torch.save(state, checkpoint_dir / "best.pt")


def _build_subset_sampler(length: int, subset_size: int, seed: int) -> SubsetRandomSampler:
    generator = torch.Generator().manual_seed(seed)
    subset_size = max(1, min(length, subset_size))
    indices = torch.randperm(length, generator=generator)[:subset_size].tolist()
    return SubsetRandomSampler(indices)


def _prepare_subset_loader(
    train_dataset: Subset,
    training_cfg: Dict,
    *,
    epoch_seed: int,
) -> DataLoader:
    subset_size = int(training_cfg.get("subset_size", 50000))
    sampler = _build_subset_sampler(len(train_dataset), subset_size, epoch_seed)
    return build_dataloader(train_dataset, training_cfg, shuffle=False, sampler=sampler)


def _prepare_config(config_source: str | Path | Dict[str, Any] | None) -> Dict[str, Any]:
    if config_source is None or isinstance(config_source, (str, Path)):
        return load_config(config_source)
    if isinstance(config_source, dict):
        return deepcopy(config_source)
    raise TypeError(f"Unsupported config source: {type(config_source)!r}")


def run_training(config_source: str | Path | Dict[str, Any] | None) -> Dict:
    config = _prepare_config(config_source)
    seed = int(config.get("seed", 2025))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    training = config.get("training", {})
    paths = resolve_paths(config.get("paths", {}))
    ensure_dirs(paths)

    train_ds, val_ds = build_datasets(config, seed, paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    val_loader = build_dataloader(val_ds, training, shuffle=False)

    model = build_tiny_vad_model(config).to(device)
    lr = float(training.get("learning_rate", 1e-3))
    wd = float(training.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    checkpoint_dir = paths.get("checkpoint_dir")
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be defined in config.paths.")
    epochs = int(training.get("epochs", 60))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_interval = int(training.get("log_interval", 100))

    early_cfg = training.get("early_stopping", {})
    use_early = bool(early_cfg.get("enabled", True))
    patience = int(early_cfg.get("patience", 3))
    subset_enabled = bool(training.get("use_random_subset", False))

    best_val = float("inf")
    no_improve = 0
    history = []
    base_train_loader = (
        None if subset_enabled else build_dataloader(train_ds, training, shuffle=True)
    )

    for epoch in range(1, epochs + 1):
        if subset_enabled:
            train_loader = _prepare_subset_loader(
                train_ds,
                training,
                epoch_seed=seed + epoch,
            )
        else:
            train_loader = base_train_loader
        train_loss, grad_norm = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_grad_norm,
            log_interval,
            epoch,
        )
        val_loss = evaluate(model, val_loader, device)
        lr_current = optimizer.param_groups[0]["lr"]
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
        save_checkpoint(checkpoint_dir, model, optimizer, epoch, is_best)
        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr_current,
            "grad_norm": grad_norm,
        }
        history.append(epoch_stats)
        LOGGER.info(
            "Epoch %d | loss %.4f | val %.4f | lr %.2e | grad %.4f | subset=%s",
            epoch,
            train_loss,
            val_loss,
            lr_current,
            grad_norm,
            subset_enabled,
        )
        if use_early and no_improve >= patience:
            print("Early stopping triggered.")
            break

    logs_dir = paths.get("logs_dir")
    if logs_dir is None:
        raise ValueError("logs_dir must be defined in config.paths.")
    history_dir = logs_dir / "training"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{checkpoint_dir.name}_train_history.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    return {"best_val": best_val, "history_path": str(history_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tiny VAD student.")
    parser.add_argument("--config", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    summary = run_training(args.config)
    print(summary)


if __name__ == "__main__":
    main()
