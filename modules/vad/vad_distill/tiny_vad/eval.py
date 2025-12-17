"""Evaluation utility for Tiny VAD."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from vad_distill.config import load_config
from vad_distill.config.data_paths import resolve_paths
from vad_distill.tiny_vad.dataset import build_dataloader, build_datasets
from vad_distill.tiny_vad.model import build_tiny_vad_model

LOGGER = logging.getLogger("vad_distill.eval")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])


def evaluate(config_path: str | None, checkpoint: str | None) -> Dict[str, float]:
    config = load_config(config_path)
    seed = int(config.get("seed", 2025))
    paths = resolve_paths(config.get("paths", {}))
    training_cfg = config.get("training", {})

    _, val_dataset = build_datasets(config, seed, paths)
    val_loader = build_dataloader(val_dataset, training_cfg, shuffle=False)

    checkpoint_dir = paths.get("checkpoint_dir")
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be defined in config.paths.")
    ckpt_path = Path(checkpoint) if checkpoint else checkpoint_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tiny_vad_model(config).to(device)
    load_checkpoint(model, ckpt_path)
    model.eval()

    mse = torch.nn.MSELoss(reduction="sum")
    mae = torch.nn.L1Loss(reduction="sum")
    mse_total = 0.0
    mae_total = 0.0
    frames = 0
    with torch.no_grad():
        for features, teacher_probs in tqdm(val_loader, desc="eval"):
            features = features.float().to(device, non_blocking=True)
            teacher_probs = teacher_probs.float().to(device, non_blocking=True)
            logits = model(features)
            probs = torch.sigmoid(logits)
            mse_total += mse(probs, teacher_probs).item()
            mae_total += mae(probs, teacher_probs).item()
            frames += teacher_probs.numel()

    metrics = {
        "mse": mse_total / frames,
        "mae": mae_total / frames,
        "frames": frames,
    }
    LOGGER.info("Eval metrics: %s", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tiny VAD student.")
    parser.add_argument("--config", help="Path to YAML config.")
    parser.add_argument("--checkpoint", help="Checkpoint path (defaults to best.pt).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    metrics = evaluate(args.config, args.checkpoint)
    print(metrics)


if __name__ == "__main__":
    main()
