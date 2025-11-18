"""ONNX export for Tiny VAD student."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from vad_distill.config import load_config
from vad_distill.config.data_paths import ensure_dirs, resolve_paths
from vad_distill.tiny_vad.model import build_tiny_vad_model

LOGGER = logging.getLogger("vad_distill.export")


def export_onnx(config_path: str | None, checkpoint: str | None, output: str | None) -> Path:
    config = load_config(config_path)
    paths = resolve_paths(config.get("paths", {}))
    ensure_dirs(paths)

    checkpoint_dir = paths.get("checkpoint_dir")
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be defined in config.paths.")
    ckpt_path = Path(checkpoint) if checkpoint else checkpoint_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_tiny_vad_model(config)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    onnx_dir = paths.get("onnx_dir")
    if onnx_dir is None:
        raise ValueError("onnx_dir must be defined in config.paths.")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(output) if output else onnx_dir / "tiny_vad.onnx"

    dummy = torch.randn(1, 100, config.get("model", {}).get("n_mels", 80))
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        input_names=["mel_features"],
        output_names=["vad_logits"],
        opset_version=12,
    )
    LOGGER.info("Exported ONNX model to %s", onnx_path)

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        ref = model(dummy).detach().numpy()
        out = session.run(None, {"mel_features": dummy.numpy().astype(np.float32)})[0]
        diff = np.abs(ref - out).max()
        LOGGER.info("ONNX verification max diff %.2e", diff)
    except Exception as exc:  # pragma: no cover - optional validation
        LOGGER.warning("Skipping ONNX runtime validation: %s", exc)

    return onnx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Tiny VAD to ONNX.")
    parser.add_argument("--config", help="Path to YAML config.")
    parser.add_argument("--checkpoint", help="Checkpoint path (defaults to best.pt).")
    parser.add_argument("--output", help="Output ONNX path.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    path = export_onnx(args.config, args.checkpoint, args.output)
    print(path)


if __name__ == "__main__":
    main()
