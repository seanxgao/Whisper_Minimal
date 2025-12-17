"""Run Tiny VAD on a single wav file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from vad_distill.config import load_config
from vad_distill.config.chunk_config import FRAME_HOP, N_MELS, SAMPLE_RATE
from vad_distill.config.data_paths import resolve_paths
from vad_distill.preprocessing.audio_utils import compute_log_mel, load_audio
from vad_distill.preprocessing.chunking import (
    chunk_fbank_features,
    reassemble_chunk_predictions,
)
from vad_distill.tiny_vad.model import build_tiny_vad_model


def _smooth(scores: np.ndarray, kernel: int = 5) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(scores, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
    return np.median(windows, axis=-1)


def _segments(scores: np.ndarray, threshold: float) -> List[Tuple[float, float]]:
    binary = scores > threshold
    segments: List[Tuple[float, float]] = []
    start = None
    for idx, val in enumerate(binary):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            segments.append((start * FRAME_HOP, idx * FRAME_HOP))
            start = None
    if start is not None:
        segments.append((start * FRAME_HOP, len(binary) * FRAME_HOP))
    return segments


def run_inference(
    wav_path: Path,
    model_path: Path,
    config_path: Path | None,
    use_onnx: bool,
) -> np.ndarray:
    wav = load_audio(wav_path, target_sr=SAMPLE_RATE)
    fbank = compute_log_mel(wav, sample_rate=SAMPLE_RATE, n_mels=N_MELS)
    chunks = chunk_fbank_features(fbank, pad_incomplete=True)

    if use_onnx:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path))
        preds = []
        for _, chunk in chunks:
            outputs = session.run(None, {"mel_features": chunk.astype(np.float32)[None, :, :]})[0]
            preds.append(torch.sigmoid(torch.from_numpy(outputs)).squeeze().numpy())
        return reassemble_chunk_predictions(chunks, preds)

    if config_path is None:
        raise ValueError("config path required for PyTorch checkpoints")
    config = load_config(str(config_path))
    model = build_tiny_vad_model(config)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    preds = []
    with torch.no_grad():
        for _, chunk in chunks:
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            logits = model(tensor)
            preds.append(torch.sigmoid(logits).squeeze(0).numpy())
    return reassemble_chunk_predictions(chunks, preds)


def test_single_wav(
    wav_path: Path,
    model_path: Path,
    output_dir: Path,
    config_path: Path | None,
    threshold: float,
    use_onnx: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = run_inference(wav_path, model_path, config_path, use_onnx)
    smoothed = _smooth(scores)
    segments = _segments(smoothed, threshold)

    scores_path = output_dir / f"{wav_path.stem}_scores.npy"
    seg_path = output_dir / f"{wav_path.stem}_segments.json"
    np.save(scores_path, smoothed.astype(np.float32))
    with open(seg_path, "w", encoding="utf-8") as handle:
        json.dump(segments, handle, indent=2)

    speech_time = sum(end - start for start, end in segments)
    total_time = len(smoothed) * FRAME_HOP
    print(
        json.dumps(
            {
                "wav": wav_path.name,
                "segments": len(segments),
                "speech_seconds": round(speech_time, 2),
                "speech_ratio": round(speech_time / total_time, 4),
                "scores_path": str(scores_path),
                "segments_path": str(seg_path),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tiny VAD on a wav file.")
    parser.add_argument("wav_path")
    parser.add_argument("model_path")
    parser.add_argument(
        "--output-dir",
        help="Directory to store inference logs (defaults to config.paths.logs_dir/single_wav).",
    )
    parser.add_argument("--config", help="Config path for PyTorch checkpoints.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--onnx", action="store_true", help="Use ONNX model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config.get("paths", {}))
    default_logs = paths.get("logs_dir", Path.cwd() / "logs") / "single_wav"
    output_dir = Path(args.output_dir) if args.output_dir else default_logs
    test_single_wav(
        wav_path=Path(args.wav_path),
        model_path=Path(args.model_path),
        output_dir=output_dir,
        config_path=Path(args.config) if args.config else None,
        threshold=args.threshold,
        use_onnx=args.onnx,
    )


if __name__ == "__main__":
    main()
