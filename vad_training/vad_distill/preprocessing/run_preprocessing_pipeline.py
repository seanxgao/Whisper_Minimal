"""End-to-end preprocessing pipeline for Tiny VAD."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from vad_distill.config import load_config, resolve_path
from vad_distill.config.data_paths import (
    CHUNKS_DIR,
    RAW_DIR,
    TEACHER_PROBS_DIR,
    ensure_dirs,
)
from vad_distill.preprocessing.audio_utils import (
    DEFAULT_AUDIO_EXTS,
    compute_log_mel,
    load_audio,
)
from vad_distill.preprocessing.chunking import create_chunk_dataset, write_metadata
from vad_distill.teacher.frame_level_teacher import FrameLevelTeacher
from vad_distill.teacher.teacher_silero import load_silero

LOGGER = logging.getLogger("vad_distill.preprocess")


def _resolve_data_path(path_value: str | None, fallback: Path, base: Path | None = None) -> Path:
    if path_value:
        return resolve_path(path_value, base=base)
    return fallback


def discover_audio(root: Path, extensions: List[str] | None) -> List[Path]:
    """List all audio files under root."""
    if extensions:
        exts = {ext.lower() for ext in extensions}
    else:
        exts = set(DEFAULT_AUDIO_EXTS)
    files: List[Path] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in exts:
            files.append(candidate)
    return files


def ensure_teacher_probs(
    wav_path: Path,
    wav_root: Path,
    teacher_dir: Path,
    teacher: FrameLevelTeacher | None,
) -> np.ndarray:
    """
    Load or compute teacher probabilities for a waveform.
    """
    relative = wav_path.relative_to(wav_root)
    prob_path = teacher_dir / relative.with_suffix(".npy")
    if prob_path.exists():
        return np.load(prob_path).astype(np.float32)
    if teacher is None:
        raise FileNotFoundError(f"Missing teacher probabilities for {relative}")
    prob_path.parent.mkdir(parents=True, exist_ok=True)
    probs = teacher.get_frame_probs(wav_path)
    np.save(prob_path, probs.astype(np.float32))
    return probs


def run_pipeline(args: argparse.Namespace) -> Dict[str, float]:
    """Execute preprocessing pipeline."""
    ensure_dirs()
    config = load_config(args.config)
    paths_cfg = config.get("paths", {})
    base_root = resolve_path(args.output_root) if args.output_root else None
    chunks_dir = _resolve_data_path(
        args.chunks_dir or paths_cfg.get("chunks_dir"),
        CHUNKS_DIR,
        base=base_root,
    )
    teacher_dir = _resolve_data_path(
        args.teacher_prob_dir or paths_cfg.get("teacher_prob_dir"),
        TEACHER_PROBS_DIR,
        base=base_root,
    )
    wav_root = _resolve_data_path(args.wav_dir or paths_cfg.get("wav_dir"), RAW_DIR)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)

    wavs = discover_audio(wav_root, args.extensions)
    if args.limit is not None:
        wavs = wavs[: args.limit]

    LOGGER.info("Discovered %d audio files under %s", len(wavs), wav_root)
    if not wavs:
        return {}

    teacher = None
    if args.compute_teacher:
        LOGGER.info("Loading Silero teacher once for offline labeling")
        model, device = load_silero(device=args.teacher_device)
        teacher = FrameLevelTeacher(
            device=str(device),
            model=model,
            window_size=args.window_size,
            hop_size=args.hop_size,
        )

    timings = {"embedding": 0.0, "teacher": 0.0, "chunk": 0.0, "io": 0.0}
    metadata: List[dict] = []
    next_chunk_id = 0

    for wav_path in tqdm(wavs, desc="Preprocessing wavs"):
        try:
            start = time.perf_counter()
            wav = load_audio(wav_path)
            timings["io"] += time.perf_counter() - start

            start = time.perf_counter()
            fbank = compute_log_mel(wav)
            timings["embedding"] += time.perf_counter() - start

            start = time.perf_counter()
            teacher_probs = ensure_teacher_probs(wav_path, wav_root, teacher_dir, teacher)
            timings["teacher"] += time.perf_counter() - start

            start = time.perf_counter()
            next_chunk_id, chunk_meta = create_chunk_dataset(
                uid=wav_path.stem,
                fbank=fbank,
                teacher_probs=teacher_probs,
                output_dir=chunks_dir,
                start_chunk_id=next_chunk_id,
            )
            timings["chunk"] += time.perf_counter() - start
            metadata.extend(chunk_meta)
        except Exception as exc:  # pragma: no cover - pipeline resilience
            LOGGER.error("Failed to process %s: %s", wav_path, exc)

    write_metadata(metadata, chunks_dir / "index.json")

    summary = {
        "num_audio": len(wavs),
        "num_chunks": len(metadata),
        **{f"{k}_seconds": round(v, 3) for k, v in timings.items()},
    }
    LOGGER.info("Pipeline summary: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standard preprocessing pipeline from wav to chunk dataset."
    )
    parser.add_argument(
        "--wav-dir",
        default=str(RAW_DIR),
        help="Directory with raw audio files (defaults to RAW_DIR).",
    )
    parser.add_argument("--output-root", help="Base directory for derived data.")
    parser.add_argument("--chunks-dir", help="Override chunk output directory.")
    parser.add_argument("--teacher-prob-dir", help="Override teacher probability directory.")
    parser.add_argument("--config", help="Path to YAML config (defaults to default_config.yaml).")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_AUDIO_EXTS),
        help="Audio extensions to scan.",
    )
    parser.add_argument("--limit", type=int, help="Optional limit on number of wav files.")
    parser.add_argument(
        "--compute-teacher",
        action="store_true",
        help="Run Silero teacher if offline probabilities are missing.",
    )
    parser.add_argument(
        "--teacher-device",
        default="auto",
        help="Teacher device (auto, cpu, cuda).",
    )
    parser.add_argument("--window-size", type=int, default=512, help="Silero window size.")
    parser.add_argument("--hop-size", type=int, default=160, help="Silero hop size.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    summary = run_pipeline(args)
    if summary:
        print(summary)


if __name__ == "__main__":
    main()
