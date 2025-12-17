"""Offline Silero teacher labeling."""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from vad_distill.config.data_paths import RAW_DIR, TEACHER_PROBS_DIR, ensure_dirs
from vad_distill.preprocessing.audio_utils import DEFAULT_AUDIO_EXTS
from vad_distill.teacher.frame_level_teacher import FrameLevelTeacher


def discover_audio(root: Path, exts: List[str] | None) -> List[Path]:
    files: List[Path] = []
    if exts:
        extensions = {ext.lower() for ext in exts}
    else:
        extensions = set(DEFAULT_AUDIO_EXTS)
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in extensions:
            files.append(candidate)
    return files


def write_metadata(
    output_dir: Path,
    stats: Dict[str, int | float | str],
    files: List[Dict[str, str | int]],
) -> None:
    data = {
        "teacher": "silero-vad",
        "silero_version": metadata.version("silero-vad"),
        **stats,
        "files": files,
    }
    meta_path = output_dir / "teacher_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Silero teacher probabilities.")
    parser.add_argument(
        "--wav-dir",
        default=str(RAW_DIR),
        help="Directory containing audio files (defaults to RAW_DIR).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(TEACHER_PROBS_DIR),
        help="Directory to store teacher probability npy files (defaults to TEACHER_PROBS_DIR).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_AUDIO_EXTS),
        help="Audio extensions to include.",
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=160)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    ensure_dirs()
    wav_root = Path(args.wav_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_audio(wav_root, args.extensions)
    if not files:
        print("No audio files found.")
        return

    teacher = FrameLevelTeacher(
        device=args.device,
        window_size=args.window_size,
        hop_size=args.hop_size,
    )

    stats = {"window_size": args.window_size, "hop_size": args.hop_size, "processed": 0}
    file_meta: List[Dict[str, str | int]] = []

    for wav_path in tqdm(files, desc="Teacher labeling"):
        rel = wav_path.relative_to(wav_root)
        out_path = output_dir / rel.with_suffix(".npy")
        if out_path.exists() and not args.overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        probs = teacher.get_frame_probs(wav_path)
        np.save(out_path, probs.astype(np.float32))
        stats["processed"] += 1
        file_meta.append({"relative_path": str(rel), "num_frames": int(len(probs))})

    write_metadata(output_dir, stats, file_meta)
    print(f"Teacher labeling complete. Saved metadata to {output_dir}")


if __name__ == "__main__":
    main()
