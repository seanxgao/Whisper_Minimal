"""Visualization tool for Tiny VAD outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vad_distill.config import load_config
from vad_distill.config.chunk_config import FRAME_HOP, FRAME_LEN, N_MELS, SAMPLE_RATE
from vad_distill.config.data_paths import resolve_paths
from vad_distill.preprocessing.audio_utils import compute_log_mel, load_audio


def visualize(
    wav_path: Path,
    scores_path: Path | None,
    segments_path: Path | None,
    output_path: Path | None,
    show_plot: bool = False,
) -> None:
    wav = load_audio(wav_path, target_sr=SAMPLE_RATE)
    duration = len(wav) / SAMPLE_RATE
    fbank = compute_log_mel(
        wav, sample_rate=SAMPLE_RATE, n_mels=N_MELS, frame_len=FRAME_LEN, frame_hop=FRAME_HOP
    )

    scores = None
    segments = None
    if scores_path and scores_path.exists():
        scores = np.load(scores_path)
    if segments_path and segments_path.exists():
        with open(segments_path, "r", encoding="utf-8") as handle:
            segments = json.load(handle)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    time_axis = np.linspace(0, duration, len(wav))
    axes[0].plot(time_axis, wav, linewidth=0.5)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(wav_path.name)

    axes[1].imshow(
        fbank.T,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, N_MELS],
        cmap="viridis",
    )
    axes[1].set_ylabel("Mel bin")
    axes[1].set_title("Log-mel spectrogram")

    if scores is not None:
        time_scores = np.arange(len(scores)) * FRAME_HOP
        axes[2].plot(time_scores, scores, label="VAD")
        axes[2].axhline(y=0.5, color="r", linestyle="--", label="0.5")
        if segments:
            for start, end in segments:
                axes[2].axvspan(start, end, alpha=0.3, color="green")
    else:
        axes[2].text(0.5, 0.5, "No scores", transform=axes[2].transAxes, ha="center")
    axes[2].set_xlabel("Seconds")
    axes[2].set_ylabel("Probability")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show_plot or output_path is None:
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Tiny VAD outputs.")
    parser.add_argument("wav_path")
    parser.add_argument("--scores")
    parser.add_argument("--segments")
    parser.add_argument(
        "--output",
        help="Optional output path. Defaults to config.paths.logs_dir/visualizations/<wav>.png.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively after saving.",
    )
    parser.add_argument("--config", help="Optional config path for log directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config.get("paths", {}))
    default_output = (
        paths.get("logs_dir", Path.cwd() / "logs")
        / "visualizations"
        / f"{Path(args.wav_path).stem}.png"
    )
    visualize(
        wav_path=Path(args.wav_path),
        scores_path=Path(args.scores) if args.scores else None,
        segments_path=Path(args.segments) if args.segments else None,
        output_path=Path(args.output) if args.output else default_output,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
