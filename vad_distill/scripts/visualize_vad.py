"""Visualize VAD predictions on audio file."""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging

from vad_distill.utils.logging_utils import setup_logging
from vad_distill.utils.audio_io import load_wav
from vad_distill.utils.features import log_mel
from preprocessing.chunk_config import (
    FRAME_LEN, FRAME_HOP, N_MELS, SAMPLE_RATE
)

logger = logging.getLogger(__name__)


def visualize_vad(
    wav_path: str | Path,
    scores_path: str | Path | None = None,
    segments_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    """
    Visualize VAD predictions.
    
    Args:
        wav_path: Path to input WAV file
        scores_path: Path to frame scores .npy file (optional)
        segments_path: Path to segments .json file (optional)
        output_path: Path to save visualization (optional)
    """
    wav_path = Path(wav_path)
    
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    
    # Load audio
    wav = load_wav(str(wav_path), target_sr=SAMPLE_RATE)
    duration = len(wav) / SAMPLE_RATE
    
    # Load scores if provided
    frame_scores = None
    if scores_path:
        scores_path = Path(scores_path)
        if scores_path.exists():
            frame_scores = np.load(scores_path)
            logger.info(f"Loaded frame scores: {frame_scores.shape}")
    
    # Load segments if provided
    segments = None
    if segments_path:
        segments_path = Path(segments_path)
        if segments_path.exists():
            with open(segments_path, 'r') as f:
                segments = json.load(f)
            logger.info(f"Loaded {len(segments)} segments")
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot 1: Waveform
    time_axis = np.linspace(0, duration, len(wav))
    axes[0].plot(time_axis, wav, linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Waveform: {wav_path.name}')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Spectrogram (using unified config from chunk_config)
    fbank = log_mel(wav, sr=SAMPLE_RATE, n_mels=N_MELS, frame_len=FRAME_LEN, frame_hop=FRAME_HOP)
    time_frames = np.arange(fbank.shape[0]) * FRAME_HOP
    axes[1].imshow(
        fbank.T,
        aspect='auto',
        origin='lower',
        extent=[0, duration, 0, 80],
        cmap='viridis'
    )
    axes[1].set_ylabel('Mel Bin')
    axes[1].set_title('Log-Mel Spectrogram')
    
    # Plot 3: VAD Scores and Segments
    if frame_scores is not None:
        time_scores = np.arange(len(frame_scores)) * FRAME_HOP
        axes[2].plot(time_scores, frame_scores, label='VAD Score', linewidth=1)
        axes[2].axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        
        # Highlight speech segments
        if segments:
            for start, end in segments:
                axes[2].axvspan(start, end, alpha=0.3, color='green', label='Speech' if start == segments[0][0] else '')
    else:
        axes[2].text(0.5, 0.5, 'No VAD scores available', 
                    transform=axes[2].transAxes, ha='center')
    
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('VAD Score')
    axes[2].set_title('Voice Activity Detection')
    axes[2].set_ylim([0, 1])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Visualize VAD predictions")
    parser.add_argument('wav_path', type=str, help='Path to input WAV file')
    parser.add_argument('--scores', type=str, help='Path to frame scores .npy file')
    parser.add_argument('--segments', type=str, help='Path to segments .json file')
    parser.add_argument('--output', type=str, help='Path to save visualization')
    
    args = parser.parse_args()
    
    visualize_vad(
        wav_path=args.wav_path,
        scores_path=args.scores,
        segments_path=args.segments,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

