"""Audio utilities for preprocessing and inference."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio

from vad_distill.config.chunk_config import (
    FRAME_HOP,
    FRAME_LEN,
    N_MELS,
    SAMPLE_RATE,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_AUDIO_EXTS: Iterable[str] = (
    ".wav",
    ".flac",
    ".opus",
    ".mp3",
    ".m4a",
    ".ogg",
)


def _torchaudio_decode(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0).numpy().astype(np.float32), sample_rate


def _ffmpeg_decode(path: Path, target_sr: int) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-",
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load an arbitrary audio file, convert to mono float32, and resample.

    Args:
        path: Audio file path.
        target_sr: Desired sampling rate.

    Returns:
        Float32 numpy array at target_sr Hz.
    """
    audio_path = Path(path)
    errors: list[str] = []

    try:
        wav, sr = _torchaudio_decode(audio_path)
        if sr != target_sr:
            wav = resample_waveform(wav, sr, target_sr)
        return wav
    except Exception as exc:  # pragma: no cover - backend fallback
        errors.append(f"torchaudio: {exc}")

    if shutil.which("ffmpeg"):
        try:
            wav = _ffmpeg_decode(audio_path, target_sr)
            return wav.astype(np.float32)
        except Exception as exc:  # pragma: no cover - ffmpeg fallback
            errors.append(f"ffmpeg: {exc}")

    LOGGER.error("Failed to load %s. Errors: %s", audio_path, errors)
    raise RuntimeError(f"Unable to decode audio file: {audio_path}")


def resample_waveform(
    wav: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample waveform using torchaudio's high-quality resampler.

    Args:
        wav: Input waveform (float32).
        orig_sr: Original sampling rate.
        target_sr: Target sampling rate.

    Returns:
        Resampled waveform.
    """
    waveform = torch.from_numpy(wav).unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    return resampled.squeeze(0).numpy().astype(np.float32)


def compute_log_mel(
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    frame_len: float = FRAME_LEN,
    frame_hop: float = FRAME_HOP,
) -> np.ndarray:
    """
    Compute log-mel spectrogram using torchaudio.kaldi.fbank.

    Args:
        wav: 1-D waveform array.
        sample_rate: Sampling rate in Hz.
        n_mels: Number of mel bins.
        frame_len: Frame length in seconds.
        frame_hop: Frame hop in seconds.

    Returns:
        Float32 spectrogram of shape (T, n_mels).
    """
    if wav.ndim != 1:
        raise ValueError(f"Expected mono waveform, got shape {wav.shape}")
    waveform = torch.from_numpy(wav).unsqueeze(0)
    frame_length_ms = frame_len * 1000.0
    frame_shift_ms = frame_hop * 1000.0
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        sample_frequency=sample_rate,
        num_mel_bins=n_mels,
        frame_length=frame_length_ms,
        frame_shift=frame_shift_ms,
        dither=0.0,
        energy_floor=1.0,
    )
    return fbank.numpy().astype(np.float32)


__all__ = ["DEFAULT_AUDIO_EXTS", "load_audio", "resample_waveform", "compute_log_mel"]
