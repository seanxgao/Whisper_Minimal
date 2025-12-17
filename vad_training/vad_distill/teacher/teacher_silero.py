"""Silero-VAD teacher helpers."""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F
from silero_vad import load_silero_vad


def _normalize_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def load_silero(device: str | torch.device | None = None) -> Tuple[Any, torch.device]:
    """
    Instantiate the Silero-VAD teacher model once.

    Returns:
        (model, device): The loaded Silero VAD model and device.
    """
    resolved_device = _normalize_device(device)
    model = load_silero_vad()
    model.eval()
    # Move model to device if needed
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        model = model.to(resolved_device)
    return model, resolved_device


def get_vad_probs(
    model: Any,
    wav_16k: torch.Tensor,
    window_size: int = 512,
    hop_size: int = 160,
    device: torch.device | None = None,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Compute per-window speech probabilities with Silero.

    Args:
        model: Silero VAD model (from load_silero_vad)
        wav_16k: 1-D waveform tensor at 16kHz
        window_size: Window size in samples
        hop_size: Hop size in samples
        device: Device to run inference on
        sample_rate: Sample rate of the waveform (default: 16000)

    Returns:
        Tensor of probabilities, shape (num_windows,)
    """
    if wav_16k.dim() != 1:
        raise ValueError(f"Expected 1-D waveform, got {wav_16k.shape}")

    device = device or _normalize_device(None)
    wav = wav_16k.to(device)

    if wav.numel() < window_size:
        wav = F.pad(wav, (0, window_size - wav.numel()))

    probs = []
    start = 0
    with torch.no_grad():
        while start + window_size <= wav.shape[0]:
            chunk = wav[start : start + window_size]
            # New API requires sr parameter
            logits = model(chunk.unsqueeze(0), sr=sample_rate)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            prob = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32, device=device))
            probs.append(prob.squeeze().clamp(0.0, 1.0))
            start += hop_size

    if not probs:
        logits = model(wav[:window_size].unsqueeze(0), sr=sample_rate)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        prob = torch.sigmoid(torch.as_tensor(logits, dtype=torch.float32, device=device))
        probs.append(prob.squeeze().clamp(0.0, 1.0))

    return torch.stack(probs).to(torch.float32)


__all__ = ["load_silero", "get_vad_probs"]
