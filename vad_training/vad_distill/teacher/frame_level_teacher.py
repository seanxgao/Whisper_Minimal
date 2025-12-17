"""Frame-level Silero teacher wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from vad_distill.preprocessing.audio_utils import load_audio
from vad_distill.teacher.teacher_silero import get_vad_probs, load_silero

logger = logging.getLogger(__name__)


class FrameLevelTeacher:
    """Frame-level Silero wrapper producing float32 numpy arrays."""

    def __init__(
        self,
        device: str | None = "auto",
        window_size: int = 512,
        hop_size: int = 160,
        model: torch.nn.Module | None = None,
    ):
        if model is None:
            self.model, self.device = load_silero(device=device)
        else:
            self.model = model
            self.device = torch.device(device or "cpu")
        self.window_size = window_size
        self.hop_size = hop_size
        logger.info(
            "Silero teacher ready (device=%s, window=%d, hop=%d)",
            self.device,
            self.window_size,
            self.hop_size,
        )

    def get_frame_probs(self, wav_path: str | Path) -> np.ndarray:
        """Compute teacher probabilities for a file on disk."""
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        wav = load_audio(wav_path)
        return self._infer_from_array(wav)

    def infer(self, wav: np.ndarray, sr: int = 16000) -> Tuple[np.ndarray, list]:
        """Backwards-compatible API returning probs and empty segments."""
        if sr != 16000:
            raise ValueError("FrameLevelTeacher expects 16 kHz waveforms.")
        probs = self._infer_from_array(wav)
        return probs, []

    def _infer_from_array(self, wav: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        if wav.ndim != 1:
            raise ValueError(f"Expected mono waveform, got {wav.shape}")
        tensor = torch.from_numpy(wav).to(torch.float32).to(self.device)
        probs = get_vad_probs(
            self.model,
            tensor,
            window_size=self.window_size,
            hop_size=self.hop_size,
            device=self.device,
            sample_rate=sample_rate,
        )
        return probs.detach().cpu().numpy().astype(np.float32)
