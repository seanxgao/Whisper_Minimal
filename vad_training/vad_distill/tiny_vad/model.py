"""Tiny frame-level VAD model."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


class TinyVADModel(nn.Module):
    """
    Dilated CNN + BiLSTM student network.
    """

    def __init__(
        self,
        n_mels: int = 80,
        cnn_channels: int = 128,
        lstm_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, cnn_channels, kernel_size=5, dilation=1, padding=2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, dilation=2, padding=4)
        self.conv3 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, dilation=4, padding=8)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, time, n_mels) tensor, got {x.shape}")
        if x.shape[1:] != (100, 80):
            raise ValueError("TinyVADModel expects chunks shaped (100, 80)")

        x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-5)
        x = x.transpose(1, 2)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.dropout(self.relu(self.conv3(x)))
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.classifier(x).squeeze(-1)
        return x


def build_tiny_vad_model(config: Dict[str, Any]) -> nn.Module:
    params = config.get("model", {})
    return TinyVADModel(
        n_mels=params.get("n_mels", 80),
        cnn_channels=params.get("cnn_channels", 128),
        lstm_hidden=params.get("lstm_hidden", 64),
        dropout=params.get("dropout", 0.1),
    )
