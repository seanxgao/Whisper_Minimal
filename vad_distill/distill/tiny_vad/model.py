"""Tiny frame-level VAD model: 1D CNN over log-mel features."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any


class TinyVADModel(nn.Module):
    """
    Small 1D CNN model for frame-level VAD prediction.
    
    Input: (batch, 100, 80) log-mel spectrogram - fixed-size chunks
    Output: (batch, 100, 1) frame-level speech probability logits
    
    This model expects fixed-size input chunks of 100 frames (1 second at 10ms hop).
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        hidden_dims: list[int] = [64, 128, 64],
        kernel_sizes: list[int] = [3, 3, 3],
        dropout: float = 0.1,
    ):
        """
        Initialize the tiny VAD model.

        Args:
            n_mels: Number of mel filter banks (input feature dimension)
            hidden_dims: List of channel dimensions for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout: Dropout probability
        """
        super().__init__()
        
        if len(hidden_dims) != len(kernel_sizes):
            raise ValueError("hidden_dims and kernel_sizes must have the same length")
        
        layers = []
        in_channels = n_mels
        
        # Build 1D CNN layers
        for i, (out_channels, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # Same padding
                )
            )
            layers.append(nn.ReLU())
            if dropout > 0 and i < len(hidden_dims) - 1:  # No dropout before final layer
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        # Final linear layer to 1 channel
        layers.append(nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 100, 80) - fixed-size chunks

        Returns:
            Output logits of shape (batch, 100, 1)
        """
        # Validate input shape: must be (batch, 100, 80)
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, time, freq), got shape {x.shape}"
            )
        if x.shape[1] != 100:
            raise ValueError(
                f"Expected time dimension 100, got {x.shape[1]}"
            )
        if x.shape[2] != 80:
            raise ValueError(
                f"Expected freq dimension 80, got {x.shape[2]}"
            )
        
        # Convert (batch, time, freq) to (batch, freq, time) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply CNN
        out = self.network(x)
        
        # Convert back to (batch, time, 1)
        out = out.transpose(1, 2)
        
        return out


def build_tiny_vad_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a TinyVADModel from configuration dictionary.

    Args:
        config: Configuration dict with 'model' key containing model parameters

    Returns:
        Initialized TinyVADModel
    """
    model_config = config.get('model', {})
    return TinyVADModel(
        n_mels=model_config.get('n_mels', 80),
        hidden_dims=model_config.get('hidden_dims', [64, 128, 64]),
        kernel_sizes=model_config.get('kernel_sizes', [3, 3, 3]),
        dropout=model_config.get('dropout', 0.1),
    )

