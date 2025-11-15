"""Chunk trigger model: predicts good chunk boundaries from VAD probability windows."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any


class ChunkTriggerMLP(nn.Module):
    """
    Small MLP model for chunk trigger prediction.
    
    Input: (batch, window_size, 1) window of VAD probabilities
    Output: (batch, 1) scalar logit for "trigger here"
    """
    
    def __init__(
        self,
        window_size: int = 50,
        hidden_dims: list[int] = [32, 16],
    ):
        """
        Initialize the chunk trigger MLP model.

        Args:
            window_size: Number of frames in the input window
            hidden_dims: List of hidden dimensions for MLP layers
        """
        super().__init__()
        
        layers = []
        input_dim = window_size  # Flatten the window
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size, 1)

        Returns:
            Output logits of shape (batch, 1)
        """
        # Flatten window dimension: (batch, window_size, 1) -> (batch, window_size)
        x = x.squeeze(-1)
        
        # Apply MLP
        out = self.network(x)
        
        return out


class ChunkTriggerCNN(nn.Module):
    """
    Small 1D CNN model for chunk trigger prediction.
    
    Input: (batch, window_size, 1) window of VAD probabilities
    Output: (batch, 1) scalar logit for "trigger here"
    """
    
    def __init__(
        self,
        window_size: int = 50,
        hidden_dims: list[int] = [32, 16],
    ):
        """
        Initialize the chunk trigger CNN model.

        Args:
            window_size: Number of frames in the input window
            hidden_dims: List of channel dimensions for conv layers
        """
        super().__init__()
        
        layers = []
        in_channels = 1
        
        # Build 1D CNN layers
        for out_channels in hidden_dims:
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.AdaptiveAvgPool1d(1))  # Global pooling
            in_channels = out_channels
        
        # Final linear layer
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, window_size, 1)

        Returns:
            Output logits of shape (batch, 1)
        """
        # Convert (batch, window_size, 1) to (batch, 1, window_size) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply CNN
        out = self.network(x)
        
        return out


def build_chunk_trigger_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build a ChunkTriggerModel from configuration dictionary.

    Args:
        config: Configuration dict with 'model' key containing model parameters

    Returns:
        Initialized ChunkTriggerModel (MLP or CNN)
    """
    model_config = config.get('model', {})
    model_type = model_config.get('model_type', 'mlp').lower()
    window_size = model_config.get('window_size', 50)
    hidden_dims = model_config.get('hidden_dims', [32, 16])
    
    if model_type == 'mlp':
        return ChunkTriggerMLP(window_size=window_size, hidden_dims=hidden_dims)
    elif model_type == 'cnn':
        return ChunkTriggerCNN(window_size=window_size, hidden_dims=hidden_dims)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'mlp' or 'cnn'")

