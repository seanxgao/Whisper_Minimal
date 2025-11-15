"""Training loop for tiny VAD model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any
import logging

from vad_distill.distill.tiny_vad.model import build_tiny_vad_model
from vad_distill.distill.tiny_vad.dataset import TinyVADDataset

logger = logging.getLogger(__name__)


def train_tiny_vad(config: Dict[str, Any]) -> None:
    """
    Train the tiny VAD model.

    Args:
        config: Configuration dictionary loaded from student_tiny_vad.yaml
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build model
    model = build_tiny_vad_model(config)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset and dataloader
    data_config = config.get('data', {})
    dataset = TinyVADDataset(
        manifest_path=data_config.get('manifest_path'),
        frame_probs_dir=data_config.get('frame_probs_dir', 'data/teacher_labels/frame_probs'),
        frame_hop=data_config.get('frame_hop', 0.01),
        frame_len=data_config.get('frame_len', 0.025),
        n_mels=data_config.get('n_mels', 80),
        cache_features=data_config.get('cache_features', False),
        cache_dir=data_config.get('cache_dir', 'data/tiny_vad_inputs') if data_config.get('cache_features') else None,
    )
    
    train_config = config.get('training', {})
    batch_size = train_config.get('batch_size', 32)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to >0 if needed
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Setup loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # TODO: Add optional soft-label distillation loss (KL divergence)
    
    optimizer_name = train_config.get('optimizer', 'adam').lower()
    lr = train_config.get('learning_rate', 0.001)
    weight_decay = train_config.get('weight_decay', 0.0001)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Training loop
    num_epochs = train_config.get('num_epochs', 50)
    checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'models/tiny_vad/checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    patience = train_config.get('early_stopping_patience', 10)
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (mel_features, frame_labels) in enumerate(dataloader):
            mel_features = mel_features.to(device)
            frame_labels = frame_labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(mel_features).squeeze(-1)  # (batch, time)
            
            # Compute loss
            loss = criterion(logits, frame_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        save_interval = train_config.get('save_interval', 5)
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    logger.info("Training completed")
    logger.info(f"Best loss: {best_loss:.4f}")

