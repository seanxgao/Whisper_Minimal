"""Common training utilities to avoid code duplication."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Callable, Tuple
import logging
import time

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float = 0.0001,
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam' or 'sgd')
        lr: Learning rate
        weight_decay: Weight decay coefficient
    
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, float]:
    """
    Load checkpoint and restore model and optimizer state.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        device: Device to load checkpoint on
    
    Returns:
        Tuple of (start_epoch, best_loss)
    """
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}, starting from scratch")
        return 0, float('inf')
    
    logger.info(f"Resuming training from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('loss', float('inf'))
    logger.info(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")
    return start_epoch, best_loss


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: float,
) -> None:
    """
    Save training checkpoint.
    
    Args:
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save
        loss: Current loss value
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 100,
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        log_interval: Log every N batches
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()
    
    for batch_idx, batch_data in enumerate(dataloader):
        # Move data to device (assumes batch_data is tuple/list of tensors)
        inputs = batch_data[0].to(device)
        targets = batch_data[1].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs)
        
        # Handle different output shapes
        if len(logits.shape) > 1 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        if len(targets.shape) > 1 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        
        # Ensure shapes match
        if logits.shape != targets.shape:
            min_len = min(logits.shape[-1], targets.shape[-1])
            logits = logits[..., :min_len]
            targets = targets[..., :min_len]
        
        # Compute loss
        loss = criterion(logits, targets)
        
        # Check for NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss detected at batch {batch_idx+1}: {loss.item()}")
            continue
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        if batch_idx == 0:  # Only check first batch to avoid overhead
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2)
                    total_grad_norm += param_grad_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            if total_grad_norm == 0.0:
                logger.warning("Zero gradient detected! Model may not be learning.")
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            loss_val = loss.item()
            # Use scientific notation for small values
            if loss_val < 0.0001:
                loss_str = f"{loss_val:.2e}"
            else:
                loss_str = f"{loss_val:.4f}"
            logger.info(
                f"Batch {batch_idx+1}/{len(dataloader)}, "
                f"Loss: {loss_str}, "
                f"Time: {elapsed:.2f}s"
            )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_with_early_stopping(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    checkpoint_dir: Path,
    start_epoch: int = 0,
    best_loss: float = float('inf'),
    patience: int = 10,
    save_interval: int = 5,
    log_interval: int = 100,
) -> float:
    """
    Train model with early stopping and checkpoint saving.
    
    Args:
        model: Model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Maximum number of epochs
        checkpoint_dir: Directory to save checkpoints
        start_epoch: Starting epoch (for resume)
        best_loss: Best loss so far (for resume)
        patience: Early stopping patience
        save_interval: Save checkpoint every N epochs
        log_interval: Log every N batches
    
    Returns:
        Best loss achieved
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    patience_counter = 0
    
    logger.info("Starting training...")
    training_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, device, log_interval)
        epoch_time = time.time() - epoch_start_time
        
        # Use scientific notation for small loss values
        if avg_loss < 0.0001:
            loss_str = f"{avg_loss:.2e}"
        else:
            loss_str = f"{avg_loss:.4f}"
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} completed, "
            f"Average Loss: {loss_str}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(checkpoint_path, epoch + 1, model, optimizer, avg_loss)
        
        # Early stopping and best model saving
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            best_path = checkpoint_dir / "best_model.pt"
            save_checkpoint(best_path, epoch + 1, model, optimizer, avg_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    total_time = time.time() - training_start_time
    logger.info("Training completed")
    if best_loss < 0.0001:
        best_loss_str = f"{best_loss:.2e}"
    else:
        best_loss_str = f"{best_loss:.4f}"
    logger.info(f"Best loss: {best_loss_str}")
    logger.info(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    return best_loss

