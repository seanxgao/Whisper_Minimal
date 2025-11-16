"""Training loop for tiny VAD model on fixed-size chunks."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any
import logging
import random
import numpy as np
import os
import json
import shutil
from datetime import datetime
from tqdm import tqdm

from vad_distill.distill.tiny_vad.model import build_tiny_vad_model
from vad_distill.distill.tiny_vad.dataset import TinyVADChunkDataset
from vad_distill.utils.training import create_optimizer

logger = logging.getLogger(__name__)


def setup_logging(logs_dir: Path) -> None:
    """Setup file logging to logs/train.log."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "train.log"
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")


def set_deterministic_seeds(seed: int = 2025) -> None:
    """Set all random seeds for complete determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set all random seeds to {seed} (deterministic mode)")


def save_checkpoint(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    step: int,
    loss: float,
    is_best: bool = False,
    config: Dict[str, Any] | None = None,
) -> None:
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save last checkpoint
    last_path = checkpoint_dir / "last.pt"
    torch.save(checkpoint, last_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best.pt"
        shutil.copy(last_path, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")
    
    # Save optimizer and scheduler separately
    if optimizer is not None:
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
    
    if scheduler is not None:
        scheduler_path = checkpoint_dir / "scheduler.pt"
        torch.save(scheduler.state_dict(), scheduler_path)
    
    # Save config used
    if config is not None:
        config_path = checkpoint_dir / "config_used.yaml"
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: torch.device = torch.device('cpu'),
) -> tuple[int, int, float]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Resuming from epoch {epoch}, step {step}, loss {loss:.6f}")
    
    return epoch, step, loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0,
    writer=None,
) -> tuple[float, int]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.to(device)  # (B, 100, 80)
        labels = labels.to(device)      # (B, 100)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)  # (B, 100)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    writer=None,
    global_step: int = 0,
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validation"):
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, global_step)
    
    return avg_loss


def train_tiny_vad(config: Dict[str, Any]) -> None:
    """
    Train the tiny VAD model on fixed-size chunks.
    
    Args:
        config: Configuration dictionary loaded from student_tiny_vad.yaml
    """
    # Setup paths
    checkpoint_dir = Path(config.get('paths', {}).get('checkpoint_dir', 'vad_distill/distill/tiny_vad/checkpoints'))
    logs_dir = Path(config.get('paths', {}).get('logs_dir', 'logs'))
    
    # Setup logging
    setup_logging(logs_dir)
    
    # Set deterministic seeds
    seed = config.get('random_seed', 2025)
    set_deterministic_seeds(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build model
    model = build_tiny_vad_model(config)
    model = model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    chunks_dir = config.get('chunks_dir', 'data/chunks')
    index_file_path = Path(chunks_dir) / "index.json"
    index_file = str(index_file_path) if index_file_path.exists() else None
    
    dataset = TinyVADChunkDataset(
        chunks_dir=chunks_dir,
        index_file=index_file,
        shuffle=True,
        seed=seed,
    )
    logger.info(f"Dataset size: {len(dataset)} chunks")
    
    # Split dataset for validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    logger.info(f"Train: {len(train_dataset)} chunks, Val: {len(val_dataset)} chunks")
    
    # Create dataloaders
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    logger.info(f"DataLoader: batch_size={batch_size}, num_workers={num_workers}")
    
    # Setup loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    learning_rate = config.get('learning_rate', 0.001)
    optimizer = create_optimizer(
        model,
        'adam',
        learning_rate,
        weight_decay=0.0001
    )
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Setup TensorBoard
    writer = None
    if config.get('use_tensorboard', True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_dir = logs_dir / "tensorboard"
            writer = SummaryWriter(log_dir=str(tensorboard_dir))
            logger.info(f"TensorBoard logging to {tensorboard_dir}")
        except ImportError:
            logger.warning("TensorBoard not available, skipping")
    
    # Resume training if specified
    start_epoch = 0
    start_step = 0
    best_loss = float('inf')
    train_history = []
    
    resume_from = config.get('resume_from')
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            start_epoch, start_step, best_loss = load_checkpoint(
                resume_path, model, optimizer, scheduler, device
            )
            # Load training history
            history_path = checkpoint_dir / "train_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    train_history = json.load(f)
        else:
            logger.warning(f"Resume checkpoint not found: {resume_path}, starting from scratch")
    
    # Training configuration
    num_epochs = config.get('epochs', 100)
    save_every_n_steps = config.get('save_every_n_steps', 1000)
    validate_every_n_epochs = config.get('validate_every_n_epochs', 5)
    max_grad_norm = 1.0
    
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Resume from epoch: {start_epoch}")
    logger.info("=" * 60)
    
    # Training loop
    global_step = start_step
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, num_batches = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, max_grad_norm, writer
        )
        
        global_step += num_batches
        
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}")
        
        if writer is not None:
            writer.add_scalar('train/epoch_loss', train_loss, epoch)
        
        # Validate
        if (epoch + 1) % validate_every_n_epochs == 0:
            val_loss = validate(model, val_loader, criterion, device, writer, epoch)
            logger.info(f"Epoch {epoch}: val_loss={val_loss:.6f}")
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                logger.info(f"New best validation loss: {best_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or (global_step % save_every_n_steps == 0):
            save_checkpoint(
                checkpoint_dir, model, optimizer, scheduler,
                epoch, global_step, train_loss, is_best=False, config=config
            )
        
        # Update training history
        train_history.append({
            'epoch': epoch,
            'step': global_step,
            'train_loss': train_loss,
            'val_loss': val_loss if (epoch + 1) % validate_every_n_epochs == 0 else None,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })
        
        # Save training history
        history_path = checkpoint_dir / "train_history.json"
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=2)
        
        # Also save to logs directory
        logs_history_path = logs_dir / "train_history.json"
        with open(logs_history_path, 'w') as f:
            json.dump(train_history, f, indent=2)
    
    # Save final checkpoint
    save_checkpoint(
        checkpoint_dir, model, optimizer, scheduler,
        num_epochs - 1, global_step, train_loss, is_best=False, config=config
    )
    
    if writer is not None:
        writer.close()
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation loss: {best_loss:.6f}")
    logger.info("=" * 60)
