"""
Training utilities including trainer class, callbacks, and checkpointing
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable, List, Tuple
import logging
from collections import defaultdict
import json
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

# ==================== CALLBACKS ====================
class Callback:
    """Base callback class"""
    
    def on_epoch_start(self, epoch: int, **kwargs): pass
    def on_epoch_end(self, epoch: int, metrics: Dict, **kwargs): pass
    def on_batch_start(self, batch: int, **kwargs): pass
    def on_batch_end(self, batch: int, loss: float, **kwargs): pass
    def on_train_start(self, **kwargs): pass
    def on_train_end(self, **kwargs): pass

class EarlyStoppingCallback(Callback):
    """Early stopping based on validation metric"""
    
    def __init__(self, metric: str = "val_loss", patience: int = 10, min_delta: float = 1e-4):
        """
        Args:
            metric: Metric to monitor
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.counter = 0
        self.stop = False
    
    def on_epoch_end(self, epoch: int, metrics: Dict, **kwargs):
        if self.metric not in metrics:
            return
        
        current_value = metrics[self.metric]
        
        if self.best_value is None:
            self.best_value = current_value
        elif current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                self.stop = True

class CheckpointCallback(Callback):
    """Checkpoint model saving"""
    
    def __init__(self, checkpoint_dir: Path, save_best_only: bool = True, 
                 monitor_metric: str = "val_loss"):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save best checkpoint
            monitor_metric: Metric to monitor for best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.best_value = None
    
    def on_epoch_end(self, epoch: int, metrics: Dict, model: nn.Module = None, **kwargs):
        if model is None:
            return
        
        # Check if this is the best epoch
        if self.monitor_metric in metrics:
            current_value = metrics[self.monitor_metric]
            
            if self.best_value is None or current_value < self.best_value:
                self.best_value = current_value
                save_best = True
            else:
                save_best = False
        else:
            save_best = True
        
        # Save checkpoint
        if save_best or not self.save_best_only:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics
            }
            torch.save(state, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

class TensorBoardCallback(Callback):
    """TensorBoard logging (requires tensorboard)"""
    
    def __init__(self, log_dir: Path):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            logger.warning("TensorBoard not installed. Skipping TensorBoard logging.")
            self.writer = None
    
    def on_epoch_end(self, epoch: int, metrics: Dict, **kwargs):
        if self.writer is None:
            return
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.writer.add_scalar(f'metrics/{metric_name}', metric_value, epoch)
        
        self.writer.flush()
    
    def on_train_end(self, **kwargs):
        if self.writer:
            self.writer.close()

# ==================== TRAINER ====================
class Trainer:
    """
    Main training loop manager
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion,
                 device: str = "cuda",
                 training_config = None):
        """
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            device: Device to use (cuda/cpu)
            training_config: Training configuration object
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.config = training_config
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Callbacks
        self.callbacks: List[Callback] = []
        
        # Metrics history
        self.history = defaultdict(list)
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    
    def _setup_optimizer(self):
        """Setup optimizer based on config"""
        lr = self.config.learning_rate
        wd = self.config.weight_decay
        
        if self.config.optimizer == "adam":
            optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif self.config.optimizer == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif self.config.optimizer == "sgd":
            optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler based on config"""
        if self.config.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.num_epochs)
        elif self.config.lr_scheduler == "step":
            scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif self.config.lr_scheduler == "exponential":
            scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        else:
            return None
        
        return scheduler
    
    def add_callback(self, callback: Callback):
        """Add callback"""
        self.callbacks.append(callback)
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        loss = self.criterion(outputs[0], labels)
                    else:
                        loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    loss = self.criterion(outputs[0], labels)
                else:
                    loss = self.criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Batch {batch_idx + 1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            if isinstance(outputs, tuple):
                loss = self.criterion(outputs[0], labels)
            else:
                loss = self.criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}
    
    def train(self) -> Dict:
        """Main training loop"""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_start()
        
        for epoch in range(self.config.num_epochs):
            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_start(epoch)
            
            # Train epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Combine metrics
            metrics = {
                "train_loss": train_loss,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]['lr']
            }
            
            # Record history
            for key, value in metrics.items():
                self.history[key].append(value)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics, model=self.model)
            
            # Check for early stopping
            for callback in self.callbacks:
                if isinstance(callback, EarlyStoppingCallback) and callback.stop:
                    logger.info("Training stopped by early stopping callback")
                    break
        
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_end()
        
        return dict(self.history)

# ==================== TRAINING UTILITIES ====================
def setup_training(model, train_loader, val_loader, config):
    """Setup trainer with callbacks"""
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=config.hardware_config.device if hasattr(config, "hardware_config") else "cuda",
        training_config=config
    )
    
    # Add callbacks
    if config.early_stopping_enabled:
        trainer.add_callback(EarlyStoppingCallback(
            metric=config.early_stopping_metric,
            patience=config.early_stopping_patience
        ))
    
    # Checkpoint callback
    trainer.add_callback(CheckpointCallback(
        checkpoint_dir=config.checkpoint_dir,
        save_best_only=config.save_best_only
    ))
    
    # TensorBoard callback
    if hasattr(config, "tensorboard_enabled") and config.tensorboard_enabled:
        trainer.add_callback(TensorBoardCallback(
            log_dir=Path(config.checkpoint_dir).parent / "tensorboard_logs"
        ))
    
    return trainer

def load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer=None) -> Dict:
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint

def save_training_history(history: Dict, save_path: Path):
    """Save training history"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {save_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Training utilities module loaded successfully")
