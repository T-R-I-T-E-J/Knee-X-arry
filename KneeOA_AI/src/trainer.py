"""
Efficient Trainer with Callback support for Knee OA analysis.
Supports mixed precision, early stopping, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json

logger = logging.getLogger(__name__)

class Trainer:
    """Main training loop orchestrator."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 device: str,
                 config):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        
        # Optimizer
        opt_map = {"adam": Adam, "adamw": AdamW, "sgd": SGD}
        self.optimizer = opt_map.get(config.optimizer, Adam)(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)
        
        # Persistence
        self.best_val_loss = float('inf')
        self.history = {"train_loss": [], "val_loss": [], "lr": []}
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    def train(self) -> Dict:
        """Runs the full training loop."""
        logger.info(f"Starting training on {self.device}")
        
        for epoch in range(self.config.num_epochs):
            train_loss = self._run_epoch(True)
            val_loss = self._run_epoch(False)
            
            self.scheduler.step()
            curr_lr = self.optimizer.param_groups[0]['lr']
            
            # Record
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(curr_lr)
            
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - Loss: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {curr_lr:.6f}")
            
            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.checkpoint_dir / "best_model.pt")
                logger.info(f"New best model saved!")
                
        return self.history

    def _run_epoch(self, train: bool) -> float:
        self.model.train() if train else self.model.eval()
        loader = self.train_loader if train else self.val_loader
        total_loss = 0.0
        
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            with torch.set_grad_enabled(train):
                # Mixed Precision
                with torch.cuda.amp.autocast(enabled=bool(self.scaler)):
                    outputs = self.model(images)
                    if isinstance(outputs, tuple):
                        logits, params = outputs
                        loss = self.criterion(logits, labels, params=params)
                    else:
                        loss = self.criterion(outputs, labels)
                
                if train:
                    self.optimizer.zero_grad()
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                        
            total_loss += loss.item()
            
        return total_loss / len(loader)

def setup_training(model, train_loader, val_loader, config, criterion, device):
    """Factory to create and configure Trainer."""
    return Trainer(model, train_loader, val_loader, criterion, device, config)

def save_training_history(history: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
