"""
Training Module for Raman Spectroscopy Classification Model

This module provides training utilities: training loop, validation, metrics,
and early stopping. All functions are independent and reusable.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import copy
import time
from datetime import timedelta


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (loss or accuracy)
            model: Model to save weights from
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self._load_checkpoint(model)
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def _save_checkpoint(self, model: nn.Module) -> None:
        """Save model weights."""
        self.best_weights = copy.deepcopy(model.state_dict())
    
    def _load_checkpoint(self, model: nn.Module) -> None:
        """Load best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    ordinal_criterion: Optional[nn.Module] = None,
    ordinal_weight: float = 0.5
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Classification loss function
        optimizer: Optimizer
        device: Device to run on
        ordinal_criterion: Optional ordinal/regression loss function
        ordinal_weight: Weight for ordinal loss if used
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    running_class_loss = 0.0
    running_ordinal_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (spectra, labels) in enumerate(dataloader):
        spectra = spectra.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        class_logits, ordinal_logits = model(spectra)
        
        # Classification loss
        class_loss = criterion(class_logits, labels)
        total_loss = class_loss
        running_class_loss += class_loss.item()
        
        # Ordinal loss (if enabled)
        if ordinal_logits is not None and ordinal_criterion is not None:
            # Assume ordinal labels are same as class labels for simplicity
            # In practice, you might have separate ordinal labels
            ordinal_loss = ordinal_criterion(ordinal_logits, labels.float())
            total_loss = (1 - ordinal_weight) * class_loss + ordinal_weight * ordinal_loss
            running_ordinal_loss += ordinal_loss.item()
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += total_loss.item()
        _, predicted = torch.max(class_logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_class_loss = running_class_loss / len(dataloader)
    epoch_ordinal_loss = running_ordinal_loss / len(dataloader) if ordinal_criterion else 0.0
    epoch_acc = 100.0 * correct / total
    
    return {
        'loss': epoch_loss,
        'class_loss': epoch_class_loss,
        'ordinal_loss': epoch_ordinal_loss,
        'accuracy': epoch_acc
    }


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ordinal_criterion: Optional[nn.Module] = None,
    ordinal_weight: float = 0.5
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Classification loss function
        device: Device to run on
        ordinal_criterion: Optional ordinal/regression loss function
        ordinal_weight: Weight for ordinal loss if used
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    running_loss = 0.0
    running_class_loss = 0.0
    running_ordinal_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for spectra, labels in dataloader:
            spectra = spectra.to(device)
            labels = labels.to(device)
            
            # Forward pass
            class_logits, ordinal_logits = model(spectra)
            
            # Classification loss
            class_loss = criterion(class_logits, labels)
            total_loss = class_loss
            running_class_loss += class_loss.item()
            
            # Ordinal loss (if enabled)
            if ordinal_logits is not None and ordinal_criterion is not None:
                ordinal_loss = ordinal_criterion(ordinal_logits, labels.float())
                total_loss = (1 - ordinal_weight) * class_loss + ordinal_weight * ordinal_loss
                running_ordinal_loss += ordinal_loss.item()
            
            # Metrics
            running_loss += total_loss.item()
            _, predicted = torch.max(class_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_class_loss = running_class_loss / len(dataloader)
    epoch_ordinal_loss = running_ordinal_loss / len(dataloader) if ordinal_criterion else 0.0
    epoch_acc = 100.0 * correct / total
    
    return {
        'loss': epoch_loss,
        'class_loss': epoch_class_loss,
        'ordinal_loss': epoch_ordinal_loss,
        'accuracy': epoch_acc,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    early_stopping: Optional[EarlyStopping] = None,
    ordinal_criterion: Optional[nn.Module] = None,
    ordinal_weight: float = 0.5,
    verbose: bool = True,
    initial_epoch: int = 1,
    previous_history: Optional[Dict[str, List[float]]] = None
) -> Dict[str, List[float]]:
    """
    Complete training loop for the model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of epochs
        optimizer: Optimizer
        criterion: Classification loss function
        device: Device to run on
        scheduler: Optional learning rate scheduler
        early_stopping: Optional early stopping callback
        ordinal_criterion: Optional ordinal/regression loss function
        ordinal_weight: Weight for ordinal loss if used
        verbose: Whether to print progress
        initial_epoch: Starting epoch number (for resuming training)
        previous_history: Previous training history to continue from
    
    Returns:
        Dictionary with training history (train_loss, val_loss, train_acc, val_acc)
    """
    # Initialize history from previous if resuming
    if previous_history:
        history = {
            'train_loss': previous_history.get('train_loss', []).copy(),
            'train_acc': previous_history.get('train_acc', []).copy(),
            'val_loss': previous_history.get('val_loss', []).copy(),
            'val_acc': previous_history.get('val_acc', []).copy()
        }
        if verbose:
            print(f"Resuming training from epoch {initial_epoch}")
            print(f"Previous training: {len(history['train_loss'])} epochs completed")
    else:
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    # Timing tracking
    start_time = time.time()
    epoch_times = []
    
    # Track best epoch
    best_epoch_num = initial_epoch
    best_val_loss = float('inf')
    if previous_history and history['val_loss']:
        best_val_loss = min(history['val_loss'])
        best_epoch_num = history['val_loss'].index(best_val_loss) + 1
    best_val_acc = 0.0
    
    for epoch in range(n_epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            ordinal_criterion, ordinal_weight
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device,
            ordinal_criterion, ordinal_weight
        )
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Check if this is the best epoch
        current_epoch_num = initial_epoch + epoch
        is_new_best = False
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['accuracy']
            best_epoch_num = current_epoch_num
            is_new_best = True
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        total_time = time.time() - start_time
        
        # Estimate remaining time
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = n_epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_metrics['loss'], model):
                if verbose:
                    print(f"Early stopping at epoch {current_epoch_num}")
                break
        
        # Print progress
        if verbose:
            total_epochs = initial_epoch + n_epochs - 1
            print(f"Epoch {current_epoch_num}/{total_epochs} (training epoch {epoch + 1}/{n_epochs})")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"  Time: {epoch_time:.2f}s | Elapsed: {str(timedelta(seconds=int(total_time)))} | Est. remaining: {str(timedelta(seconds=int(estimated_remaining)))}")
            
            # Sanity check: Show best epoch
            if is_new_best:
                print(f"  ✓ NEW BEST! Epoch {best_epoch_num} is now the best (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)")
            else:
                print(f"  Best so far: Epoch {best_epoch_num} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)")
            print()
    
    # Print training summary
    if verbose and epoch_times:
        total_time = time.time() - start_time
        avg_epoch_time = np.mean(epoch_times)
        print(f"\nTraining complete!")
        print(f"  Total epochs: {len(epoch_times)}")
        print(f"  Total time: {str(timedelta(seconds=int(total_time)))}")
        print(f"  Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"  Best epoch: {best_epoch_num} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)")
        print()
    
    return history


def create_dataloader(
    spectra: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a PyTorch DataLoader from numpy arrays.
    
    Args:
        spectra: Spectrum array of shape (n_samples, n_wavenumbers)
        labels: Label array of shape (n_samples,)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    # Convert to tensors and add channel dimension
    spectra_tensor = torch.FloatTensor(spectra).unsqueeze(1)  # (n_samples, 1, n_wavenumbers)
    labels_tensor = torch.LongTensor(labels)
    
    dataset = TensorDataset(spectra_tensor, labels_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    additional_info: Optional[Dict] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        additional_info: Optional additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    filepath: str,
    device: torch.device
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to load on
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint

