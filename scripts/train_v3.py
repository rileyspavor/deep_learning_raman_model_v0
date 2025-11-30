"""
Training Script for V3 Data
Trains CNN model on v3 data without preprocessing (data is already prepared).
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from typing import Optional, Tuple, Dict

from src.data_ingestion import load_npz_dataset
from src.model import create_model
from src.training import train_model, create_dataloader, EarlyStopping, save_checkpoint
from src.utils import stratified_split, compute_metrics, plot_confusion_matrix, plot_training_history, print_classification_report


def load_v3_data(npz_file: str):
    """
    Load v3 data from npz file.
    
    Args:
        npz_file: Path to npz file
        
    Returns:
        Dictionary with loaded data
    """
    print("=" * 60)
    print("Loading V3 Data")
    print("=" * 60)
    
    # Load the dataset
    spectra, wavenumbers, labels, label_names, metadata = load_npz_dataset(
        file_path=npz_file,
        spectra_key="spectra",
        wavenumbers_key="wavenumbers",
        labels_key="y",
        label_names_key="label_names"
    )
    
    print(f"\nDataset loaded successfully!")
    print(f"  Spectra shape: {spectra.shape}")
    print(f"  Wavenumbers shape: {wavenumbers.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Spectra dtype: {spectra.dtype}")
    print(f"  Number of classes: {len(np.unique(labels))}")
    
    # Convert int16 to float32 if needed (just type conversion for PyTorch, not preprocessing)
    if spectra.dtype == np.int16:
        print(f"  Converting spectra from int16 to float32 (type conversion only, no preprocessing)...")
        spectra = spectra.astype(np.float32)
    
    # Create label mapping
    unique_labels = np.unique(labels)
    if label_names is not None:
        label_mapping = {int(label): str(label_names[int(label)]) for label in unique_labels}
        print(f"\n  Label names: {label_names}")
    else:
        label_mapping = {int(label): f"Class_{label}" for label in unique_labels}
        label_names = np.array([label_mapping[int(label)] for label in unique_labels])
        print(f"\n  Using default label names")
    
    print(f"  Label mapping: {label_mapping}")
    
    # Show metadata
    if metadata:
        print(f"\n  Additional metadata keys: {list(metadata.keys())}")
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
    
    # Check if train/val/test splits are already in the file
    has_splits = all(key in metadata for key in ['train_idx', 'val_idx', 'test_idx']) or \
                 all(key in metadata for key in ['X_train', 'X_val', 'X_test'])
    
    return {
        'spectra': spectra,
        'wavenumbers': wavenumbers,
        'labels': labels,
        'label_names': label_names,
        'label_mapping': label_mapping,
        'metadata': metadata,
        'has_splits': has_splits
    }


def prepare_train_val_test(data_dict, use_existing_splits=False):
    """
    Prepare train/val/test splits from loaded data.
    
    Args:
        data_dict: Dictionary with loaded data
        use_existing_splits: Whether to use existing splits from metadata
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, target_grid)
    """
    print("\n" + "=" * 60)
    print("Preparing Train/Val/Test Splits")
    print("=" * 60)
    
    spectra = data_dict['spectra']
    labels = data_dict['labels']
    wavenumbers = data_dict['wavenumbers']
    metadata = data_dict['metadata']
    
    # Check if splits already exist
    if use_existing_splits:
        if 'train_idx' in metadata and 'val_idx' in metadata and 'test_idx' in metadata:
            print("  Using existing split indices from metadata...")
            train_idx = metadata['train_idx']
            val_idx = metadata['val_idx']
            test_idx = metadata['test_idx']
            
            X_train = spectra[train_idx]
            X_val = spectra[val_idx]
            X_test = spectra[test_idx]
            y_train = labels[train_idx]
            y_val = labels[val_idx]
            y_test = labels[test_idx]
        elif 'X_train' in metadata and 'X_val' in metadata and 'X_test' in metadata:
            print("  Using existing split arrays from metadata...")
            X_train = metadata['X_train']
            X_val = metadata['X_val']
            X_test = metadata['X_test']
            y_train = metadata['y_train']
            y_val = metadata['y_val']
            y_test = metadata['y_test']
        else:
            print("  No existing splits found, creating new stratified splits...")
            X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
                spectra, labels,
                train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                random_state=42
            )
    else:
        print("  Creating new stratified splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
            spectra, labels,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            random_state=42
        )
    
    # Get target grid (wavenumbers)
    if wavenumbers.ndim == 1:
        target_grid = wavenumbers
    else:
        # If wavenumbers are per-spectrum, use the first one
        target_grid = wavenumbers[0]
        print(f"  Warning: Using first wavenumber grid as target")
    
    print(f"\n  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Wavenumber grid length: {len(target_grid)}")
    print(f"  Wavenumber range: {target_grid.min():.1f} - {target_grid.max():.1f} cm⁻¹")
    
    # Show class distribution
    print(f"\n  Class distribution (train set):")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        class_name = data_dict['label_mapping'].get(int(label), f"Class_{label}")
        print(f"    {class_name}: {count} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, target_grid


def get_next_model_version(base_dir: Path) -> int:
    """
    Find the next available model version number.
    
    Args:
        base_dir: Base directory containing versioned model folders
        
    Returns:
        Next version number (e.g., 3 if model_v2 exists)
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Find existing model versions
    existing_versions = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith('model_v'):
            try:
                version = int(item.name.replace('model_v', ''))
                existing_versions.append(version)
            except ValueError:
                continue
    
    if not existing_versions:
        return 1
    
    return max(existing_versions) + 1


def load_pretrained_model(
    model_dir: str,
    version: Optional[int] = None,
    device: torch.device = None
) -> Tuple[torch.nn.Module, Optional[Dict], Optional[int]]:
    """
    Load a pretrained model from a specific version.
    
    Args:
        model_dir: Base directory containing models
        version: Specific version to load (None = latest)
        device: Device to load model on
    
    Returns:
        Tuple of (model, checkpoint_info, starting_epoch)
        checkpoint_info contains optimizer state, etc. if available
    """
    from scripts.test_real_data_v3 import find_latest_model_version
    
    model_dir = Path(model_dir)
    
    # Find the model directory
    if version is not None:
        model_path = model_dir / f"model_v{version}"
        if not model_path.exists():
            raise FileNotFoundError(f"Model version {version} not found in {model_dir}")
    else:
        model_path = find_latest_model_version(model_dir)
    
    print(f"\n{'=' * 60}")
    print(f"Loading Pretrained Model")
    print(f"{'=' * 60}")
    print(f"  Model directory: {model_path}")
    
    # Find model files
    model_state_file = None
    checkpoint_file = None
    
    for pattern in ["model_state_v*.pth", "model_state_v3.pth"]:
        matches = list(model_path.glob(pattern))
        if matches:
            model_state_file = matches[0]
            break
    
    for pattern in ["model_checkpoint_v*.pth", "model_checkpoint_v3.pth"]:
        matches = list(model_path.glob(pattern))
        if matches:
            checkpoint_file = matches[0]
            break
    
    # Load checkpoint if available (contains more info)
    checkpoint_info = None
    starting_epoch = 0
    
    if checkpoint_file:
        print(f"  Loading from checkpoint: {checkpoint_file.name}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        checkpoint_info = checkpoint
        starting_epoch = checkpoint.get('epoch', 0) + 1  # Next epoch to train
        print(f"  Resuming from epoch: {starting_epoch}")
    elif model_state_file:
        print(f"  Loading from state file: {model_state_file.name}")
        state_dict = torch.load(model_state_file, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model file found in {model_path}")
    
    # Load model config from checkpoint if available
    if checkpoint_info:
        input_length = checkpoint_info.get('additional_info', {}).get('input_length')
        n_classes = checkpoint_info.get('additional_info', {}).get('n_classes')
        
        if input_length and n_classes:
            print(f"  Model config from checkpoint:")
            print(f"    Input length: {input_length}")
            print(f"    Number of classes: {n_classes}")
    
    return state_dict, checkpoint_info, starting_epoch


def train_v3_model(
    npz_file: str,
    output_dir: str = "models/saved_models_v3",
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    use_existing_splits: bool = False,
    pretrained_model_dir: Optional[str] = None,
    pretrained_version: Optional[int] = None
):
    """
    Main training function for v3 data.
    
    Args:
        npz_file: Path to npz file
        output_dir: Base directory to save model and outputs (will create versioned subfolder)
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_existing_splits: Whether to use existing train/val/test splits
        pretrained_model_dir: Directory containing pretrained model (default: same as output_dir)
        pretrained_version: Specific version to load (None = latest, or train from scratch)
    """
    # Create base output directory
    base_output_path = Path(output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get next version number and create versioned directory
    version = get_next_model_version(base_output_path)
    output_path = base_output_path / f"model_v{version}"
    output_path.mkdir(exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print(f"Model Version: v{version}")
    print(f"Output Directory: {output_path}")
    print(f"{'=' * 60}")
    
    # Load data
    data_dict = load_v3_data(npz_file)
    
    # Prepare splits
    X_train, X_val, X_test, y_train, y_val, y_test, target_grid = prepare_train_val_test(
        data_dict, use_existing_splits=use_existing_splits
    )
    
    # Get number of classes
    n_classes = len(np.unique(data_dict['labels']))
    input_length = len(target_grid)
    
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)
    print(f"  Input length: {input_length}")
    print(f"  Number of classes: {n_classes}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    model = create_model(
        input_length=input_length,
        n_classes=n_classes,
        config={
            'n_channels': [32, 64, 128, 256],
            'kernel_sizes': [7, 5, 5, 3],
            'pool_sizes': [2, 2, 2, 2],
            'use_batch_norm': True,
            'dropout': 0.3,
            'fc_hidden': [128, 64],
            'use_ordinal_head': False
        }
    )
    model.to(device)
    
    # Load pretrained model if specified
    pretrained_state_dict = None
    checkpoint_info = None
    starting_epoch = 0
    
    if pretrained_version is not None or pretrained_model_dir:
        pretrained_dir = pretrained_model_dir if pretrained_model_dir else output_dir
        try:
            pretrained_state_dict, checkpoint_info, starting_epoch = load_pretrained_model(
                model_dir=pretrained_dir,
                version=pretrained_version,
                device=device
            )
            
            # Try to load weights (handle architecture mismatches gracefully)
            try:
                model.load_state_dict(pretrained_state_dict, strict=True)
                print(f"  ✓ Successfully loaded pretrained weights")
            except RuntimeError as e:
                print(f"  ⚠ Warning: Could not load all weights (architecture mismatch)")
                print(f"    Error: {str(e)[:100]}...")
                print(f"    Attempting partial load...")
                try:
                    model.load_state_dict(pretrained_state_dict, strict=False)
                    print(f"  ✓ Loaded compatible weights (some layers may be randomly initialized)")
                except Exception as e2:
                    print(f"  ✗ Failed to load weights: {e2}")
                    print(f"    Training from scratch instead")
                    pretrained_state_dict = None
                    checkpoint_info = None
                    starting_epoch = 0
        except Exception as e:
            print(f"  ✗ Warning: Could not load pretrained model: {e}")
            print(f"    Training from scratch instead")
            pretrained_state_dict = None
            checkpoint_info = None
            starting_epoch = 0
    
    if pretrained_state_dict is None:
        print(f"  Training from scratch (random initialization)")
    else:
        print(f"  Continuing training from epoch: {starting_epoch}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")
    
    # Create data loaders
    print("\n" + "=" * 60)
    print("Creating Data Loaders")
    print("=" * 60)
    
    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Setup training
    print("\n" + "=" * 60)
    print("Setting Up Training")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Load optimizer state from checkpoint if available
    if checkpoint_info and 'optimizer_state_dict' in checkpoint_info:
        try:
            optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
            print(f"  ✓ Loaded optimizer state from checkpoint")
        except Exception as e:
            print(f"  ⚠ Could not load optimizer state: {e}")
            print(f"    Using fresh optimizer state")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    
    print(f"  Loss: CrossEntropyLoss")
    print(f"  Optimizer: Adam (lr={learning_rate}, weight_decay=1e-4)")
    print(f"  Scheduler: ReduceLROnPlateau")
    print(f"  Early stopping: patience=10")
    
    # Train model
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    print("Press Ctrl+C to stop training early (model will be saved)")
    print()
    
    # Load previous training history if resuming
    previous_history = None
    if checkpoint_info and 'history' in checkpoint_info.get('additional_info', {}):
        previous_history = checkpoint_info['additional_info']['history']
        print(f"  ✓ Loaded previous training history ({len(previous_history.get('train_loss', []))} epochs)")
    
    # Initialize history in case of early interruption
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # If resuming, start with previous history
    if previous_history:
        history = previous_history.copy()
        print(f"\n  {'=' * 60}")
        print(f"  RESUMING TRAINING")
        print(f"  {'=' * 60}")
        print(f"  Previous training: {len(history['train_loss'])} epochs completed")
        print(f"  Starting from epoch: {starting_epoch}")
        print(f"  Will train for {n_epochs} more epochs")
        print(f"  Total epochs after training: {starting_epoch + n_epochs - 1}")
        if history['val_loss']:
            print(f"  Previous best val loss: {min(history['val_loss']):.4f}")
        print(f"  {'=' * 60}\n")
    
    try:
        history_new = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            early_stopping=early_stopping,
            verbose=True,
            initial_epoch=starting_epoch,
            previous_history=history if previous_history else None
        )
        
        # Merge histories if resuming
        if previous_history:
            # Append new history to previous
            for key in history.keys():
                if key in history_new:
                    history[key].extend(history_new[key])
        else:
            history = history_new
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Training interrupted by user (Ctrl+C)")
        print("=" * 60)
        print("Saving current model state...")
        
        # Save the model with current progress
        model_state_path = output_path / f"model_state_v{version}_interrupted.pth"
        torch.save(model.state_dict(), model_state_path)
        print(f"  Saved interrupted model state: {model_state_path}")
        
        # Also save checkpoint
        checkpoint_path = output_path / f"model_checkpoint_v{version}_interrupted.pth"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=len(history.get('train_loss', [])) if history else 0,
            loss=history.get('val_loss', [0.0])[-1] if history and history.get('val_loss') else 0.0,
            filepath=str(checkpoint_path),
            additional_info={
                'n_classes': n_classes,
                'input_length': input_length,
                'interrupted': True,
                'history': history  # Save training history for resuming
            }
        )
        print(f"  Saved interrupted checkpoint: {checkpoint_path}")
        
        # Save training history if available
        if history and (history.get('train_loss') or history.get('val_loss')):
            history_path = output_path / "training_history_v3_interrupted.json"
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"  Saved training history: {history_path}")
        
        print("\nModel saved! You can resume training later or use this model for inference.")
        print(f"Note: Use 'model_state_v{version}_interrupted.pth' or 'model_checkpoint_v{version}_interrupted.pth'")
        raise  # Re-raise to exit cleanly
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)
    
    from training import validate_epoch
    
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    test_acc = test_metrics['accuracy']
    test_loss = test_metrics['loss']
    
    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.2f}%")
    
    # Compute detailed metrics
    y_test_pred = test_metrics['predictions']
    y_test_true = test_metrics['labels']
    
    metrics = compute_metrics(y_test_true, y_test_pred, 
                             class_names=[data_dict['label_mapping'][i] for i in range(n_classes)])
    
    print(f"\n  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    
    # Print classification report
    print("\n  Classification Report:")
    print_classification_report(
        y_test_true, y_test_pred,
        class_names=[data_dict['label_mapping'][i] for i in range(n_classes)]
    )
    
    # Save model and outputs
    print("\n" + "=" * 60)
    print("Saving Model and Outputs")
    print("=" * 60)
    
    # Save model checkpoint
    model_path = output_path / f"model_checkpoint_v{version}.pth"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=len(history['train_loss']),
        loss=test_loss,
        filepath=str(model_path),
        additional_info={
            'n_classes': n_classes,
            'input_length': input_length,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'model_version': version,
            'history': history  # Save training history for resuming
        }
    )
    print(f"  Saved model checkpoint: {model_path}")
    
    # Save model state only
    model_state_path = output_path / f"model_state_v{version}.pth"
    torch.save(model.state_dict(), model_state_path)
    print(f"  Saved model state: {model_state_path}")
    
    # Save target grid
    target_grid_path = output_path / f"target_grid_v{version}.npy"
    np.save(target_grid_path, target_grid)
    print(f"  Saved target grid: {target_grid_path}")
    
    # Save class names
    class_names_path = output_path / f"class_names_v{version}.json"
    with open(class_names_path, 'w') as f:
        json.dump(data_dict['label_mapping'], f, indent=2)
    print(f"  Saved class names: {class_names_path}")
    
    # Save training history
    history_path = output_path / f"training_history_v{version}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Saved training history: {history_path}")
    
    # Save metrics
    metrics_path = output_path / f"test_metrics_v{version}.json"
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy types to native Python types."""
        # Check for numpy arrays FIRST (before scalars)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Check for numpy integer/float scalars (direct isinstance checks)
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        # Handle np.int_ separately (may not exist in NumPy 2.0+)
        try:
            if isinstance(obj, (np.int_, np.intc, np.intp)):
                return int(obj)
        except AttributeError:
            pass
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        # Handle np.float_ separately (removed in NumPy 2.0+)
        try:
            if isinstance(obj, np.float_):
                return float(obj)
        except AttributeError:
            pass
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Check for numpy scalars using dtype (for scalar values with dtype attribute)
        if hasattr(obj, 'dtype') and not isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.integer):
                return int(obj)
            elif np.issubdtype(obj.dtype, np.floating):
                return float(obj)
            elif np.issubdtype(obj.dtype, np.bool_):
                return bool(obj)
        # Recursively handle dictionaries
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        # Recursively handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    metrics_serializable = convert_to_serializable(metrics)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"  Saved test metrics: {metrics_path}")
    
    # Plot training history
    history_plot_path = output_path / f"training_history_v{version}.png"
    plot_training_history(history, save_path=str(history_plot_path))
    print(f"  Saved training history plot: {history_plot_path}")
    
    # Plot confusion matrix
    cm_path = output_path / f"confusion_matrix_v{version}.png"
    plot_confusion_matrix(
        y_test_true, y_test_pred,
        class_names=[data_dict['label_mapping'][i] for i in range(n_classes)],
        save_path=str(cm_path)
    )
    print(f"  Saved confusion matrix: {cm_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel Version: v{version}")
    print(f"All outputs saved to: {output_path}")
    
    return model, history, metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN model on v3 data")
    parser.add_argument(
        "--data", type=str,
        default="data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz",
        help="Path to npz file"
    )
    parser.add_argument(
        "--output", type=str,
        default="models/saved_models_v3",
        help="Output directory for saved model"
    )
    parser.add_argument(
        "--epochs", type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--use-existing-splits", action="store_true",
        help="Use existing train/val/test splits from metadata if available"
    )
    parser.add_argument(
        "--pretrained-model", type=str, default=None,
        help="Directory containing pretrained model (default: same as --output)"
    )
    parser.add_argument(
        "--pretrained-version", type=int, default=None,
        help="Specific model version to load (e.g., 3). If not specified and --pretrained-model is set, uses latest version"
    )
    parser.add_argument(
        "--from-checkpoint", type=int, default=None,
        metavar="VERSION",
        help="Shortcut: Load latest checkpoint from --output directory, specify version number"
    )
    
    args = parser.parse_args()
    
    # Handle --from-checkpoint shortcut
    pretrained_model_dir = args.pretrained_model
    pretrained_version = args.pretrained_version
    
    if args.from_checkpoint is not None:
        pretrained_model_dir = args.output
        pretrained_version = args.from_checkpoint
        print(f"Using checkpoint from version {pretrained_version} in {pretrained_model_dir}")
    
    train_v3_model(
        npz_file=args.data,
        output_dir=args.output,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_existing_splits=args.use_existing_splits,
        pretrained_model_dir=pretrained_model_dir,
        pretrained_version=pretrained_version
    )

