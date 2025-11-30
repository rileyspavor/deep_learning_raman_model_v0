"""
Example Workflow: Complete pipeline from data generation to inference

This script demonstrates the full workflow for training and using
a Raman spectroscopy classification model.
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

# Import our modules
from src.data_ingestion import generate_synthetic_dataset, save_label_mapping
from src.preprocessing import preprocess_dataset
from src.model import create_model
from src.training import train_model, create_dataloader, EarlyStopping
from src.inference import RamanSpectrumClassifier, save_classifier
from src.utils import stratified_split, compute_metrics, plot_confusion_matrix, plot_training_history


def main():
    """Complete example workflow."""
    
    print("=" * 60)
    print("Raman Spectroscopy Classification - Example Workflow")
    print("=" * 60)
    
    # ===================================================================
    # Step 1: Generate Synthetic Data
    # ===================================================================
    print("\n[Step 1] Generating synthetic Raman spectra...")
    
    # Define wavenumber grid (800-3200 cm^-1, typical for graphene)
    wavenumber_grid = np.arange(800, 3201, 1.0)
    print(f"Wavenumber grid: {wavenumber_grid.min():.0f} - {wavenumber_grid.max():.0f} cm⁻¹")
    
    # Define class labels
    class_labels = {
        'graphite': 0,
        'graphene': 1,
        'GO': 2,
        'rGO': 3,
        'GNP': 4
    }
    
    # Generate synthetic dataset
    spectra, labels = generate_synthetic_dataset(
        wavenumber_grid=wavenumber_grid,
        n_samples_per_class=200,  # 200 samples per class
        class_labels=class_labels,
        noise_level=0.02,
        baseline_level=0.1,
        peak_variation=0.1
    )
    
    print(f"Generated {len(spectra)} spectra")
    print(f"Spectrum shape: {spectra.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Save label mapping for later use
    save_label_mapping(class_labels, 'data/class_labels.json')
    
    # ===================================================================
    # Step 2: Preprocess Data
    # ===================================================================
    print("\n[Step 2] Preprocessing spectra...")
    
    # Convert to list format (as expected by preprocess_dataset)
    wavenumbers_list = [wavenumber_grid] * len(spectra)
    intensities_list = list(spectra)
    
    # Preprocess
    target_grid, processed_spectra = preprocess_dataset(
        wavenumbers_list=wavenumbers_list,
        intensities_list=intensities_list,
        target_grid=wavenumber_grid,  # Use same grid
        align=True,
        baseline_correct=True,
        baseline_method='als',
        normalize=True,
        normalize_method='max',
        smooth=False,  # No smoothing for this example
    )
    
    print(f"Preprocessed spectra shape: {processed_spectra.shape}")
    print(f"Target grid length: {len(target_grid)}")
    
    # ===================================================================
    # Step 3: Split Data
    # ===================================================================
    print("\n[Step 3] Splitting data into train/val/test sets...")
    
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        processed_spectra, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # ===================================================================
    # Step 4: Create Data Loaders
    # ===================================================================
    print("\n[Step 4] Creating data loaders...")
    
    train_loader = create_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=32, shuffle=False)
    test_loader = create_dataloader(X_test, y_test, batch_size=32, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # ===================================================================
    # Step 5: Create Model
    # ===================================================================
    print("\n[Step 5] Creating 1D CNN model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_model(
        input_length=len(target_grid),
        n_classes=len(class_labels),
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} trainable parameters")
    
    # ===================================================================
    # Step 6: Setup Training
    # ===================================================================
    print("\n[Step 6] Setting up training...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    
    print("Optimizer: Adam (lr=0.001, weight_decay=1e-4)")
    print("Loss: CrossEntropyLoss")
    print("Scheduler: ReduceLROnPlateau")
    print("Early stopping: patience=10")
    
    # ===================================================================
    # Step 7: Train Model
    # ===================================================================
    print("\n[Step 7] Training model...")
    print("-" * 60)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=50,  # Reduced for example
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        early_stopping=early_stopping,
        verbose=True
    )
    
    print("-" * 60)
    print("Training completed!")
    
    # ===================================================================
    # Step 8: Evaluate on Test Set
    # ===================================================================
    print("\n[Step 8] Evaluating on test set...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for spectra_batch, labels_batch in test_loader:
            spectra_batch = spectra_batch.to(device)
            logits, _ = model(spectra_batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    class_names = list(class_labels.keys())
    metrics = compute_metrics(all_labels, all_preds, class_names=class_names)
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Macro F1: {metrics['macro_f1']:.3f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.3f}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        all_labels, all_preds,
        class_names=class_names,
        save_path='confusion_matrix.png'
    )
    
    # Plot training history
    print("Generating training history plot...")
    plot_training_history(history, save_path='training_history.png')
    
    # ===================================================================
    # Step 9: Create Inference Pipeline
    # ===================================================================
    print("\n[Step 9] Creating inference pipeline...")
    
    classifier = RamanSpectrumClassifier(
        model=model,
        target_grid=target_grid,
        class_names={v: k for k, v in class_labels.items()},
        device=device,
        preprocessing_config={
            'baseline_correct': True,
            'baseline_method': 'als',
            'normalize': True,
            'normalize_method': 'max'
        }
    )
    
    # Test inference on a sample from test set
    sample_idx = 0
    sample_spectrum = X_test[sample_idx]
    sample_label = y_test[sample_idx]
    
    # Reconstruct wavenumbers and intensities for inference
    sample_wavenumbers = target_grid
    sample_intensities = sample_spectrum
    
    result = classifier.predict(sample_wavenumbers, sample_intensities)
    
    print(f"\nInference Example:")
    print(f"  True label: {class_names[sample_label]}")
    print(f"  Predicted: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    
    # Quality check
    qc_result = classifier.quality_check(
        sample_wavenumbers, sample_intensities,
        confidence_threshold=0.7
    )
    print(f"  QC Passed: {qc_result['qc_passed']}")
    
    # ===================================================================
    # Step 10: Save Model
    # ===================================================================
    print("\n[Step 10] Saving model and classifier...")
    
    # Save classifier
    save_classifier(
        classifier,
        model_path='trained_model.pth',
        target_grid_path='target_grid.npy',
        class_names_path='class_names.json'
    )
    
    print("Model saved successfully!")
    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)
    print("\nFiles created:")
    print("  - trained_model.pth (model weights)")
    print("  - target_grid.npy (wavenumber grid)")
    print("  - class_names.json (class name mapping)")
    print("  - data/class_labels.json (label mapping)")
    print("  - confusion_matrix.png (confusion matrix plot)")
    print("  - training_history.png (training curves)")


if __name__ == "__main__":
    main()

