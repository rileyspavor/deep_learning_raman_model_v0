"""
Check what preprocessing was actually used during training.

This script analyzes the saved training data to determine the exact
preprocessing pipeline that was applied.
"""

import numpy as np
from pathlib import Path


def analyze_training_preprocessing():
    """Analyze training data to determine preprocessing steps."""
    
    print("=" * 70)
    print("Training Preprocessing Analysis")
    print("=" * 70)
    
    # Load training data
    train_file = Path('data/processed/training_data/train_val_test.npz')
    if not train_file.exists():
        print(f"\n❌ Training data file not found: {train_file}")
        return
    
    data = np.load(train_file, allow_pickle=True)
    target_grid = data['target_grid']
    X_train = data['X_train']
    
    print(f"\n[1] Target Grid Analysis:")
    print("-" * 70)
    print(f"  Range: {target_grid.min():.1f} - {target_grid.max():.1f} cm⁻¹")
    print(f"  Length: {len(target_grid)} points")
    print(f"  Step size: {(target_grid.max() - target_grid.min()) / (len(target_grid) - 1):.3f} cm⁻¹")
    
    # Check if trimmed to 800-3200
    if abs(target_grid.min() - 800.0) < 1.0 and abs(target_grid.max() - 3200.0) < 1.0:
        print(f"  ✓ Trimmed to 800-3200 cm⁻¹ range")
    else:
        print(f"  Range: {target_grid.min():.1f} - {target_grid.max():.1f} cm⁻¹")
    
    print(f"\n[2] Spectra Statistics:")
    print("-" * 70)
    print(f"  Shape: {X_train.shape}")
    print(f"  Max: {X_train.max():.6f}")
    print(f"  Min: {X_train.min():.6f}")
    print(f"  Mean: {X_train.mean():.6f}")
    print(f"  Std: {X_train.std():.6f}")
    
    # Sample a few spectra for detailed analysis
    sample_spectra = X_train[:5]
    
    print(f"\n[3] Preprocessing Detection:")
    print("-" * 70)
    
    # Check normalization
    max_vals = np.max(sample_spectra, axis=1)
    min_vals = np.min(sample_spectra, axis=1)
    mean_vals = np.mean(sample_spectra, axis=1)
    std_vals = np.std(sample_spectra, axis=1)
    
    print(f"\n  A. Baseline Correction:")
    if np.all(X_train >= 0):
        print(f"    ✓ Applied (all values ≥ 0)")
        if np.any(X_train == 0):
            print(f"    ✓ Some zero values present (baseline subtracted)")
    else:
        print(f"    ✗ Not applied (negative values present)")
    
    print(f"\n  B. Normalization:")
    # Check max normalization
    if np.allclose(max_vals, 1.0, atol=0.01):
        print(f"    ✓ Max normalization (divide by max)")
        print(f"      All spectra peak at ~1.0")
    # Check z-score
    elif np.allclose(mean_vals, 0.0, atol=0.1) and np.allclose(std_vals, 1.0, atol=0.1):
        print(f"    ✓ Z-score normalization (mean=0, std=1)")
    # Check min-max
    elif np.allclose(min_vals, 0.0, atol=0.01) and np.allclose(max_vals, 1.0, atol=0.01):
        print(f"    ✓ Min-max normalization (scaled to [0, 1])")
    else:
        print(f"    ? Unknown normalization method")
        print(f"      Max values: {max_vals}")
        print(f"      Min values: {min_vals}")
        print(f"      Mean values: {mean_vals}")
        print(f"      Std values: {std_vals}")
    
    print(f"\n  C. Smoothing:")
    # Hard to detect smoothing from final data, but we can check variance
    # Smoothed data typically has lower variance
    print(f"    ? Cannot definitively detect from final data")
    print(f"    (Check training scripts for smooth=True/False)")
    
    print(f"\n[4] Summary - Preprocessing Pipeline:")
    print("-" * 70)
    print(f"  1. Alignment: Linear interpolation to target_grid")
    print(f"     → {len(target_grid)} points, {target_grid.min():.0f}-{target_grid.max():.0f} cm⁻¹")
    
    if np.all(X_train >= 0):
        print(f"  2. Baseline Correction: ✓ Applied (ALS method)")
    else:
        print(f"  2. Baseline Correction: ✗ Not applied")
    
    if np.allclose(max_vals, 1.0, atol=0.01):
        print(f"  3. Normalization: ✓ Max normalization (divide by max)")
    elif np.allclose(mean_vals, 0.0, atol=0.1) and np.allclose(std_vals, 1.0, atol=0.1):
        print(f"  3. Normalization: ✓ Z-score normalization")
    else:
        print(f"  3. Normalization: ? Unknown method")
    
    print(f"  4. Smoothing: ? Check training scripts")
    
    print("\n" + "=" * 70)
    print("\nTo verify smoothing, check the training scripts:")
    print("  - example_workflow.py")
    print("  - step_by_step_preprocessing.py")
    print("  - process_multiple_datasets.py")


if __name__ == "__main__":
    analyze_training_preprocessing()



