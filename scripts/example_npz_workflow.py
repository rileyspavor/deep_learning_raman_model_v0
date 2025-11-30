"""
Example Workflow for .npz Dataset Format

This script demonstrates how to load and preprocess data from .npz files.
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.data_ingestion import load_npz_dataset, convert_npz_to_list_format
from src.preprocessing import preprocess_aligned_spectra, preprocess_dataset
from src.utils import stratified_split


def load_and_explore_npz(file_path: str):
    """
    Load .npz file and explore its contents.
    
    Args:
        file_path: Path to .npz file
    """
    print("=" * 60)
    print("Loading .npz Dataset")
    print("=" * 60)
    
    # Load the dataset
    spectra, wavenumbers, labels, label_names, metadata = load_npz_dataset(
        file_path=file_path,
        spectra_key="spectra",
        wavenumbers_key="wavenumbers",
        labels_key="y",
        label_names_key="label_names"
    )
    
    print(f"\nDataset loaded successfully!")
    print(f"  Spectra shape: {spectra.shape}")
    print(f"  Wavenumbers shape: {wavenumbers.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Number of classes: {len(np.unique(labels))}")
    
    if label_names is not None:
        print(f"  Label names: {label_names}")
        # Create label mapping
        unique_labels = np.unique(labels)
        label_mapping = {int(label): str(label_names[int(label)]) for label in unique_labels}
        print(f"  Label mapping: {label_mapping}")
    else:
        unique_labels = np.unique(labels)
        label_mapping = {int(label): f"Class_{label}" for label in unique_labels}
        print(f"  Label mapping: {label_mapping}")
    
    # Show metadata
    if metadata:
        print(f"\n  Additional metadata keys: {list(metadata.keys())}")
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"    {key}: {type(value).__name__}")
    
    # Check if spectra are already aligned
    if wavenumbers.ndim == 1:
        print(f"\n  ✓ Spectra are already aligned on common wavenumber grid")
        print(f"    Wavenumber range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
    else:
        print(f"\n  ⚠ Spectra have different wavenumber grids (will need alignment)")
    
    return spectra, wavenumbers, labels, label_names, metadata, label_mapping


def preprocess_npz_data(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    already_aligned: bool = True
):
    """
    Preprocess data from .npz file.
    
    Args:
        spectra: Array of shape (n_samples, n_wavenumbers)
        wavenumbers: Array of shape (n_wavenumbers,) or (n_samples, n_wavenumbers)
        already_aligned: Whether spectra are already on the same grid
    """
    print("\n" + "=" * 60)
    print("Preprocessing Data")
    print("=" * 60)
    
    if already_aligned and wavenumbers.ndim == 1:
        # Use efficient preprocessing for aligned data
        print("\nUsing aligned preprocessing (more efficient)...")
        target_grid, processed_spectra = preprocess_aligned_spectra(
            spectra=spectra,
            wavenumbers=wavenumbers,
            align=False,  # Already aligned
            baseline_correct=True,
            baseline_method='als',
            normalize=True,
            normalize_method='max',
            smooth=False
        )
    else:
        # Convert to list format and use standard preprocessing
        print("\nConverting to list format and preprocessing...")
        wavenumbers_list, intensities_list, _ = convert_npz_to_list_format(
            spectra, wavenumbers, labels=np.zeros(spectra.shape[0])  # Dummy labels
        )
        
        target_grid, processed_spectra = preprocess_dataset(
            wavenumbers_list=wavenumbers_list,
            intensities_list=intensities_list,
            target_grid=wavenumbers if wavenumbers.ndim == 1 else None,
            align=True,
            baseline_correct=True,
            baseline_method='als',
            normalize=True,
            normalize_method='max',
            smooth=False
        )
    
    print(f"\nPreprocessing complete!")
    print(f"  Processed spectra shape: {processed_spectra.shape}")
    print(f"  Target grid length: {len(target_grid)}")
    print(f"  Wavenumber range: {target_grid.min():.1f} - {target_grid.max():.1f} cm⁻¹")
    
    return target_grid, processed_spectra


def main():
    """Main workflow for .npz file."""
    
    # File path - UPDATE THIS to your .npz file path
    npz_file = "goldie_graphitic_synthetic_dataset.npz"
    
    try:
        # Step 1: Load and explore
        spectra, wavenumbers, labels, label_names, metadata, label_mapping = load_and_explore_npz(npz_file)
        
        # Step 2: Preprocess
        already_aligned = wavenumbers.ndim == 1
        target_grid, processed_spectra = preprocess_npz_data(
            spectra, wavenumbers, already_aligned=already_aligned
        )
        
        # Step 3: Split data (ready for training)
        print("\n" + "=" * 60)
        print("Splitting Data")
        print("=" * 60)
        
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
            processed_spectra, labels,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        print(f"\nData split complete!")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Step 4: Display class distribution
        print(f"\nClass distribution (train set):")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            class_name = label_mapping.get(int(label), f"Class_{label}")
            print(f"  {class_name}: {count} samples")
        
        # Save metadata for later use
        print(f"\n" + "=" * 60)
        print("Ready for Training!")
        print("=" * 60)
        print(f"\nYou can now use:")
        print(f"  - X_train, X_val, X_test: Preprocessed spectra")
        print(f"  - y_train, y_val, y_test: Labels")
        print(f"  - target_grid: Wavenumber grid")
        print(f"  - label_mapping: Class name mapping")
        
        # Return data for use in training script
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'target_grid': target_grid,
            'label_mapping': label_mapping,
            'label_names': label_names,
            'metadata': metadata
        }
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{npz_file}' not found!")
        print(f"Please update the 'npz_file' variable in the script with your file path.")
    except KeyError as e:
        print(f"\n❌ Error: Missing key in .npz file: {e}")
        print(f"Please check that your .npz file contains the expected keys:")
        print(f"  - 'spectra' (or specify spectra_key)")
        print(f"  - 'wavenumbers' (or specify wavenumbers_key)")
        print(f"  - 'y' (or specify labels_key)")
        print(f"  - 'label_names' (optional)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    data = main()


