"""
Step-by-Step Preprocessing for Multiple Datasets

This script allows you to process multiple .npz datasets one at a time,
format and normalize them, then combine them for training.
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.data_ingestion import load_npz_dataset
from src.preprocessing import preprocess_aligned_spectra
from src.utils import stratified_split
import json


# ============================================================================
# STEP 1: Load and Explore Each Dataset
# ============================================================================
def step1_load_dataset(file_path: str, dataset_name: str = None):
    """
    Step 1: Load a single .npz dataset and explore its contents.
    
    Args:
        file_path: Path to .npz file
        dataset_name: Optional name for this dataset
    
    Returns:
        Dictionary with loaded data and metadata
    """
    if dataset_name is None:
        dataset_name = Path(file_path).stem
    
    print("=" * 70)
    print(f"STEP 1: Loading Dataset - {dataset_name}")
    print("=" * 70)
    
    try:
        # Load the dataset
        spectra, wavenumbers, labels, label_names, metadata = load_npz_dataset(
            file_path=file_path,
            spectra_key="spectra",
            wavenumbers_key="wavenumbers",
            labels_key="y",
            label_names_key="label_names"
        )
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  File: {file_path}")
        print(f"  Spectra shape: {spectra.shape}")
        print(f"  Wavenumbers shape: {wavenumbers.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Number of samples: {len(spectra)}")
        print(f"  Number of classes: {len(np.unique(labels))}")
        
        # Check alignment
        if wavenumbers.ndim == 1:
            print(f"  ✓ Spectra are aligned on common grid")
            print(f"    Wavenumber range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
            already_aligned = True
        else:
            print(f"  ⚠ Spectra have different wavenumber grids")
            already_aligned = False
        
        # Show label information
        unique_labels = np.unique(labels)
        if label_names is not None:
            label_mapping = {int(label): str(label_names[int(label)]) for label in unique_labels}
            print(f"\n  Label mapping:")
            for label in unique_labels:
                print(f"    {int(label)}: {label_mapping[int(label)]}")
        else:
            label_mapping = {int(label): f"Class_{label}" for label in unique_labels}
            print(f"\n  Label indices: {unique_labels.tolist()}")
        
        # Show class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n  Class distribution:")
        for label, count in zip(unique, counts):
            name = label_mapping.get(int(label), f"Class_{label}")
            print(f"    {name}: {count} samples")
        
        # Show metadata
        if metadata:
            print(f"\n  Additional metadata:")
            for key, value in metadata.items():
                if isinstance(value, np.ndarray):
                    print(f"    {key}: shape {value.shape}, dtype {value.dtype}")
        
        return {
            'dataset_name': dataset_name,
            'file_path': file_path,
            'spectra': spectra,
            'wavenumbers': wavenumbers,
            'labels': labels,
            'label_names': label_names,
            'metadata': metadata,
            'label_mapping': label_mapping,
            'already_aligned': already_aligned
        }
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{file_path}' not found!")
        return None
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# STEP 2: Preprocess Each Dataset
# ============================================================================
def step2_preprocess_dataset(
    dataset_dict: dict,
    baseline_correct: bool = True,
    baseline_method: str = "als",
    normalize: bool = True,
    normalize_method: str = "max",
    smooth: bool = False
):
    """
    Step 2: Preprocess a loaded dataset.
    
    Args:
        dataset_dict: Dictionary from step1_load_dataset()
        baseline_correct: Whether to correct baseline
        baseline_method: Baseline correction method
        normalize: Whether to normalize
        normalize_method: Normalization method
        smooth: Whether to smooth
    
    Returns:
        Updated dataset_dict with processed_spectra and target_grid
    """
    dataset_name = dataset_dict['dataset_name']
    
    print("\n" + "=" * 70)
    print(f"STEP 2: Preprocessing Dataset - {dataset_name}")
    print("=" * 70)
    
    spectra = dataset_dict['spectra']
    wavenumbers = dataset_dict['wavenumbers']
    already_aligned = dataset_dict['already_aligned']
    
    print(f"\nPreprocessing configuration:")
    print(f"  Baseline correction: {baseline_correct} ({baseline_method if baseline_correct else 'N/A'})")
    print(f"  Normalization: {normalize} ({normalize_method if normalize else 'N/A'})")
    print(f"  Smoothing: {smooth}")
    
    if already_aligned:
        print(f"\nUsing aligned preprocessing (efficient)...")
        target_grid, processed_spectra = preprocess_aligned_spectra(
            spectra=spectra,
            wavenumbers=wavenumbers,
            align=False,
            baseline_correct=baseline_correct,
            baseline_method=baseline_method,
            normalize=normalize,
            normalize_method=normalize_method,
            smooth=smooth
        )
    else:
        print(f"\n⚠ Warning: Spectra not aligned. Using list-based preprocessing...")
        from data_ingestion import convert_npz_to_list_format
        from preprocessing import preprocess_dataset
        
        wavenumbers_list, intensities_list, _ = convert_npz_to_list_format(
            spectra, wavenumbers, labels=np.zeros(spectra.shape[0])
        )
        
        target_grid, processed_spectra = preprocess_dataset(
            wavenumbers_list=wavenumbers_list,
            intensities_list=intensities_list,
            target_grid=wavenumbers if wavenumbers.ndim == 1 else None,
            align=True,
            baseline_correct=baseline_correct,
            baseline_method=baseline_method,
            normalize=normalize,
            normalize_method=normalize_method,
            smooth=smooth
        )
    
    print(f"\n✓ Preprocessing complete!")
    print(f"  Processed spectra shape: {processed_spectra.shape}")
    print(f"  Target grid length: {len(target_grid)}")
    print(f"  Wavenumber range: {target_grid.min():.1f} - {target_grid.max():.1f} cm⁻¹")
    
    # Update dataset dict
    dataset_dict['processed_spectra'] = processed_spectra
    dataset_dict['target_grid'] = target_grid
    
    return dataset_dict


# ============================================================================
# STEP 3: Combine Multiple Datasets
# ============================================================================
def step3_combine_datasets(dataset_list: list, align_grids: bool = True):
    """
    Step 3: Combine multiple preprocessed datasets.
    
    Args:
        dataset_list: List of dataset dictionaries (from step2)
        align_grids: Whether to align all datasets to a common grid
    
    Returns:
        Combined dataset dictionary
    """
    print("\n" + "=" * 70)
    print(f"STEP 3: Combining {len(dataset_list)} Datasets")
    print("=" * 70)
    
    if len(dataset_list) == 0:
        print("❌ Error: No datasets to combine!")
        return None
    
    # Check if all datasets have the same grid
    first_grid = dataset_list[0]['target_grid']
    all_same_grid = all(
        np.allclose(d['target_grid'], first_grid, atol=0.1)
        for d in dataset_list[1:]
    )
    
    if all_same_grid:
        print(f"\n✓ All datasets use the same wavenumber grid")
        target_grid = first_grid
    else:
        print(f"\n⚠ Datasets have different wavenumber grids")
        if align_grids:
            print(f"  Aligning to common grid...")
            # Find common range
            all_min = min(d['target_grid'].min() for d in dataset_list)
            all_max = max(d['target_grid'].max() for d in dataset_list)
            # Use finest resolution
            all_resolutions = [np.mean(np.diff(d['target_grid'])) for d in dataset_list]
            resolution = min(all_resolutions)
            target_grid = np.arange(all_min, all_max + resolution, resolution)
            
            # Interpolate each dataset to common grid
            from preprocessing import align_spectrum
            for i, dataset in enumerate(dataset_list):
                aligned_spectra = []
                for spectrum in dataset['processed_spectra']:
                    aligned = align_spectrum(
                        dataset['target_grid'], spectrum, target_grid
                    )
                    aligned_spectra.append(aligned)
                dataset['processed_spectra'] = np.array(aligned_spectra)
                dataset['target_grid'] = target_grid
                print(f"  ✓ Dataset {i+1} aligned")
        else:
            print(f"  Using first dataset's grid")
            target_grid = first_grid
    
    # Combine spectra and labels
    all_spectra = []
    all_labels = []
    all_label_mappings = []
    
    for i, dataset in enumerate(dataset_list):
        spectra = dataset['processed_spectra']
        labels = dataset['labels']
        label_mapping = dataset['label_mapping']
        
        all_spectra.append(spectra)
        all_labels.append(labels)
        all_label_mappings.append({
            'dataset': dataset['dataset_name'],
            'mapping': label_mapping
        })
        
        print(f"\n  Dataset {i+1}: {dataset['dataset_name']}")
        print(f"    Samples: {len(spectra)}")
        print(f"    Classes: {len(np.unique(labels))}")
    
    combined_spectra = np.vstack(all_spectra)
    combined_labels = np.concatenate(all_labels)
    
    # Create unified label mapping
    # Find all unique label names across datasets
    all_label_names = set()
    for mapping_dict in all_label_mappings:
        for label_idx, label_name in mapping_dict['mapping'].items():
            all_label_names.add(label_name)
    
    # Create new integer mapping for combined dataset
    unique_names = sorted(all_label_names)
    unified_mapping = {i: name for i, name in enumerate(unique_names)}
    reverse_mapping = {name: i for i, name in enumerate(unique_names)}
    
    # Remap labels
    remapped_labels = []
    label_offset = 0
    for dataset_idx, dataset in enumerate(dataset_list):
        labels = dataset['labels']
        label_mapping = dataset['label_mapping']
        
        for label in labels:
            old_name = label_mapping[int(label)]
            new_label = reverse_mapping[old_name]
            remapped_labels.append(new_label)
        
        label_offset += len(np.unique(labels))
    
    remapped_labels = np.array(remapped_labels)
    
    print(f"\n✓ Datasets combined!")
    print(f"  Total samples: {len(combined_spectra)}")
    print(f"  Total classes: {len(unique_names)}")
    print(f"  Combined spectra shape: {combined_spectra.shape}")
    print(f"  Target grid length: {len(target_grid)}")
    
    print(f"\n  Unified label mapping:")
    for idx, name in unified_mapping.items():
        count = np.sum(remapped_labels == idx)
        print(f"    {idx}: {name} ({count} samples)")
    
    return {
        'spectra': combined_spectra,
        'labels': remapped_labels,
        'target_grid': target_grid,
        'label_mapping': unified_mapping,
        'dataset_info': all_label_mappings
    }


# ============================================================================
# STEP 4: Split Combined Dataset
# ============================================================================
def step4_split_dataset(combined_dict: dict, train_ratio: float = 0.7, 
                        val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Step 4: Split combined dataset into train/val/test sets.
    
    Args:
        combined_dict: Dictionary from step3_combine_datasets()
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    
    Returns:
        Dictionary with split data
    """
    print("\n" + "=" * 70)
    print(f"STEP 4: Splitting Dataset")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        combined_dict['spectra'],
        combined_dict['labels'],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    print(f"\n✓ Data split complete!")
    print(f"  Train: {len(X_train)} samples ({train_ratio*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({val_ratio*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({test_ratio*100:.1f}%)")
    
    # Show class distribution in each split
    print(f"\n  Class distribution (train set):")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        name = combined_dict['label_mapping'][int(label)]
        print(f"    {name}: {count} samples")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'target_grid': combined_dict['target_grid'],
        'label_mapping': combined_dict['label_mapping']
    }


# ============================================================================
# MAIN WORKFLOW - Run Steps One at a Time
# ============================================================================
def main():
    """
    Main workflow - process datasets step by step.
    Modify the dataset paths and run each step manually.
    """
    
    # ========================================================================
    # CONFIGURATION: Update these paths to your 4 datasets
    # ========================================================================
    dataset_paths = [
        "goldie_graphitic_synthetic_dataset.npz",  # Dataset 1
        # "dataset2.npz",  # Dataset 2
        # "dataset3.npz",  # Dataset 3
        # "dataset4.npz",  # Dataset 4
    ]
    
    # Preprocessing configuration (same for all datasets)
    preprocessing_config = {
        'baseline_correct': True,
        'baseline_method': 'als',
        'normalize': True,
        'normalize_method': 'max',
        'smooth': False
    }
    
    # ========================================================================
    # STEP 1: Load each dataset (run one at a time)
    # ========================================================================
    print("\n" + "=" * 70)
    print("LOADING DATASETS - Run Step 1 for each dataset")
    print("=" * 70)
    
    loaded_datasets = []
    
    for i, file_path in enumerate(dataset_paths):
        print(f"\n>>> Processing Dataset {i+1}/{len(dataset_paths)}")
        dataset = step1_load_dataset(file_path, f"dataset_{i+1}")
        if dataset is not None:
            loaded_datasets.append(dataset)
            print(f"\n✓ Dataset {i+1} loaded. Ready for preprocessing.")
            input("Press Enter to continue to next dataset (or Ctrl+C to stop)...")
    
    if not loaded_datasets:
        print("\n❌ No datasets loaded successfully!")
        return
    
    # ========================================================================
    # STEP 2: Preprocess each dataset (run one at a time)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PREPROCESSING DATASETS - Run Step 2 for each dataset")
    print("=" * 70)
    
    preprocessed_datasets = []
    
    for i, dataset in enumerate(loaded_datasets):
        print(f"\n>>> Preprocessing Dataset {i+1}/{len(loaded_datasets)}")
        processed = step2_preprocess_dataset(dataset, **preprocessing_config)
        preprocessed_datasets.append(processed)
        print(f"\n✓ Dataset {i+1} preprocessed. Ready for next step.")
        input("Press Enter to continue to next dataset (or Ctrl+C to stop)...")
    
    # ========================================================================
    # STEP 3: Combine all datasets
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMBINING DATASETS")
    print("=" * 70)
    
    combined = step3_combine_datasets(preprocessed_datasets, align_grids=True)
    
    if combined is None:
        print("\n❌ Failed to combine datasets!")
        return
    
    # ========================================================================
    # STEP 4: Split combined dataset
    # ========================================================================
    print("\n" + "=" * 70)
    print("SPLITTING DATASET")
    print("=" * 70)
    
    split_data = step4_split_dataset(combined)
    
    # ========================================================================
    # Save preprocessed data for training
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING PREPROCESSED DATA")
    print("=" * 70)
    
    output_dir = Path("preprocessed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save as .npz
    np.savez(
        output_dir / "combined_preprocessed.npz",
        X_train=split_data['X_train'],
        X_val=split_data['X_val'],
        X_test=split_data['X_test'],
        y_train=split_data['y_train'],
        y_val=split_data['y_val'],
        y_test=split_data['y_test'],
        target_grid=split_data['target_grid']
    )
    
    # Save label mapping
    with open(output_dir / "label_mapping.json", 'w') as f:
        json.dump(split_data['label_mapping'], f, indent=2)
    
    print(f"\n✓ Preprocessed data saved to: {output_dir}/")
    print(f"  - combined_preprocessed.npz")
    print(f"  - label_mapping.json")
    
    print("\n" + "=" * 70)
    print("READY FOR TRAINING!")
    print("=" * 70)
    print(f"\nYou can now load the preprocessed data:")
    print(f"  data = np.load('preprocessed_data/combined_preprocessed.npz')")
    print(f"  X_train = data['X_train']")
    print(f"  y_train = data['y_train']")
    print(f"  # ... etc")


if __name__ == "__main__":
    main()


