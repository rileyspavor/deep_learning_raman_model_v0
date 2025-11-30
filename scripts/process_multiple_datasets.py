"""
Interactive Script to Process Multiple Datasets Step-by-Step

This script allows you to:
1. Load each of your 4 datasets one at a time
2. Preprocess each dataset individually
3. Combine all datasets
4. Split for training

Run each step manually to verify everything works.
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
# CONFIGURATION: Update these with your 4 dataset paths
# ============================================================================
DATASET_PATHS = [
    "full_training_raman_labels/goldie_graphitic_synthetic_dataset.npz",
    "full_training_raman_labels/synthetic_go_dataset.npz",
    "full_training_raman_labels/final_combined_graphene_dataset.npz",
    "full_training_raman_labels/rgo_graphitization_small_synthetic_dataset.npz",
]

PREPROCESSING_CONFIG = {
    'baseline_correct': True,
    'baseline_method': 'als',
    'normalize': True,
    'normalize_method': 'max',
    'smooth': True
}


def load_single_dataset(file_path: str):
    """Load and display info about one dataset."""
    print("\n" + "="*70)
    print(f"Loading: {file_path}")
    print("="*70)
    
    # Check if file exists
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        print(f"\n❌ File not found: {file_path}")
        print(f"   Current directory: {Path.cwd()}")
        print(f"   Looking for: {file_path_obj.absolute()}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    spectra, wavenumbers, labels, label_names, metadata = load_npz_dataset(file_path)
    
    print(f"\n✓ Loaded successfully!")
    print(f"  Spectra: {spectra.shape}")
    print(f"  Wavenumbers: {wavenumbers.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Classes: {len(np.unique(labels))}")
    
    if label_names is not None:
        unique_labels = np.unique(labels)
        label_mapping = {int(l): str(label_names[int(l)]) for l in unique_labels}
        print(f"  Label names: {label_mapping}")
    else:
        print(f"  Label indices: {np.unique(labels).tolist()}")
    
    if wavenumbers.ndim == 1:
        print(f"  ✓ Aligned on common grid: {wavenumbers.min():.1f}-{wavenumbers.max():.1f} cm⁻¹")
    
    return {
        'spectra': spectra,
        'wavenumbers': wavenumbers,
        'labels': labels,
        'label_names': label_names,
        'metadata': metadata,
        'file_path': file_path
    }


def preprocess_single_dataset(dataset_dict):
    """Preprocess one dataset."""
    print("\n" + "="*70)
    print("Preprocessing...")
    print("="*70)
    
    spectra = dataset_dict['spectra']
    wavenumbers = dataset_dict['wavenumbers']
    
    target_grid, processed = preprocess_aligned_spectra(
        spectra=spectra,
        wavenumbers=wavenumbers,
        **PREPROCESSING_CONFIG
    )
    
    print(f"\n✓ Preprocessed!")
    print(f"  Shape: {processed.shape}")
    print(f"  Grid: {len(target_grid)} points")
    
    dataset_dict['processed_spectra'] = processed
    dataset_dict['target_grid'] = target_grid
    return dataset_dict


def combine_all_datasets(dataset_list):
    """Combine multiple preprocessed datasets."""
    print("\n" + "="*70)
    print(f"Combining {len(dataset_list)} datasets...")
    print("="*70)
    
    # Check if grids are compatible
    first_grid = dataset_list[0]['target_grid']
    all_same = all(np.allclose(d['target_grid'], first_grid, atol=0.1) for d in dataset_list[1:])
    
    if not all_same:
        print("⚠ Different grids detected. Aligning...")
        # Find common range and resolution
        all_mins = [d['target_grid'].min() for d in dataset_list]
        all_maxs = [d['target_grid'].max() for d in dataset_list]
        all_res = [np.mean(np.diff(d['target_grid'])) for d in dataset_list]
        
        common_min = min(all_mins)
        common_max = max(all_maxs)
        common_res = min(all_res)
        common_grid = np.arange(common_min, common_max + common_res, common_res)
        
        # Interpolate each dataset
        from preprocessing import align_spectrum
        for d in dataset_list:
            aligned = []
            for spec in d['processed_spectra']:
                aligned.append(align_spectrum(d['target_grid'], spec, common_grid))
            d['processed_spectra'] = np.array(aligned)
            d['target_grid'] = common_grid
        
        target_grid = common_grid
    else:
        target_grid = first_grid
        print("✓ All datasets use same grid")
    
    # Combine
    all_spectra = np.vstack([d['processed_spectra'] for d in dataset_list])
    all_labels = np.concatenate([d['labels'] for d in dataset_list])
    
    # Create unified label mapping
    all_unique_names = set()
    for d in dataset_list:
        if d['label_names'] is not None:
            unique = np.unique(d['labels'])
            for label in unique:
                all_unique_names.add(str(d['label_names'][int(label)]))
    
    unique_names = sorted(all_unique_names)
    unified_mapping = {i: name for i, name in enumerate(unique_names)}
    name_to_idx = {name: i for i, name in enumerate(unique_names)}
    
    # Remap labels
    remapped = []
    for d in dataset_list:
        if d['label_names'] is not None:
            for label in d['labels']:
                old_name = str(d['label_names'][int(label)])
                remapped.append(name_to_idx[old_name])
        else:
            # Use original labels with offset (simple approach)
            remapped.extend(d['labels'].tolist())
    
    remapped_labels = np.array(remapped)
    
    print(f"\n✓ Combined!")
    print(f"  Total samples: {len(all_spectra)}")
    print(f"  Total classes: {len(unique_names)}")
    print(f"  Unified mapping: {unified_mapping}")
    
    return {
        'spectra': all_spectra,
        'labels': remapped_labels,
        'target_grid': target_grid,
        'label_mapping': unified_mapping
    }


def main():
    """Main workflow - run steps interactively."""
    
    print("="*70)
    print("MULTI-DATASET PREPROCESSING WORKFLOW")
    print("="*70)
    print("\nThis script will help you:")
    print("  1. Load each of your 4 datasets")
    print("  2. Preprocess each one")
    print("  3. Combine them")
    print("  4. Split for training")
    
    # Step 1: Load all datasets
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASETS")
    print("="*70)
    
    loaded_datasets = []
    for i, path in enumerate(DATASET_PATHS):
        print(f"\n>>> Dataset {i+1}/{len(DATASET_PATHS)}")
        try:
            dataset = load_single_dataset(path)
            loaded_datasets.append(dataset)
            print(f"\n✓ Dataset {i+1} loaded successfully!")
        except Exception as e:
            print(f"\n❌ Error loading dataset {i+1}: {e}")
            continue
    
    if not loaded_datasets:
        print("\n❌ No datasets loaded! Check your file paths.")
        return
    
    print(f"\n✓ Loaded {len(loaded_datasets)} datasets")
    
    # Step 2: Preprocess each dataset
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING DATASETS")
    print("="*70)
    
    preprocessed = []
    for i, dataset in enumerate(loaded_datasets):
        print(f"\n>>> Preprocessing Dataset {i+1}")
        try:
            processed = preprocess_single_dataset(dataset)
            preprocessed.append(processed)
            print(f"\n✓ Dataset {i+1} preprocessed!")
        except Exception as e:
            print(f"\n❌ Error preprocessing dataset {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not preprocessed:
        print("\n❌ No datasets preprocessed successfully!")
        return
    
    # Step 3: Combine
    print("\n" + "="*70)
    print("STEP 3: COMBINING DATASETS")
    print("="*70)
    
    try:
        combined = combine_all_datasets(preprocessed)
        print(f"\n✓ All datasets combined!")
    except Exception as e:
        print(f"\n❌ Error combining datasets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Split
    print("\n" + "="*70)
    print("STEP 4: SPLITTING FOR TRAINING")
    print("="*70)
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
            combined['spectra'],
            combined['labels'],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        print(f"\n✓ Split complete!")
        print(f"  Train: {len(X_train)}")
        print(f"  Val: {len(X_val)}")
        print(f"  Test: {len(X_test)}")
    except Exception as e:
        print(f"\n❌ Error splitting: {e}")
        return
    
    # Save
    print("\n" + "="*70)
    print("STEP 5: SAVING PREPROCESSED DATA")
    print("="*70)
    
    output_dir = Path("preprocessed_data")
    output_dir.mkdir(exist_ok=True)
    
    np.savez(
        output_dir / "combined_data.npz",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        target_grid=combined['target_grid']
    )
    
    with open(output_dir / "label_mapping.json", 'w') as f:
        json.dump(combined['label_mapping'], f, indent=2)
    
    print(f"\n✓ Saved to: {output_dir}/")
    print(f"  - combined_data.npz")
    print(f"  - label_mapping.json")
    
    print("\n" + "="*70)
    print("READY FOR TRAINING!")
    print("="*70)
    print(f"\nLoad with:")
    print(f"  data = np.load('preprocessed_data/combined_data.npz')")
    print(f"  X_train = data['X_train']")
    print(f"  y_train = data['y_train']")


if __name__ == "__main__":
    main()

