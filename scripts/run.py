"""
Simple Run Script for Raman Spectroscopy Classification

Automatically processes all spectrum files in data/test/testing_real_data/ folder
and prints predictions.

Usage:
    python scripts/run.py
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
from scripts.inference_simple import (
    load_model_and_config,
    classify_raman_file,
    print_prediction,
    set_seeds
)

# Set seed for reproducibility
set_seeds(42)


def find_spectrum_files(data_dir="data/test/testing_real_data"):
    """
    Find all spectrum files in the testing directory.
    
    Args:
        data_dir: Directory to search for spectrum files
    
    Returns:
        List of file paths
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Warning: Directory '{data_dir}' not found.")
        return []
    
    # Find all .txt and .csv files
    files = []
    for ext in ['*.txt', '*.csv']:
        files.extend(data_path.glob(ext))
    
    return sorted(files)


def main():
    """Main function to run inference on all test files."""
    print("=" * 70)
    print("Raman Spectroscopy Classification - Batch Inference")
    print("=" * 70)
    
    # Load model (one time)
    print("\n[1/3] Loading model and configuration...")
    try:
        model, target_grid, idx_to_class, device = load_model_and_config()
        print(f"  ✓ Model loaded on device: {device}")
        print(f"  ✓ Target grid: {len(target_grid)} points")
        print(f"  ✓ Classes: {len(idx_to_class)}")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return
    
    # Find all test files
    print("\n[2/3] Scanning for spectrum files...")
    test_files = find_spectrum_files()
    
    if not test_files:
        print("  ✗ No spectrum files found in 'data/test/testing_real_data/' folder")
        print("  Add .txt or .csv files to the data/test/testing_real_data/ folder")
        return
    
    print(f"  ✓ Found {len(test_files)} file(s):")
    for f in test_files:
        print(f"    - {f.name}")
    
    # Process each file
    print("\n[3/3] Processing files and making predictions...")
    print("=" * 70)
    
    results = []
    for i, file_path in enumerate(test_files, 1):
        print(f"\n[{i}/{len(test_files)}] Processing: {file_path.name}")
        print("-" * 70)
        
        try:
            result = classify_raman_file(
                file_path,
                model,
                target_grid,
                idx_to_class,
                device
            )
            print_prediction(result)
            results.append(result)
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}:")
            print(f"    {type(e).__name__}: {e}")
            results.append({'file': str(file_path), 'error': str(e)})
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(test_files)}")
    print(f"Successful: {sum(1 for r in results if 'error' not in r)}")
    print(f"Failed: {sum(1 for r in results if 'error' in r)}")
    
    if results and 'predicted_class' in results[0]:
        print("\nPredictions:")
        for result in results:
            if 'error' not in result:
                print(f"  {Path(result['file']).name}: {result['predicted_class']} "
                      f"(confidence: {result['confidence']:.3f})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

