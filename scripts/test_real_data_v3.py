"""
Test v3 model on real Raman spectra data files.
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import json
import warnings
import os
from typing import List, Tuple, Optional
import re

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from src.data_ingestion import load_raman_spectrum
from src.model import create_model
from scipy.interpolate import interp1d


def list_available_models(base_dir: Path) -> List[Tuple[int, Path]]:
    """
    List all available model versions.
    
    Args:
        base_dir: Base directory containing versioned model folders
    
    Returns:
        List of (version_number, path) tuples, sorted by version
    """
    if not base_dir.exists():
        return []
    
    # Check if base_dir contains model files directly (old structure)
    if (base_dir / "model_state_v3.pth").exists() or (base_dir / "model_checkpoint_v3.pth").exists():
        return [(3, base_dir)]  # Assume version 3 for old structure
    
    # Find versioned subdirectories
    versioned_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith('model_v'):
            try:
                version = int(item.name.replace('model_v', ''))
                versioned_dirs.append((version, item))
            except ValueError:
                continue
    
    return sorted(versioned_dirs, key=lambda x: x[0])


def is_model_complete(model_dir: Path) -> bool:
    """
    Check if a model directory has all required files.
    
    Args:
        model_dir: Path to model version directory
    
    Returns:
        True if model is complete, False otherwise
    """
    # Check for class names file
    has_class_names = any(model_dir.glob("class_names_v*.json"))
    
    # Check for target grid file
    has_target_grid = any(model_dir.glob("target_grid_v*.npy"))
    
    # Check for model state file (either checkpoint or state)
    has_model_state = any(model_dir.glob("model_state_v*.pth")) or any(model_dir.glob("model_checkpoint_v*.pth"))
    
    return has_class_names and has_target_grid and has_model_state


def find_latest_model_version(base_dir: Path) -> Path:
    """
    Find the latest complete model version directory.
    
    Args:
        base_dir: Base directory containing versioned model folders
    
    Returns:
        Path to the latest complete model version directory
    """
    available_models = list_available_models(base_dir)
    
    if not available_models:
        raise FileNotFoundError(f"No model versions found in {base_dir}")
    
    # Try models from latest to oldest, return first complete one
    for version, model_dir in reversed(available_models):
        if is_model_complete(model_dir):
            return model_dir
    
    # If no complete model found, raise error
    raise FileNotFoundError(
        f"No complete model versions found in {base_dir}. "
        f"A complete model needs: class_names_v*.json, target_grid_v*.npy, and model_state_v*.pth"
    )


def prompt_model_selection(base_dir: Path) -> int:
    """
    Prompt user to select a model version.
    
    Args:
        base_dir: Base directory containing versioned model folders
    
    Returns:
        Selected version number
    """
    available_models = list_available_models(base_dir)
    
    if not available_models:
        raise FileNotFoundError(f"No model versions found in {base_dir}")
    
    print("\n" + "=" * 70)
    print("Available Model Versions")
    print("=" * 70)
    
    for version, path in available_models:
        # Check if model files exist
        has_checkpoint = any(path.glob("model_checkpoint_v*.pth"))
        has_state = any(path.glob("model_state_v*.pth"))
        status = "✓" if (has_checkpoint or has_state) else "✗"
        print(f"  {status} model_v{version}")
    
    print("=" * 70)
    
    # Get user selection
    while True:
        try:
            if len(available_models) == 1:
                selected_version = available_models[0][0]
                print(f"\nOnly one model available. Using model_v{selected_version}")
                return selected_version
            
            latest_version = available_models[-1][0]
            user_input = input(f"\nSelect model version (1-{latest_version}, or 'latest' for {latest_version}): ").strip().lower()
            
            if user_input == 'latest' or user_input == '':
                return latest_version
            
            selected_version = int(user_input)
            
            # Check if version exists
            if any(v == selected_version for v, _ in available_models):
                return selected_version
            else:
                print(f"  ✗ Model version {selected_version} not found. Please try again.")
        except ValueError:
            print("  ✗ Invalid input. Please enter a number or 'latest'.")
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            raise


def load_v3_model(model_dir="models/saved_models_v3", verbose=False, version=None, interactive=True):
    """
    Load the v3 model and associated files.
    
    Args:
        model_dir: Base directory containing models (or specific version directory)
        verbose: Whether to print loading information
        version: Specific version to load (e.g., 3). If None and interactive=True, prompts user.
        interactive: If True and version is None, prompt user to select a model
    """
    model_dir = Path(model_dir)
    
    # If a specific version is requested, use that directory
    if version is not None:
        model_dir = model_dir / f"model_v{version}"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found in {model_dir.parent}")
    else:
        # If interactive, prompt user to select
        if interactive:
            version = prompt_model_selection(model_dir)
            model_dir = model_dir / f"model_v{version}"
        else:
            # Find the latest version
            model_dir = find_latest_model_version(model_dir)
    
    if verbose:
        print(f"Loading model from: {model_dir}")
    
    # Find class names file (try versioned first, then fallback to v3)
    class_names_file = None
    for pattern in ["class_names_v*.json", "class_names_v3.json"]:
        matches = list(model_dir.glob(pattern))
        if matches:
            class_names_file = matches[0]
            break
    
    if class_names_file is None:
        raise FileNotFoundError(f"Class names file not found in {model_dir}")
    
    with open(class_names_file, 'r') as f:
        class_names_dict = json.load(f)
    
    # Convert to list ordered by index
    n_classes = len(class_names_dict)
    class_names = [class_names_dict.get(str(i)) or class_names_dict.get(i) 
                   for i in range(n_classes)]
    
    # Find target grid file
    target_grid_file = None
    for pattern in ["target_grid_v*.npy", "target_grid_v3.npy"]:
        matches = list(model_dir.glob(pattern))
        if matches:
            target_grid_file = matches[0]
            break
    
    if target_grid_file is None:
        raise FileNotFoundError(f"Target grid file not found in {model_dir}")
    
    target_grid = np.load(target_grid_file)
    input_length = len(target_grid)
    
    # Find model state file
    model_state_file = None
    for pattern in ["model_state_v*.pth", "model_state_v3.pth"]:
        matches = list(model_dir.glob(pattern))
        if matches:
            model_state_file = matches[0]
            break
    
    if model_state_file is None:
        # Try checkpoint file
        checkpoint_file = None
        for pattern in ["model_checkpoint_v*.pth", "model_checkpoint_v3.pth"]:
            matches = list(model_dir.glob(pattern))
            if matches:
                checkpoint_file = matches[0]
                break
        
        if checkpoint_file is None:
            raise FileNotFoundError(f"Model file not found in {model_dir}")
        
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = torch.load(model_state_file, map_location="cpu")
    
    # Create model
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
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Extract version number from model_dir path
    model_version = None
    if 'model_v' in str(model_dir):
        try:
            model_version = int(str(model_dir).split('model_v')[-1].split('/')[0])
        except ValueError:
            pass
    
    if verbose:
        print(f"Model loaded: {len(class_names)} classes")
        print(f"Target grid: {target_grid.min():.1f} - {target_grid.max():.1f} cm⁻¹ ({len(target_grid)} points)")
        print(f"Model directory: {model_dir}")
        if model_version:
            print(f"Model version: v{model_version}")
    
    return model, target_grid, class_names, model_version, model_dir


def align_spectrum_to_grid(wavenumbers, intensities, target_grid):
    """Align spectrum to target wavenumber grid using interpolation."""
    # Remove any NaN or Inf values
    mask = np.isfinite(wavenumbers) & np.isfinite(intensities)
    wavenumbers = wavenumbers[mask]
    intensities = intensities[mask]
    
    # Sort by wavenumber
    sort_idx = np.argsort(wavenumbers)
    wavenumbers = wavenumbers[sort_idx]
    intensities = intensities[sort_idx]
    
    # Find overlapping range
    min_wavenumber = max(wavenumbers.min(), target_grid.min())
    max_wavenumber = min(wavenumbers.max(), target_grid.max())
    
    # Interpolate to target grid
    f = interp1d(wavenumbers, intensities, kind='linear', 
                 bounds_error=False, fill_value=0.0)
    aligned_intensities = f(target_grid)
    
    return aligned_intensities


def predict_spectrum(model, target_grid, class_names, wavenumbers, intensities):
    """Predict class for a spectrum."""
    # Align to target grid
    aligned_intensities = align_spectrum_to_grid(wavenumbers, intensities, target_grid)
    
    # Convert to tensor: (batch=1, channels=1, length=N)
    x = torch.tensor(aligned_intensities, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        class_logits, _ = model(x)
        probs = torch.softmax(class_logits, dim=1).cpu().numpy()[0]
    
    pred_idx = int(probs.argmax())
    predicted_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    # Get all probabilities
    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    
    return predicted_class, confidence, prob_dict, aligned_intensities


def test_real_data(data_file, model_dir="models/saved_models_v3", model=None, target_grid=None, class_names=None):
    """Test model on a single real data file."""
    # Load model if not provided (for efficiency when testing multiple files)
    if model is None or target_grid is None or class_names is None:
        model, target_grid, class_names, _, _ = load_v3_model(model_dir, interactive=False)
    
    # Load real spectrum
    try:
        # Load using pandas with proper header handling
        import pandas as pd
        
        # Read file and detect if it has a header
        with open(data_file, 'r') as f:
            first_line = f.readline().strip()
        
        # Check if first line looks like a header (contains non-numeric text)
        has_header = not first_line.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').strip().replace(' ', '').isdigit()
        
        if has_header:
            # Has header - read with pandas
            df = pd.read_csv(data_file, sep=r'\s+', skipinitialspace=True)
            # Try to find wavenumber and intensity columns
            if 'Raman_shift_cm-1' in df.columns or 'Raman_shift_cm-1' in df.columns.values[0]:
                col0 = df.columns[0]
                col1 = df.columns[1] if len(df.columns) > 1 else None
                wavenumbers = df.iloc[:, 0].values
                intensities = df.iloc[:, 1].values if col1 else df.iloc[:, 0].values
            else:
                wavenumbers = df.iloc[:, 0].values
                intensities = df.iloc[:, 1].values
        else:
            # No header - just read the data
            data = np.loadtxt(data_file)
            wavenumbers = data[:, 0]
            intensities = data[:, 1]
    except Exception as e:
        print(f"ERROR loading {data_file}: {e}")
        return None
    
    # Predict
    predicted_class, confidence, prob_dict, aligned_intensities = predict_spectrum(
        model, target_grid, class_names, wavenumbers, intensities
    )
    
    return {
        'file': str(Path(data_file).name),
        'file_path': str(Path(data_file).resolve()),
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': prob_dict
    }


def get_next_prediction_file_number(output_dir: Path = None) -> int:
    """
    Find the next available prediction results file number.
    
    Args:
        output_dir: Directory to search for existing files (default: results/)
    
    Returns:
        Next available file number (e.g., 1 if no files exist, 2 if prediction_results_1.txt exists)
    """
    if output_dir is None:
        output_dir = Path("results")
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find existing prediction_results_n.txt files
    existing_numbers = []
    for file in output_dir.glob("prediction_results_*.txt"):
        try:
            # Extract number from filename like "prediction_results_1.txt"
            number = int(file.stem.split('_')[-1])
            existing_numbers.append(number)
        except (ValueError, IndexError):
            continue
    
    if not existing_numbers:
        return 1
    
    return max(existing_numbers) + 1


def save_results_to_file(results, output_file="prediction_results.txt", model_version=None, model_dir=None):
    """
    Save prediction results to a text file.
    
    Args:
        results: List of result dictionaries
        output_file: Output file path (relative paths will be saved to results/ directory)
        model_version: Model version number (e.g., 3)
        model_dir: Model directory path
    """
    output_path = Path(output_file)
    
    # If it's a relative path and not already in results/, save to results/
    if not output_path.is_absolute() and "results" not in str(output_path.parent):
        output_path = Path("results") / output_path.name
    
    # Create results directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Raman Spectroscopy Classification - Prediction Results\n")
        f.write("=" * 80 + "\n\n")
        
        for i, r in enumerate(results, 1):
            f.write(f"{i}. {r['file']}\n")
            f.write(f"   Predicted Class: {r['predicted_class']}\n")
            f.write(f"   Confidence: {r['confidence']:.2%}\n")
            
            # Write all probabilities
            f.write(f"   All Class Probabilities:\n")
            sorted_probs = sorted(r['probabilities'].items(), key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs:
                marker = "✓" if class_name == r['predicted_class'] else " "
                f.write(f"     {marker} {class_name:30s}: {prob:.2%}\n")
            
            f.write("\n")
        
        # Summary
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total files processed: {len(results)}\n")
        
        # Model information
        if model_version is not None:
            f.write(f"Model version: v{model_version}\n")
        if model_dir is not None:
            f.write(f"Model directory: {model_dir}\n")
        
        f.write("\n")
        
        # Summary of each prediction
        f.write("Prediction Summary:\n")
        f.write("-" * 80 + "\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i}. {r['file']}\n")
            f.write(f"   → {r['predicted_class']} ({r['confidence']:.1%} confidence)\n")
            f.write("\n")


def save_results_to_csv(results, output_file="prediction_results.csv"):
    """
    Save prediction results to a CSV file.
    
    Args:
        results: List of result dictionaries
        output_file: Output CSV file path (relative paths will be saved to results/ directory)
    """
    import csv
    
    output_path = Path(output_file)
    
    # If it's a relative path and not already in results/, save to results/
    if not output_path.is_absolute() and "results" not in str(output_path.parent):
        output_path = Path("results") / output_path.name
    
    # Create results directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all class names (from first result)
    if not results:
        return
    
    class_names = sorted(list(results[0]['probabilities'].keys()))
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['File', 'File_Path', 'Predicted_Class', 'Confidence'] + class_names
        writer.writerow(header)
        
        # Data rows
        for r in results:
            row = [
                r['file'],
                r.get('file_path', r['file']),
                r['predicted_class'],
                f"{r['confidence']:.4f}"
            ]
            # Add probabilities for each class
            row.extend([f"{r['probabilities'].get(cn, 0.0):.4f}" for cn in class_names])
            writer.writerow(row)


def main():
    """Test model on all real data files."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test v3 model on real Raman spectra")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/test/testing_real_data",
        help="Directory containing real data files"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific file to test (if not provided, tests all files)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/saved_models_v3",
        help="Directory containing saved model"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Specific model version to use (e.g., 3). If not specified, will prompt for selection."
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Don't prompt for model selection, use latest version automatically"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prediction_results.txt",
        help="Output file for prediction results (default: prediction_results.txt)"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Save results as CSV file instead of text file"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if args.file:
        # Test single file
        files_to_test = [data_dir / args.file]
    else:
        # Test all .txt files in directory
        # Use os.listdir() to preserve filesystem order (as they appear in directory)
        try:
            # Get files in directory order (no sorting to preserve filesystem order)
            all_files = [data_dir / f for f in os.listdir(data_dir) 
                        if f.endswith('.txt') and (data_dir / f).is_file()]
            files_to_test = all_files
        except Exception:
            # Fallback to glob if listdir fails, but don't sort to preserve order
            files_to_test = list(data_dir.glob("*.txt"))
    
    if not files_to_test:
        print(f"No .txt files found in {data_dir}")
        return
    
    # Load model once (more efficient for multiple files)
    model, target_grid, class_names, model_version, model_dir_path = load_v3_model(
        args.model_dir, 
        verbose=True,
        version=args.version,
        interactive=not args.no_interactive
    )
    
    # Process all files silently
    results = []
    for data_file in files_to_test:
        if not data_file.exists():
            continue
        
        result = test_real_data(
            data_file, 
            model_dir=args.model_dir,
            model=model,
            target_grid=target_grid,
            class_names=class_names
        )
        
        if result:
            results.append(result)
    
    # Save results to file
    if results:
        # Ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if args.csv:
            output_file = args.output if args.output.endswith('.csv') else args.output.replace('.txt', '.csv')
            # If default name, use numbered format
            if output_file == "prediction_results.csv":
                file_number = get_next_prediction_file_number(results_dir)
                output_file = f"prediction_results_{file_number}.csv"
            
            # Ensure output_file path includes results/ directory
            output_path = Path(output_file)
            if not output_path.is_absolute() and "results" not in str(output_path.parent):
                output_file = str(results_dir / output_path.name)
            
            save_results_to_csv(results, output_file)
        else:
            # Use numbered filename format: prediction_results_n.txt
            if args.output == "prediction_results.txt" or args.output.endswith("prediction_results.txt"):
                # Auto-increment file number
                file_number = get_next_prediction_file_number(results_dir)
                output_file = f"prediction_results_{file_number}.txt"
            else:
                output_file = args.output if args.output.endswith('.txt') else args.output + '.txt'
            
            # Ensure output_file path includes results/ directory
            output_path = Path(output_file)
            if not output_path.is_absolute() and "results" not in str(output_path.parent):
                output_file = str(results_dir / output_path.name)
            
            save_results_to_file(
                results, 
                output_file,
                model_version=model_version,
                model_dir=str(model_dir_path) if model_dir_path else None
            )
        
        print(f"Processed {len(results)} file(s)")
        print(f"Results saved to: {Path(output_file).resolve()}")
    else:
        print("No files processed successfully")


if __name__ == "__main__":
    main()

