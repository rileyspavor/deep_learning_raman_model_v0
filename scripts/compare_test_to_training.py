"""
Compare Test Spectrum to Training Data

This script allows you to:
1. Pick a test spectrum
2. Run it through the model to get prediction
3. Visualize the test spectrum against training spectra from the predicted class
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import torch
import warnings
import os
from typing import Optional, Union, Tuple
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from src.data_ingestion import load_raman_spectrum, load_npz_dataset
from scripts.test_real_data_v3 import load_v3_model, predict_spectrum, align_spectrum_to_grid


def list_test_files(data_dir: Path) -> list:
    """List all .txt files in the test data directory."""
    if not data_dir.exists():
        return []
    
    txt_files = sorted([f for f in data_dir.glob("*.txt") if f.is_file()])
    return txt_files


def prompt_test_file_selection(data_dir: Path) -> Optional[Path]:
    """Prompt user to select a test file."""
    test_files = list_test_files(data_dir)
    
    if not test_files:
        print(f"No .txt files found in {data_dir}")
        return None
    
    print("\n" + "=" * 70)
    print("Available Test Files")
    print("=" * 70)
    
    for i, test_file in enumerate(test_files, 1):
        print(f"  {i}. {test_file.name}")
    
    print("=" * 70)
    
    while True:
        try:
            user_input = input(f"\nSelect a test file (1-{len(test_files)}) or 'q' to quit: ").strip()
            
            if user_input.lower() == 'q':
                return None
            
            choice = int(user_input)
            if 1 <= choice <= len(test_files):
                selected = test_files[choice - 1]
                print(f"\nSelected: {selected.name}")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(test_files)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


def load_training_spectra_for_class(
    data_file: Union[str, Path],
    predicted_class: str,
    n_samples: int = 10,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training spectra for a specific class.
    
    Args:
        data_file: Path to .npz training data file
        predicted_class: Class name to filter by
        n_samples: Number of random samples to load
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (spectra, wavenumbers, selected_indices)
    """
    np.random.seed(random_seed)
    
    # Load dataset
    spectra, wavenumbers, labels, label_names, metadata = load_npz_dataset(
        file_path=data_file,
        spectra_key="spectra",
        wavenumbers_key="wavenumbers",
        labels_key="y",
        label_names_key="label_names"
    )
    
    # Handle wavenumbers if 2D
    if wavenumbers.ndim == 2:
        if np.allclose(wavenumbers[0], wavenumbers[-1]):
            wavenumbers = wavenumbers[0]
        else:
            wavenumbers = wavenumbers[0]
    
    # Find class index
    if label_names is None:
        raise ValueError("Label names not found in dataset")
    
    try:
        class_idx = list(label_names).index(predicted_class)
    except ValueError:
        # Try case-insensitive
        label_names_lower = [str(n).lower() for n in label_names]
        try:
            class_idx = label_names_lower.index(predicted_class.lower())
        except ValueError:
            raise ValueError(f"Class '{predicted_class}' not found in label names: {list(label_names)}")
    
    # Filter spectra by class
    class_mask = labels == class_idx
    class_spectra = spectra[class_mask]
    class_indices = np.where(class_mask)[0]
    
    if len(class_spectra) == 0:
        raise ValueError(f"No training spectra found for class '{predicted_class}'")
    
    # Randomly select n_samples
    n_to_select = min(n_samples, len(class_spectra))
    selected_indices = np.random.choice(len(class_spectra), size=n_to_select, replace=False)
    selected_spectra = class_spectra[selected_indices]
    selected_original_indices = class_indices[selected_indices]
    
    return selected_spectra, wavenumbers, selected_original_indices


def plot_comparison(
    test_wavenumbers: np.ndarray,
    test_intensities: np.ndarray,
    test_file: str,
    predicted_class: str,
    confidence: float,
    training_spectra: np.ndarray,
    training_wavenumbers: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    figsize: tuple = (16, 10)
):
    """
    Plot test spectrum against training spectra from predicted class.
    
    Args:
        test_wavenumbers: Test spectrum wavenumbers
        test_intensities: Test spectrum intensities
        test_file: Test file name
        predicted_class: Predicted class name
        confidence: Prediction confidence
        training_spectra: Training spectra array (n_samples, n_points)
        training_wavenumbers: Training wavenumbers array
        output_path: Output file path
        show_plot: Whether to display plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training spectra (light, semi-transparent)
    colors = plt.cm.Blues(np.linspace(0.3, 0.7, len(training_spectra)))
    for i, train_spec in enumerate(training_spectra):
        ax.plot(
            training_wavenumbers,
            train_spec,
            color=colors[i],
            linewidth=1.0,
            alpha=0.4,
            label='Training spectra' if i == 0 else None
        )
    
    # Plot test spectrum (bold, prominent)
    ax.plot(
        test_wavenumbers,
        test_intensities,
        color='red',
        linewidth=2.5,
        alpha=0.9,
        label=f'Test: {test_file}'
    )
    
    # Formatting
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Test Spectrum vs Training Data\n'
        f'Predicted: {predicted_class} ({confidence:.1%} confidence)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits
    all_wavenumbers = np.concatenate([test_wavenumbers, training_wavenumbers])
    ax.set_xlim(all_wavenumbers.min(), all_wavenumbers.max())
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add info text
    info_text = f'Test file: {test_file}\n'
    info_text += f'Predicted class: {predicted_class}\n'
    info_text += f'Confidence: {confidence:.1%}\n'
    info_text += f'Training samples shown: {len(training_spectra)}'
    
    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        fontsize=9,
        family='monospace'
    )
    
    plt.tight_layout()
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        if not output_path.is_absolute() and "results" not in str(output_path.parent):
            output_path = Path("results") / output_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {output_path.resolve()}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare a test spectrum to training data from predicted class"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/test/testing_real_data",
        help="Directory with test .txt spectrum files (default: data/test/testing_real_data)"
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz",
        help="Path to training .npz file (default: data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/saved_models_v3",
        help="Model directory (default: models/saved_models_v3)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of training spectra to show (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/test_vs_training_comparison.png",
        help="Output path for plot (default: results/test_vs_training_comparison.png)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific test file to use (skips selection prompt)"
    )
    
    args = parser.parse_args()
    
    # Check directories
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Test data directory not found: {data_dir}")
        return
    
    training_data = Path(args.training_data)
    if not training_data.exists():
        print(f"Error: Training data file not found: {training_data}")
        return
    
    # Select test file
    if args.file:
        test_file = data_dir / args.file
        if not test_file.exists():
            print(f"Error: Test file not found: {test_file}")
            return
    else:
        test_file = prompt_test_file_selection(data_dir)
        if test_file is None:
            print("No test file selected. Exiting.")
            return
    
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)
    
    # Load model
    model, target_grid, class_names, model_version, model_dir_path = load_v3_model(
        args.model_dir,
        interactive=False
    )
    
    print(f"Model loaded: {len(class_names)} classes")
    print(f"Model version: v{model_version}")
    
    # Load test spectrum
    print("\n" + "=" * 70)
    print("Loading Test Spectrum")
    print("=" * 70)
    print(f"File: {test_file.name}")
    
    test_wavenumbers, test_intensities = load_raman_spectrum(test_file)
    print(f"  Loaded: {len(test_wavenumbers)} points")
    print(f"  Wavenumber range: {test_wavenumbers.min():.1f} - {test_wavenumbers.max():.1f} cm⁻¹")
    
    # Predict
    print("\n" + "=" * 70)
    print("Running Prediction")
    print("=" * 70)
    
    predicted_class, confidence, prob_dict, aligned_intensities = predict_spectrum(
        model, target_grid, class_names, test_wavenumbers, test_intensities
    )
    
    print(f"  Predicted Class: {predicted_class}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"\n  All Probabilities:")
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_probs[:5]:  # Show top 5
        marker = "✓" if class_name == predicted_class else " "
        print(f"    {marker} {class_name:30s}: {prob:.2%}")
    
    # Load training spectra for predicted class
    print("\n" + "=" * 70)
    print(f"Loading Training Spectra for '{predicted_class}'")
    print("=" * 70)
    
    try:
        training_spectra, training_wavenumbers, _ = load_training_spectra_for_class(
            training_data,
            predicted_class,
            n_samples=args.n_samples,
            random_seed=42
        )
        print(f"  Loaded {len(training_spectra)} training spectra")
    except Exception as e:
        print(f"  Error loading training spectra: {e}")
        return
    
    # Create comparison plot
    print("\n" + "=" * 70)
    print("Creating Comparison Plot")
    print("=" * 70)
    
    # Generate output filename based on test file
    test_file_stem = test_file.stem
    output_path = args.output
    if output_path == "results/test_vs_training_comparison.png":
        output_path = f"results/test_vs_training_{test_file_stem}.png"
    
    plot_comparison(
        test_wavenumbers=test_wavenumbers,
        test_intensities=test_intensities,
        test_file=test_file.name,
        predicted_class=predicted_class,
        confidence=confidence,
        training_spectra=training_spectra,
        training_wavenumbers=training_wavenumbers,
        output_path=output_path,
        show_plot=not args.no_show
    )
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test file: {test_file.name}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Training spectra shown: {len(training_spectra)}")
    print(f"Plot saved to: {Path(output_path).resolve()}")


if __name__ == "__main__":
    main()

