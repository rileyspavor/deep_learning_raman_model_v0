"""
Plot Training Spectra by Class

This module provides a function to visualize one random Raman spectrum
from each class in the training dataset. This helps understand what each
class looks like and verify data quality.

Usage:
    # As a script
    python scripts/plot_training_spectra.py --data-file "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz"
    
    # As a module
    from scripts.plot_training_spectra import plot_random_spectra_per_class
    
    plot_random_spectra_per_class(
        data_file="data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz",
        output_path="training_spectra_by_class.png",
        random_seed=42
    )
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List
import argparse

from src.data_ingestion import load_npz_dataset


def plot_random_spectra_per_class(
    data_file: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    random_seed: int = 42,
    show_plot: bool = True,
    figsize: tuple = (16, 12),
    n_samples_per_class: int = 5,
    spectra_key: str = "spectra",
    wavenumbers_key: str = "wavenumbers",
    labels_key: str = "y",
    label_names_key: str = "label_names"
):
    """
    Plot random Raman spectra for each class in the training dataset.
    
    Args:
        data_file: Path to .npz training data file
        output_path: Path to save the plot (optional)
        random_seed: Random seed for reproducibility
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
        n_samples_per_class: Number of random spectra to plot per class (default: 5)
        spectra_key: Key for spectra array in .npz file
        wavenumbers_key: Key for wavenumbers array in .npz file
        labels_key: Key for labels array in .npz file
        label_names_key: Key for label names array in .npz file
    
    Returns:
        Dictionary with selected spectrum indices and class information
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load dataset
    print("=" * 70)
    print("Loading Training Dataset")
    print("=" * 70)
    
    spectra, wavenumbers, labels, label_names, metadata = load_npz_dataset(
        file_path=data_file,
        spectra_key=spectra_key,
        wavenumbers_key=wavenumbers_key,
        labels_key=labels_key,
        label_names_key=label_names_key
    )
    
    print(f"  Dataset shape: {spectra.shape}")
    print(f"  Number of classes: {len(np.unique(labels))}")
    print(f"  Wavenumber range: {wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
    
    # Handle wavenumbers if 2D
    if wavenumbers.ndim == 2:
        if wavenumbers.shape[0] == spectra.shape[0]:
            # Check if all rows are the same
            if np.allclose(wavenumbers[0], wavenumbers[-1]):
                wavenumbers = wavenumbers[0]
            else:
                print("  Warning: Different wavenumber grids per spectrum detected")
                # Use first spectrum's wavenumbers for plotting
                wavenumbers = wavenumbers[0]
        else:
            wavenumbers = wavenumbers[0]
    
    # Get unique classes
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Create label mapping
    if label_names is not None:
        # Create mapping from label index to name
        label_mapping = {}
        for label in unique_labels:
            label_idx = int(label)
            if label_idx < len(label_names):
                label_mapping[label] = str(label_names[label_idx])
            else:
                label_mapping[label] = f"Class_{label}"
    else:
        label_mapping = {label: f"Class_{label}" for label in unique_labels}
    
    print(f"\n  Classes found: {list(label_mapping.values())}")
    
    # Select random spectra per class
    selected_indices = {}
    selected_spectra = {}
    
    print("\n" + "=" * 70)
    print(f"Selecting {n_samples_per_class} Random Spectra Per Class")
    print("=" * 70)
    
    for label in unique_labels:
        # Find all indices for this class
        class_indices = np.where(labels == label)[0]
        n_samples = len(class_indices)
        
        # Randomly select n_samples_per_class (or all if less available)
        n_to_select = min(n_samples_per_class, n_samples)
        random_indices = np.random.choice(class_indices, size=n_to_select, replace=False)
        selected_indices[label] = random_indices.tolist()
        selected_spectra[label] = spectra[random_indices]
        
        print(f"  {label_mapping[label]:30s}: Selected {n_to_select:2d} spectra (from {n_samples:5d} samples)")
    
    # Create plot
    print("\n" + "=" * 70)
    print("Creating Visualization")
    print("=" * 70)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_classes + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array for easier indexing
    if n_classes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each class
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for idx, (label, class_spectra) in enumerate(selected_spectra.items()):
        ax = axes[idx]
        
        # Plot all selected spectra for this class (overlayed)
        # Use a color gradient for multiple spectra
        n_spectra = len(class_spectra)
        if n_spectra == 1:
            spectrum_colors = [colors[idx]]
        else:
            # Create a color gradient from the class color
            base_color = colors[idx]
            spectrum_colors = plt.cm.colors.to_rgba_array([base_color] * n_spectra)
            # Vary alpha slightly for better visibility
            for i in range(n_spectra):
                spectrum_colors[i, 3] = 0.6 + 0.4 * (i / max(1, n_spectra - 1))
        
        for spec_idx, spectrum in enumerate(class_spectra):
            ax.plot(wavenumbers, spectrum, 
                   color=spectrum_colors[spec_idx] if n_spectra > 1 else colors[idx],
                   linewidth=1.2, alpha=0.7 if n_spectra > 1 else 0.8)
        
        # Formatting
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
        ax.set_ylabel('Intensity', fontsize=10)
        ax.set_title(f'{label_mapping[label]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(wavenumbers.min(), wavenumbers.max())
        
        # Add sample count
        n_samples = np.sum(labels == label)
        ax.text(0.02, 0.98, f'n={n_samples} (showing {n_spectra})', 
               transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_classes, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    fig.suptitle(
        f'Random Raman Spectra by Class\nDataset: {Path(data_file).name}',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save if requested
    if output_path:
        output_path = Path(output_path)
        # If relative path and not already in results/, save to results/
        if not output_path.is_absolute() and "results" not in str(output_path.parent):
            output_path = Path("results") / output_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {output_path.resolve()}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Return information about selected spectra
    return {
        'selected_indices': selected_indices,
        'label_mapping': label_mapping,
        'n_classes': n_classes,
        'n_samples_per_class': {label: np.sum(labels == label) for label in unique_labels}
    }


def find_available_datasets(base_dir: Path = None) -> List[Path]:
    """
    Find all .npz files in common data directories.
    
    Args:
        base_dir: Base directory to search from (default: project root)
    
    Returns:
        List of paths to .npz files
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    datasets = []
    
    # Common data directories to search
    search_dirs = [
        base_dir / "data" / "processed" / "v3_data",
        base_dir / "data" / "processed" / "full_training_raman_labels",
        base_dir / "data" / "processed" / "training_data",
        base_dir / "data" / "processed",
    ]
    
    # Find all .npz files
    for search_dir in search_dirs:
        if search_dir.exists():
            for npz_file in search_dir.glob("*.npz"):
                datasets.append(npz_file)
    
    # Remove duplicates and sort
    datasets = sorted(set(datasets))
    return datasets


def prompt_dataset_selection() -> Optional[str]:
    """
    Prompt user to select a dataset from available .npz files.
    
    Returns:
        Selected dataset path, or None if cancelled
    """
    datasets = find_available_datasets()
    
    if not datasets:
        print("No .npz datasets found in data directories.")
        print("Please specify a dataset with --data-file")
        return None
    
    print("\n" + "=" * 70)
    print("Available Datasets")
    print("=" * 70)
    
    for i, dataset in enumerate(datasets, 1):
        # Show relative path from project root
        try:
            rel_path = dataset.relative_to(Path(__file__).parent.parent)
            print(f"  {i}. {rel_path}")
        except ValueError:
            print(f"  {i}. {dataset}")
    
    print("=" * 70)
    
    while True:
        try:
            user_input = input(f"\nSelect a dataset (1-{len(datasets)}) or 'q' to quit: ").strip()
            
            if user_input.lower() == 'q':
                return None
            
            choice = int(user_input)
            if 1 <= choice <= len(datasets):
                selected = datasets[choice - 1]
                print(f"\nSelected: {selected}")
                return str(selected)
            else:
                print(f"Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")


def main():
    """Command-line interface for plotting training spectra."""
    parser = argparse.ArgumentParser(
        description="Plot random Raman spectra for each class in the training dataset (default: 5 per class)"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to .npz training data file (if not provided, will prompt to select from available datasets)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/training_spectra_by_class.png",
        help="Output path for the plot (default: auto-generated as results/training_data_visual_{dataset_name}.png)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of random spectra to plot per class (if not provided, will prompt interactively)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save)"
    )
    parser.add_argument(
        "--spectra-key",
        type=str,
        default="spectra",
        help="Key for spectra array in .npz file (default: 'spectra')"
    )
    parser.add_argument(
        "--wavenumbers-key",
        type=str,
        default="wavenumbers",
        help="Key for wavenumbers array in .npz file (default: 'wavenumbers')"
    )
    parser.add_argument(
        "--labels-key",
        type=str,
        default="y",
        help="Key for labels array in .npz file (default: 'y')"
    )
    parser.add_argument(
        "--label-names-key",
        type=str,
        default="label_names",
        help="Key for label names array in .npz file (default: 'label_names')"
    )
    
    args = parser.parse_args()
    
    # If no data file provided, prompt user to select from available datasets
    data_file = args.data_file
    if data_file is None:
        data_file = prompt_dataset_selection()
        if data_file is None:
            print("No dataset selected. Exiting.")
            return
    
    # Check if file exists
    if not Path(data_file).exists():
        print(f"Error: File not found: {data_file}")
        return
    
    # Generate output filename based on dataset name if using default
    output_path = args.output
    if output_path == "results/training_spectra_by_class.png" or output_path == "training_spectra_by_class.png":
        # Extract dataset name (filename without extension)
        dataset_name = Path(data_file).stem
        output_path = f"results/training_data_visual_{dataset_name}.png"
    
    # Prompt for number of samples per class if not provided
    n_samples = args.n_samples
    if n_samples is None:
        while True:
            try:
                user_input = input("\nHow many random spectra per class would you like to plot? (default: 5): ").strip()
                if user_input == "":
                    n_samples = 5
                    break
                n_samples = int(user_input)
                if n_samples < 1:
                    print("  Please enter a positive integer.")
                    continue
                break
            except ValueError:
                print("  Invalid input. Please enter a positive integer.")
        print(f"  Using {n_samples} samples per class.\n")
    
    # Plot spectra
    result = plot_random_spectra_per_class(
        data_file=data_file,
        output_path=output_path,
        random_seed=args.seed,
        show_plot=not args.no_show,
        n_samples_per_class=n_samples,
        spectra_key=args.spectra_key,
        wavenumbers_key=args.wavenumbers_key,
        labels_key=args.labels_key,
        label_names_key=args.label_names_key
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total classes: {result['n_classes']}")
    print(f"\nSamples per class:")
    for label, count in result['n_samples_per_class'].items():
        class_name = result['label_mapping'][label]
        print(f"  {class_name:30s}: {count:5d} samples")


if __name__ == "__main__":
    main()

