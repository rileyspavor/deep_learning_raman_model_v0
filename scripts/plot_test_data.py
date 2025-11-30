"""
Plot Test Real Data Spectra

This module provides a function to visualize all real Raman spectra from the test dataset
on a single plot. This helps visualize the test data distribution and characteristics.

Usage:
    # As a script
    python scripts/plot_test_data.py
    
    # As a script with custom data directory
    python scripts/plot_test_data.py --data-dir "data/test/testing_real_data"
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple
import argparse
import os

from src.data_ingestion import load_raman_spectrum


def plot_test_data_spectra(
    data_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    figsize: tuple = (16, 10),
    alpha: float = 0.7,
    linewidth: float = 1.0
) -> dict:
    """
    Plot all Raman spectra from test data directory on a single plot.
    
    Args:
        data_dir: Directory containing .txt spectrum files
        output_path: Path to save the plot (default: results/test_data.png)
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
        alpha: Transparency of lines (0-1)
        linewidth: Width of spectrum lines
    
    Returns:
        Dictionary with information about plotted spectra
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all .txt files
    txt_files = sorted([f for f in data_dir.glob("*.txt") if f.is_file()])
    
    if not txt_files:
        raise ValueError(f"No .txt files found in {data_dir}")
    
    print("=" * 70)
    print("Loading Test Data Spectra")
    print("=" * 70)
    print(f"Found {len(txt_files)} spectrum files")
    
    # Load all spectra
    spectra_data = []
    for txt_file in txt_files:
        try:
            wavenumbers, intensities = load_raman_spectrum(txt_file)
            spectra_data.append({
                'file': txt_file.name,
                'path': txt_file,
                'wavenumbers': wavenumbers,
                'intensities': intensities
            })
            print(f"  ✓ {txt_file.name}: {len(wavenumbers)} points, "
                  f"range {wavenumbers.min():.1f}-{wavenumbers.max():.1f} cm⁻¹")
        except Exception as e:
            print(f"  ✗ {txt_file.name}: Error loading - {e}")
    
    if not spectra_data:
        raise ValueError("No spectra could be loaded successfully")
    
    print(f"\nSuccessfully loaded {len(spectra_data)} spectra")
    
    # Create plot
    print("\n" + "=" * 70)
    print("Creating Visualization")
    print("=" * 70)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use color map for different spectra
    colors = plt.cm.tab20(np.linspace(0, 1, len(spectra_data)))
    
    # Plot all spectra
    for idx, spec_data in enumerate(spectra_data):
        ax.plot(
            spec_data['wavenumbers'],
            spec_data['intensities'],
            color=colors[idx],
            linewidth=linewidth,
            alpha=alpha,
            label=spec_data['file']
        )
    
    # Formatting
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    ax.set_title(f'Test Real Data Spectra ({len(spectra_data)} files)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Set x-axis limits based on all spectra
    all_wavenumbers = np.concatenate([spec['wavenumbers'] for spec in spectra_data])
    ax.set_xlim(all_wavenumbers.min(), all_wavenumbers.max())
    
    # Add legend (can be large, so place it outside or make it scrollable)
    # For many files, legend might be too large, so we'll make it compact
    if len(spectra_data) <= 15:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        # For many files, just show count
        ax.text(0.02, 0.98, f'{len(spectra_data)} spectra plotted', 
               transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=10)
    
    plt.tight_layout()
    
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
    
    # Return information about plotted spectra
    return {
        'n_files': len(spectra_data),
        'files': [spec['file'] for spec in spectra_data],
        'wavenumber_ranges': {
            spec['file']: (spec['wavenumbers'].min(), spec['wavenumbers'].max())
            for spec in spectra_data
        }
    }


def main():
    """Command-line interface for plotting test data spectra."""
    parser = argparse.ArgumentParser(
        description="Plot all real Raman spectra from test data directory on a single plot"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/test/testing_real_data",
        help="Directory containing .txt spectrum files (default: data/test/testing_real_data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/test_data.png",
        help="Output path for the plot (default: results/test_data.png)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Transparency of spectrum lines (0-1, default: 0.7)"
    )
    parser.add_argument(
        "--linewidth",
        type=float,
        default=1.0,
        help="Width of spectrum lines (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not Path(args.data_dir).exists():
        print(f"Error: Directory not found: {args.data_dir}")
        return
    
    # Ensure output path is in results/ directory if using default
    output_path = args.output
    if output_path == "test_data.png":
        output_path = "results/test_data.png"
    
    # Plot spectra
    result = plot_test_data_spectra(
        data_dir=args.data_dir,
        output_path=output_path,
        show_plot=not args.no_show,
        alpha=args.alpha,
        linewidth=args.linewidth
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total spectra plotted: {result['n_files']}")
    print(f"\nFiles:")
    for file in result['files']:
        wav_min, wav_max = result['wavenumber_ranges'][file]
        print(f"  {file:40s}: {wav_min:7.1f} - {wav_max:7.1f} cm⁻¹")


if __name__ == "__main__":
    main()

