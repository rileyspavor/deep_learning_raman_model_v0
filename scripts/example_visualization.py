"""
Example script demonstrating how to use the CNN visualization component.

This script shows how to visualize how the CNN processes and classifies
Raman spectroscopy spectra.
"""

from visualize_cnn_classification import visualize_classification

# Example 1: Visualize a single spectrum
print("=" * 70)
print("Example 1: Visualizing a single spectrum")
print("=" * 70)

visualize_classification(
    spectrum_file="data/test/testing_real_data/G-1 With Peaks.txt",
        model_dir="models/saved_models_v3",
    output_dir="visualizations",
    show_plot=True,
    max_feature_maps=8
)

# Example 2: Visualize all spectra in a directory (batch mode)
print("\n" + "=" * 70)
print("Example 2: Batch visualization of all spectra")
print("=" * 70)

visualize_classification(
    spectrum_file="data/test/testing_real_data",
        model_dir="models/saved_models_v3",
    output_dir="visualizations",
    show_plot=False,  # Don't show plots in batch mode
    batch_mode=True,
    max_feature_maps=8
)

print("\nVisualization complete! Check the 'visualizations' directory for output files.")

