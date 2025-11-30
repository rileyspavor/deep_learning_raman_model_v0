"""
CNN Classification Visualization Component

This module provides tools to visualize how the CNN processes and classifies
Raman spectroscopy spectra. It shows:
- Input spectrum
- Intermediate feature maps from convolutional layers
- Prediction probabilities
- Layer-wise activations

Usage:
    from visualize_cnn_classification import visualize_classification
    
    # Visualize a single spectrum
    visualize_classification(
        spectrum_file="data/test/testing_real_data/G-1 With Peaks.txt",
        model_dir="models/saved_models_v3",
        output_dir="visualizations"
    )
    
    # Or visualize multiple spectra
    visualize_classification(
        spectrum_file="data/test/testing_real_data",
        model_dir="models/saved_models_v3",
        output_dir="visualizations",
        batch_mode=True
    )
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
from typing import Optional, List, Dict, Tuple
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

from scipy.interpolate import interp1d
import sys
from pathlib import Path
# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from scripts.test_real_data_v3 import load_v3_model, align_spectrum_to_grid


class ActivationHook:
    """Hook to capture intermediate activations from the model."""
    
    def __init__(self):
        self.activations = {}
    
    def __call__(self, module, input, output):
        """Store the activation output."""
        module_name = module.__class__.__name__
        if isinstance(output, torch.Tensor):
            self.activations[module_name] = output.detach().cpu()
        return output


class CNNVisualizer:
    """Visualization component for CNN classification."""
    
    def __init__(self, model, target_grid, class_names, device='cpu'):
        """
        Initialize the visualizer.
        
        Args:
            model: Trained Raman1DCNN model
            target_grid: Target wavenumber grid
            class_names: List of class names
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.target_grid = target_grid
        self.class_names = class_names
        self.device = device
        
        # Register hooks to capture activations
        self.hooks = []
        self.activation_hook = ActivationHook()
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture layer activations."""
        # Clear existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Register hooks on convolutional layers and pooling layers
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.MaxPool1d, torch.nn.BatchNorm1d)):
                hook = module.register_forward_hook(self.activation_hook)
                self.hooks.append(hook)
    
    def load_spectrum(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a Raman spectrum from file.
        
        Args:
            file_path: Path to spectrum file
            
        Returns:
            Tuple of (wavenumbers, intensities)
        """
        try:
            import pandas as pd
            
            # Read file and detect if it has a header
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            # Check if first line looks like a header
            has_header = not first_line.replace('.', '').replace('-', '').replace('e', '').replace('E', '').replace('+', '').strip().replace(' ', '').isdigit()
            
            if has_header:
                df = pd.read_csv(file_path, sep=r'\s+', skipinitialspace=True)
                wavenumbers = df.iloc[:, 0].values
                intensities = df.iloc[:, 1].values if len(df.columns) > 1 else df.iloc[:, 0].values
            else:
                data = np.loadtxt(file_path)
                wavenumbers = data[:, 0]
                intensities = data[:, 1]
            
            return wavenumbers, intensities
        except Exception as e:
            raise ValueError(f"Error loading spectrum from {file_path}: {e}")
    
    def preprocess_spectrum(self, wavenumbers: np.ndarray, intensities: np.ndarray) -> torch.Tensor:
        """
        Preprocess spectrum to match model input format.
        
        Args:
            wavenumbers: Original wavenumber array
            intensities: Original intensity array
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Align to target grid
        aligned_intensities = align_spectrum_to_grid(wavenumbers, intensities, self.target_grid)
        
        # Convert to tensor: (batch=1, channels=1, length=N)
        x = torch.tensor(aligned_intensities, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return x.to(self.device)
    
    def classify_with_activations(self, x: torch.Tensor) -> Dict:
        """
        Classify spectrum and capture intermediate activations.
        
        Args:
            x: Input tensor (batch_size, 1, length)
            
        Returns:
            Dictionary with predictions and activations
        """
        # Clear previous activations
        self.activation_hook.activations = {}
        
        # Forward pass
        with torch.no_grad():
            class_logits, _ = self.model(x)
            probs = F.softmax(class_logits, dim=1).cpu().numpy()[0]
        
        pred_idx = int(probs.argmax())
        predicted_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Get all probabilities
        prob_dict = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': prob_dict,
            'logits': class_logits.cpu().numpy()[0],
            'activations': self.activation_hook.activations.copy()
        }
    
    def extract_layer_outputs(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract outputs from each layer in the backbone.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping layer names to outputs
        """
        layer_outputs = {}
        
        # Manually forward through backbone to capture each layer
        with torch.no_grad():
            current = x
            block_idx = 0
            
            # Process through each module in Sequential backbone
            # Pattern: Conv1DBlock, MaxPool1d, Conv1DBlock, MaxPool1d, ...
            for i, module in enumerate(self.model.backbone):
                # Get output before this module
                prev_output = current
                
                # Forward through module
                current = module(current)
                
                # If this is a Conv1DBlock, capture output after the block
                if hasattr(module, 'conv') and isinstance(module.conv, torch.nn.Conv1d):
                    # This is a Conv1DBlock - save output after the block
                    block_idx = (i // 2) + 1
                    layer_outputs[f'conv_block_{block_idx}_conv'] = current.cpu().numpy()[0]
                
                # If this is MaxPool1d, capture output after pooling
                elif isinstance(module, torch.nn.MaxPool1d):
                    block_idx = (i // 2) + 1
                    layer_outputs[f'conv_block_{block_idx}_pool'] = current.cpu().numpy()[0]
        
        return layer_outputs
    
    def visualize_classification(
        self,
        spectrum_file: str,
        output_path: Optional[str] = None,
        show_plot: bool = True,
        max_feature_maps: int = 8
    ):
        """
        Create comprehensive visualization of CNN classification.
        
        Args:
            spectrum_file: Path to spectrum file
            output_path: Path to save visualization (optional)
            show_plot: Whether to display the plot
            max_feature_maps: Maximum number of feature maps to show per layer
        """
        # Load spectrum
        wavenumbers, intensities = self.load_spectrum(spectrum_file)
        
        # Preprocess
        x = self.preprocess_spectrum(wavenumbers, intensities)
        aligned_intensities = x.cpu().numpy()[0, 0]
        
        # Classify and get activations
        results = self.classify_with_activations(x)
        layer_outputs = self.extract_layer_outputs(x)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Input Spectrum
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.target_grid, aligned_intensities, 'b-', linewidth=1.5)
        ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        ax1.set_ylabel('Intensity', fontsize=12)
        ax1.set_title(f'Input Spectrum: {Path(spectrum_file).name}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(self.target_grid.min(), self.target_grid.max())
        
        # 2. Prediction Probabilities
        ax2 = fig.add_subplot(gs[1, :])
        sorted_probs = sorted(results['probabilities'].items(), key=lambda x: x[1], reverse=True)
        classes = [item[0] for item in sorted_probs]
        probs = [item[1] for item in sorted_probs]
        colors = ['green' if i == 0 else 'gray' for i in range(len(classes))]
        bars = ax2.barh(range(len(classes)), probs, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(classes)))
        ax2.set_yticklabels(classes, fontsize=10)
        ax2.set_xlabel('Probability', fontsize=12)
        ax2.set_title(f'Classification Results: {results["predicted_class"]} ({results["confidence"]:.2%} confidence)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add probability values on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=9)
        
        # 3-6. Feature Maps from Each Convolutional Block
        conv_layers = [f'conv_block_{i}_conv' for i in range(1, 5)]
        
        for idx, layer_name in enumerate(conv_layers):
            if layer_name in layer_outputs:
                ax = fig.add_subplot(gs[2 + idx // 3, idx % 3])
                
                feature_maps = layer_outputs[layer_name]
                n_features = feature_maps.shape[0]
                
                # Show up to max_feature_maps
                n_show = min(max_feature_maps, n_features)
                indices = np.linspace(0, n_features - 1, n_show, dtype=int)
                
                # Calculate wavenumber grid for this layer (after pooling)
                pool_factor = 2 ** (idx + 1)  # Each block has pooling
                layer_length = feature_maps.shape[1]
                layer_wavenumbers = np.linspace(
                    self.target_grid.min(),
                    self.target_grid.max(),
                    layer_length
                )
                
                # Plot selected feature maps
                for i, feat_idx in enumerate(indices):
                    ax.plot(layer_wavenumbers, feature_maps[feat_idx], 
                           alpha=0.6, linewidth=1, label=f'FM {feat_idx}')
                
                ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=10)
                ax.set_ylabel('Activation', fontsize=10)
                ax.set_title(f'Block {idx + 1} Feature Maps\n({n_features} total, showing {n_show})', 
                           fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right', fontsize=7, ncol=2)
        
        # Add overall title
        fig.suptitle('CNN Classification Visualization', fontsize=16, fontweight='bold', y=0.995)
        
        # Save if requested
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return results
    
    def visualize_layer_activations(
        self,
        spectrum_file: str,
        output_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """
        Create detailed visualization of activations through the network.
        
        Args:
            spectrum_file: Path to spectrum file
            output_path: Path to save visualization (optional)
            show_plot: Whether to display the plot
        """
        # Load and preprocess
        wavenumbers, intensities = self.load_spectrum(spectrum_file)
        x = self.preprocess_spectrum(wavenumbers, intensities)
        aligned_intensities = x.cpu().numpy()[0, 0]
        
        # Get layer outputs
        layer_outputs = self.extract_layer_outputs(x)
        results = self.classify_with_activations(x)
        
        # Create figure
        fig, axes = plt.subplots(5, 1, figsize=(16, 14))
        
        # Plot 0: Input
        axes[0].plot(self.target_grid, aligned_intensities, 'b-', linewidth=1.5)
        axes[0].set_title('Input Spectrum', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Intensity', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(self.target_grid.min(), self.target_grid.max())
        
        # Plot 1-4: Each convolutional block output (averaged across channels)
        for idx in range(1, 5):
            layer_name = f'conv_block_{idx}_pool'
            if layer_name in layer_outputs:
                feature_maps = layer_outputs[layer_name]
                
                # Average across all feature maps
                avg_activation = np.mean(feature_maps, axis=0)
                
                # Calculate wavenumber grid for this layer
                pool_factor = 2 ** idx
                layer_length = feature_maps.shape[1]
                layer_wavenumbers = np.linspace(
                    self.target_grid.min(),
                    self.target_grid.max(),
                    layer_length
                )
                
                axes[idx].plot(layer_wavenumbers, avg_activation, 'r-', linewidth=1.5)
                axes[idx].set_title(f'Block {idx} Output (Averaged across {feature_maps.shape[0]} channels)', 
                                   fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('Avg Activation', fontsize=10)
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_xlim(self.target_grid.min(), self.target_grid.max())
        
        axes[-1].set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        
        # Add prediction info
        fig.suptitle(
            f'Layer-wise Activations: {Path(spectrum_file).name}\n'
            f'Predicted: {results["predicted_class"]} ({results["confidence"]:.2%})',
            fontsize=14, fontweight='bold', y=0.995
        )
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Activation visualization saved to: {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return results


def visualize_classification(
    spectrum_file: str,
    model_dir: str = "models/saved_models_v3",
    output_dir: Optional[str] = None,
    show_plot: bool = True,
    batch_mode: bool = False,
    max_feature_maps: int = 8
):
    """
    Main function to visualize CNN classification.
    
    Args:
        spectrum_file: Path to spectrum file or directory containing spectra
        model_dir: Directory containing saved model
        output_dir: Directory to save visualizations (optional)
        show_plot: Whether to display plots
        batch_mode: If True and spectrum_file is a directory, process all files
        max_feature_maps: Maximum feature maps to show per layer
    """
    # Load model
    print("Loading model...")
    model, target_grid, class_names, _, _ = load_v3_model(model_dir, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create visualizer
    visualizer = CNNVisualizer(model, target_grid, class_names, device)
    
    # Determine files to process
    spectrum_path = Path(spectrum_file)
    if spectrum_path.is_dir() and batch_mode:
        files_to_process = list(spectrum_path.glob("*.txt"))
        print(f"Found {len(files_to_process)} files to process")
    elif spectrum_path.is_file():
        files_to_process = [spectrum_path]
    else:
        raise ValueError(f"Invalid spectrum_file: {spectrum_file}")
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    results_list = []
    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] Processing: {file_path.name}")
        
        # Determine output path
        if output_dir:
            output_file = output_path / f"{file_path.stem}_visualization.png"
            activation_file = output_path / f"{file_path.stem}_activations.png"
        else:
            output_file = None
            activation_file = None
        
        try:
            # Create comprehensive visualization
            results = visualizer.visualize_classification(
                str(file_path),
                output_path=str(output_file) if output_file else None,
                show_plot=show_plot and len(files_to_process) == 1,  # Only show if single file
                max_feature_maps=max_feature_maps
            )
            
            # Create activation visualization
            visualizer.visualize_layer_activations(
                str(file_path),
                output_path=str(activation_file) if activation_file else None,
                show_plot=False  # Don't show individual plots in batch mode
            )
            
            results_list.append({
                'file': file_path.name,
                'predicted_class': results['predicted_class'],
                'confidence': results['confidence']
            })
            
        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")
            results_list.append({
                'file': file_path.name,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)
    for result in results_list:
        if 'error' in result:
            print(f"  ✗ {result['file']}: {result['error']}")
        else:
            print(f"  ✓ {result['file']}: {result['predicted_class']} ({result['confidence']:.2%})")
    
    if output_dir:
        print(f"\nVisualizations saved to: {output_dir}")
    
    return results_list


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize CNN classification on real test data")
    parser.add_argument(
        "--spectrum-file",
        type=str,
        default="data/test/testing_real_data",
        help="Path to spectrum file or directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/saved_models_v3",
        help="Directory containing saved model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (only save)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all files in directory (batch mode)"
    )
    parser.add_argument(
        "--max-feature-maps",
        type=int,
        default=8,
        help="Maximum feature maps to show per layer"
    )
    
    args = parser.parse_args()
    
    visualize_classification(
        spectrum_file=args.spectrum_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        show_plot=not args.no_show,
        batch_mode=args.batch or Path(args.spectrum_file).is_dir(),
        max_feature_maps=args.max_feature_maps
    )

