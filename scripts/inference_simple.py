"""
Simple Inference Script for Raman Spectroscopy Classification

This script loads a trained model and makes predictions on real Raman spectra.
It matches the EXACT preprocessing pipeline used during training (RAW format).

IMPORTANT: Training data analysis showed NO baseline correction and NO normalization
was applied during training. This script matches that format.

This script ensures deterministic results by:
- Setting random seeds
- Using model.eval() to disable dropout
- Using torch.no_grad() for inference

Usage:
    # Basic usage - load model and classify a file
    from inference_simple import load_model_and_config, classify_raman_file, print_prediction
    
    model, target_grid, idx_to_class, device = load_model_and_config()
    result = classify_raman_file('path/to/spectrum.txt', model, target_grid, idx_to_class, device)
    print_prediction(result)
    
    # Or run the script directly and add test files to the test_files list
    python inference_simple.py

Preprocessing (matches training - RAW format):
    - Trim to 800-3200 cm⁻¹ range
    - Linear interpolation to target_grid.npy (1500 points)
    - NO baseline correction
    - NO normalization
    - NO smoothing
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from io import StringIO
import random

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model and preprocessing
from src.model import Raman1DCNN
from src.preprocessing import preprocess_spectrum

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make operations deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Use deterministic algorithms where possible
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        pass  # Older PyTorch versions may not have this


def load_raman_txt(path):
    """
    Load a real Raman .txt file.
    Assumes 2-column file: wavenumber  intensity
    Skips header lines that don't start with a number.
    """
    lines = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            first = s.split()[0]
            try:
                float(first)
                lines.append(s)
            except ValueError:
                # header line (e.g. 'Raman_shift_cm-1 Intensity')
                continue

    buf = StringIO("\n".join(lines))
    data = np.loadtxt(buf)
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    w = data[:, 0]
    I = data[:, 1]
    return w, I


def load_model_and_config(
    model_path="models/saved_models/model_state.pth",
    target_grid_path="models/saved_models/target_grid.npy",
    class_names_path="models/saved_models/class_names.json",
    device=None,
    seed=42
):
    """
    Load trained model, target grid, and class names.
    
    Args:
        model_path: Path to saved model
        target_grid_path: Path to target grid
        class_names_path: Path to class names JSON
        device: Device to use (None for auto-detect)
        seed: Random seed for reproducibility
    
    Returns:
        model, target_grid, idx_to_class, device
    """
    # Set seeds for reproducibility
    set_seeds(seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load class names
    with open(class_names_path, "r") as f:
        class_map = json.load(f)  # {"0": "graphite", ...}
    idx_to_class = {int(k): v for k, v in class_map.items()}
    n_classes = len(idx_to_class)
    
    # Load target grid
    target_grid = np.load(target_grid_path)
    input_length = len(target_grid)
    
    # Create model with same architecture as training
    model = Raman1DCNN(
        input_length=input_length,
        n_classes=n_classes,
        n_channels=[32, 64, 128, 256],
        kernel_sizes=[7, 5, 5, 3],
        pool_sizes=[2, 2, 2, 2],
        use_batch_norm=True,
        dropout=0.3,
        fc_hidden=[128, 64],
        use_ordinal_head=False
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Try loading with strict=False to handle architecture mismatches
    # (e.g., if saved model had batch norm in FC layers but current doesn't)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        print("  Warning: Architecture mismatch detected. Loading with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()  # Disable dropout and batch norm training mode
    
    # Ensure dropout is disabled (double-check)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    
    return model, target_grid, idx_to_class, device


@torch.no_grad()
def predict_spectrum(model, I_1d, idx_to_class, device):
    """
    Run model inference on a preprocessed spectrum.
    
    Args:
        model: Trained model
        I_1d: Preprocessed intensity array of shape (N,) on target_grid
        idx_to_class: Dictionary mapping indices to class names
        device: Device to run on
    
    Returns:
        Tuple of (class_name, confidence, all_probs)
    """
    x = torch.from_numpy(I_1d).float().unsqueeze(0).unsqueeze(0)  # (1, 1, N)
    x = x.to(device)
    class_logits, ordinal_logits = model(x)  # shape (1, num_classes)
    probs = F.softmax(class_logits, dim=1)[0]  # (num_classes,)
    conf, idx = torch.max(probs, dim=0)
    cls_name = idx_to_class[int(idx.item())]
    return cls_name, float(conf.item()), probs.cpu().numpy()


def classify_raman_file(
    file_path,
    model,
    target_grid,
    idx_to_class,
    device,
    preprocessing_config=None
):
    """
    Complete pipeline: load file, preprocess, predict.
    
    Args:
        file_path: Path to Raman spectrum file
        model: Trained model
        target_grid: Target wavenumber grid
        idx_to_class: Dictionary mapping indices to class names
        device: Device to run on
        preprocessing_config: Optional preprocessing config (defaults to training settings)
    
    Returns:
        Dictionary with prediction results
    """
    # Load spectrum
    w_raw, I_raw = load_raman_txt(file_path)
    
    # Preprocess to match training data format (RAW - no baseline correction, no normalization)
    # Training data analysis showed:
    # - No baseline correction (negative values present)
    # - No normalization (max values ~3000, not ~1.0)
    # - Only alignment to target_grid (800-3200 cm⁻¹)
    
    from scipy import interpolate
    
    # Step 1: Trim/clip to [800, 3200] cm⁻¹ range (matching training)
    min_w = 800.0
    max_w = 3200.0
    mask = (w_raw >= min_w) & (w_raw <= max_w)
    w_trimmed = w_raw[mask]
    I_trimmed = I_raw[mask]
    
    if len(w_trimmed) == 0:
        raise ValueError(f"No data points in range [{min_w}, {max_w}] cm⁻¹")
    
    # Step 2: Remove any NaN or inf values
    valid_mask = np.isfinite(w_trimmed) & np.isfinite(I_trimmed)
    w_clean = w_trimmed[valid_mask]
    I_clean = I_trimmed[valid_mask]
    
    if len(w_clean) == 0:
        raise ValueError("No valid data points in spectrum")
    
    # Step 3: Interpolate to target_grid (linear interpolation, fill with 0.0 outside range)
    f = interpolate.interp1d(
        w_clean, I_clean,
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )
    I_processed = f(target_grid)
    
    # Step 4: NO baseline correction (matching training)
    # Step 5: NO normalization (matching training)
    # Step 6: NO smoothing (matching training)
    
    # Result: Raw intensities aligned to target_grid, ready for model
    
    # Predict
    cls_name, conf, probs = predict_spectrum(model, I_processed, idx_to_class, device)
    
    # Format results
    result = {
        'file': str(file_path),
        'predicted_class': cls_name,
        'confidence': conf,
        'class_probabilities': {
            idx_to_class[i]: float(prob) for i, prob in enumerate(probs)
        }
    }
    
    return result


def print_prediction(result):
    """Print prediction results in a readable format."""
    print(f"\nFile: {result['file']}")
    print(f"  Predicted class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"\n  All class probabilities:")
    for cls_name, prob in sorted(
        result['class_probabilities'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"    {cls_name}: {prob:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Raman Spectroscopy Classification - Inference")
    print("=" * 60)
    
    # Load model and config
    print("\nLoading model and configuration...")
    model, target_grid, idx_to_class, device = load_model_and_config()
    print(f"  Model loaded on device: {device}")
    print(f"  Target grid: {len(target_grid)} points")
    print(f"  Classes: {len(idx_to_class)}")
    print(f"  Class names: {list(idx_to_class.values())}")
    
    # Example: test on files (update these paths to your actual files)
    test_files = [
        # Add your test files here, e.g.:
        # "testing_real_data/GO 2_5_02.txt",
        # "testing_real_data/GO 3_03.txt",
        # "testing_real_data/graphite_03.txt",
    ]
    
    # If no test files specified, prompt for a file
    if not test_files:
        print("\nNo test files specified in script.")
        print("To test on a file, either:")
        print("  1. Add file paths to the 'test_files' list in the script")
        print("  2. Or use the classify_raman_file() function directly")
        print("\nExample usage:")
        print("  from inference_simple import load_model_and_config, classify_raman_file, print_prediction")
        print("  model, target_grid, idx_to_class, device = load_model_and_config()")
        print("  result = classify_raman_file('path/to/spectrum.txt', model, target_grid, idx_to_class, device)")
        print("  print_prediction(result)")
    else:
        print(f"\nTesting on {len(test_files)} files...")
        print("-" * 60)
        
        for file_path in test_files:
            try:
                result = classify_raman_file(
                    file_path, model, target_grid, idx_to_class, device
                )
                print_prediction(result)
            except Exception as e:
                print(f"\nError processing {file_path}:")
                print(f"  {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)

