"""
Model Quality Diagnostic Script

This script helps diagnose whether inconsistent results are due to:
1. Non-deterministic operations (should be fixed now)
2. Low model confidence (model quality issue)
3. Poor training data quality
4. Mismatch between training and test data
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from scripts.inference_simple import load_model_and_config, classify_raman_file, set_seeds


def diagnose_model_quality(file_path, n_runs=5):
    """
    Run multiple predictions and analyze consistency and confidence.
    
    Args:
        file_path: Path to spectrum file
        n_runs: Number of times to run prediction
    """
    print("=" * 70)
    print("Model Quality Diagnostic")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seeds(42)
    
    # Load model
    print("\n[1] Loading model...")
    model, target_grid, idx_to_class, device = load_model_and_config(seed=42)
    print(f"  ✓ Model loaded: {len(idx_to_class)} classes")
    
    # Run predictions multiple times
    print(f"\n[2] Running {n_runs} predictions on: {Path(file_path).name}")
    results = []
    
    for i in range(n_runs):
        result = classify_raman_file(
            file_path, model, target_grid, idx_to_class, device
        )
        results.append(result)
        print(f"  Run {i+1}: {result['predicted_class']} (conf: {result['confidence']:.4f})")
    
    # Analyze results
    print("\n[3] Analysis:")
    print("-" * 70)
    
    # Check consistency
    classes = [r['predicted_class'] for r in results]
    confs = [r['confidence'] for r in results]
    
    unique_classes = len(set(classes))
    if unique_classes == 1:
        print(f"  ✓ Consistent predictions: All runs predicted '{classes[0]}'")
    else:
        print(f"  ✗ Inconsistent predictions: {unique_classes} different classes")
        print(f"    Classes: {set(classes)}")
    
    # Check confidence
    avg_conf = np.mean(confs)
    std_conf = np.std(confs)
    max_conf = np.max(confs)
    
    print(f"\n  Confidence Statistics:")
    print(f"    Average: {avg_conf:.4f}")
    print(f"    Std Dev: {std_conf:.6f}")
    print(f"    Max: {max_conf:.4f}")
    
    # Interpret confidence
    print(f"\n  Confidence Interpretation:")
    if avg_conf > 0.7:
        print(f"    ✓ High confidence ({avg_conf:.1%}) - Model is confident")
    elif avg_conf > 0.5:
        print(f"    ⚠ Moderate confidence ({avg_conf:.1%}) - Model is somewhat uncertain")
    elif avg_conf > 0.3:
        print(f"    ⚠ Low confidence ({avg_conf:.1%}) - Model is uncertain")
    else:
        print(f"    ✗ Very low confidence ({avg_conf:.1%}) - Model is very uncertain")
    
    # Check class probabilities distribution
    print(f"\n  Class Probability Distribution:")
    all_probs = results[0]['class_probabilities']
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    
    for i, (cls, prob) in enumerate(sorted_probs[:5]):
        marker = "→" if i == 0 else " "
        print(f"    {marker} {cls:25s}: {prob:.4f}")
    
    # Check if probabilities are close (indicates uncertainty)
    top2_diff = sorted_probs[0][1] - sorted_probs[1][1]
    print(f"\n  Top 2 classes difference: {top2_diff:.4f}")
    if top2_diff < 0.1:
        print(f"    ⚠ Very close probabilities - Model can't clearly distinguish")
    elif top2_diff < 0.2:
        print(f"    ⚠ Close probabilities - Model is somewhat uncertain")
    else:
        print(f"    ✓ Clear distinction between top classes")
    
    # Diagnose potential issues
    print("\n[4] Potential Issues:")
    print("-" * 70)
    
    issues = []
    if avg_conf < 0.5:
        issues.append("Low confidence suggests:")
        issues.append("  • Model may not have been trained well")
        issues.append("  • Training data may not match test data")
        issues.append("  • Model may need more training data")
        issues.append("  • Classes may be too similar")
    
    if top2_diff < 0.1:
        issues.append("Close probabilities suggest:")
        issues.append("  • Spectrum doesn't clearly match any class")
        issues.append("  • Model needs better training")
        issues.append("  • Consider ensemble methods or better features")
    
    if unique_classes > 1:
        issues.append("Inconsistent predictions suggest:")
        issues.append("  • Non-deterministic operations (should be fixed)")
        issues.append("  • Numerical precision issues")
        issues.append("  • Model is right at decision boundary")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✓ No obvious issues detected")
    
    # Recommendations
    print("\n[5] Recommendations:")
    print("-" * 70)
    
    if avg_conf < 0.5:
        print("  • Check training data quality and quantity")
        print("  • Verify preprocessing matches training exactly")
        print("  • Consider retraining with more data")
        print("  • Check if test spectrum is similar to training data")
    
    if top2_diff < 0.15:
        print("  • Model may benefit from:")
        print("    - More training epochs")
        print("    - Better data augmentation")
        print("    - More diverse training samples")
        print("    - Feature engineering")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Default to first file in testing_real_data
        test_dir = Path("data/test/testing_real_data")
        files = list(test_dir.glob("*.txt")) + list(test_dir.glob("*.csv"))
        if files:
            file_path = str(files[0])
        else:
            print("No test files found. Usage: python diagnose_model.py <file_path>")
            sys.exit(1)
    
    diagnose_model_quality(file_path)



