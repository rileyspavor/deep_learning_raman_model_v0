"""
Inference Module for Raman Spectroscopy Classification

This module provides the inference pipeline for making predictions on new
Raman spectra. All functions are independent and reusable.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

from src.data_ingestion import load_raman_spectrum
from src.preprocessing import preprocess_spectrum, compute_peak_ratios


class RamanSpectrumClassifier:
    """
    Complete inference pipeline for Raman spectrum classification.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_grid: np.ndarray,
        class_names: Dict[int, str],
        device: Optional[torch.device] = None,
        preprocessing_config: Optional[Dict] = None
    ):
        """
        Initialize classifier.
        
        Args:
            model: Trained PyTorch model
            target_grid: Wavenumber grid used during training
            class_names: Dictionary mapping class indices to names
            device: Device to run inference on (None for auto-detect)
            preprocessing_config: Configuration for preprocessing
        """
        self.model = model
        self.target_grid = target_grid
        self.class_names = class_names
        self.preprocessing_config = preprocessing_config or {}
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray
    ) -> np.ndarray:
        """
        Preprocess a single spectrum.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
        
        Returns:
            Preprocessed spectrum on target_grid
        """
        _, processed = preprocess_spectrum(
            wavenumbers,
            intensities,
            target_grid=self.target_grid,
            **self.preprocessing_config
        )
        return processed
    
    def predict(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        return_probs: bool = True,
        return_confidence: bool = True
    ) -> Dict:
        """
        Predict class for a single spectrum.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            return_probs: Whether to return class probabilities
            return_confidence: Whether to return confidence score
        
        Returns:
            Dictionary with predictions and metadata
        """
        # Preprocess
        processed = self.preprocess(wavenumbers, intensities)
        
        # Convert to tensor
        spectrum_tensor = torch.FloatTensor(processed).unsqueeze(0).unsqueeze(0)
        spectrum_tensor = spectrum_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            class_logits, ordinal_logits = self.model(spectrum_tensor)
            
            if return_probs:
                class_probs = torch.softmax(class_logits, dim=1)
            else:
                class_probs = class_logits
            
            predicted_class_idx = torch.argmax(class_probs, dim=1).item()
            predicted_class_name = self.class_names.get(predicted_class_idx, f"Class_{predicted_class_idx}")
            
            result = {
                'predicted_class': predicted_class_name,
                'predicted_class_idx': predicted_class_idx
            }
            
            if return_probs:
                probs = class_probs[0].cpu().numpy()
                result['class_probabilities'] = {
                    self.class_names.get(i, f"Class_{i}"): float(prob)
                    for i, prob in enumerate(probs)
                }
            
            if return_confidence:
                max_prob = float(class_probs[0, predicted_class_idx].item())
                result['confidence'] = max_prob
            
            if ordinal_logits is not None:
                result['ordinal_prediction'] = float(ordinal_logits[0].item())
        
        return result
    
    def predict_batch(
        self,
        wavenumbers_list: List[np.ndarray],
        intensities_list: List[np.ndarray],
        batch_size: int = 32,
        return_probs: bool = True
    ) -> List[Dict]:
        """
        Predict classes for multiple spectra.
        
        Args:
            wavenumbers_list: List of wavenumber arrays
            intensities_list: List of intensity arrays
            batch_size: Batch size for processing
            return_probs: Whether to return probabilities
        
        Returns:
            List of prediction dictionaries
        """
        # Preprocess all spectra
        processed_spectra = []
        for w, i in zip(wavenumbers_list, intensities_list):
            processed = self.preprocess(w, i)
            processed_spectra.append(processed)
        
        processed_spectra = np.array(processed_spectra)
        
        # Batch predictions
        all_results = []
        n_samples = len(processed_spectra)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_spectra = processed_spectra[start_idx:end_idx]
            
            # Convert to tensor
            batch_tensor = torch.FloatTensor(batch_spectra).unsqueeze(1)
            batch_tensor = batch_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                class_logits, ordinal_logits = self.model(batch_tensor)
                
                if return_probs:
                    class_probs = torch.softmax(class_logits, dim=1)
                else:
                    class_probs = class_logits
                
                predicted_indices = torch.argmax(class_probs, dim=1).cpu().numpy()
                max_probs = torch.max(class_probs, dim=1)[0].cpu().numpy()
            
            # Format results
            for i, (pred_idx, max_prob) in enumerate(zip(predicted_indices, max_probs)):
                result = {
                    'predicted_class': self.class_names.get(pred_idx, f"Class_{pred_idx}"),
                    'predicted_class_idx': int(pred_idx),
                    'confidence': float(max_prob)
                }
                
                if return_probs:
                    probs = class_probs[i].cpu().numpy()
                    result['class_probabilities'] = {
                        self.class_names.get(j, f"Class_{j}"): float(prob)
                        for j, prob in enumerate(probs)
                    }
                
                all_results.append(result)
        
        return all_results
    
    def predict_from_file(
        self,
        file_path: Union[str, Path],
        wavenumber_col: str = "wavenumber",
        intensity_col: str = "intensity"
    ) -> Dict:
        """
        Load and predict from a spectrum file.
        
        Args:
            file_path: Path to spectrum file
            wavenumber_col: Column name for wavenumbers
            intensity_col: Column name for intensities
        
        Returns:
            Prediction dictionary
        """
        wavenumbers, intensities = load_raman_spectrum(
            file_path, wavenumber_col, intensity_col
        )
        return self.predict(wavenumbers, intensities)
    
    def quality_check(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        confidence_threshold: float = 0.7,
        min_quality_score: Optional[float] = None
    ) -> Dict:
        """
        Perform quality check on a spectrum with prediction.
        
        Args:
            wavenumbers: Wavenumber array
            intensities: Intensity array
            confidence_threshold: Minimum confidence for acceptance
            min_quality_score: Optional minimum quality score
        
        Returns:
            Dictionary with prediction and QC flags
        """
        # Get prediction
        prediction = self.predict(wavenumbers, intensities)
        
        # Compute peak ratios for additional QC
        peak_ratios = compute_peak_ratios(intensities, wavenumbers)
        
        # QC flags
        qc_result = {
            'prediction': prediction,
            'peak_ratios': peak_ratios,
            'qc_passed': True,
            'qc_flags': []
        }
        
        # Check confidence
        if prediction['confidence'] < confidence_threshold:
            qc_result['qc_passed'] = False
            qc_result['qc_flags'].append(
                f"Low confidence: {prediction['confidence']:.3f} < {confidence_threshold}"
            )
        
        # Check quality score if provided
        if min_quality_score is not None and 'ordinal_prediction' in prediction:
            if prediction['ordinal_prediction'] < min_quality_score:
                qc_result['qc_passed'] = False
                qc_result['qc_flags'].append(
                    f"Low quality score: {prediction['ordinal_prediction']:.3f} < {min_quality_score}"
                )
        
        # Check peak ratios (basic sanity checks)
        if peak_ratios['D/G'] > 2.0:  # Very high defect density
            qc_result['qc_flags'].append("High D/G ratio detected")
        
        if qc_result['qc_passed'] and len(qc_result['qc_flags']) > 0:
            qc_result['qc_passed'] = False
        
        return qc_result


def load_classifier(
    model_path: str,
    target_grid_path: str,
    class_names_path: str,
    device: Optional[torch.device] = None,
    preprocessing_config: Optional[Dict] = None
) -> RamanSpectrumClassifier:
    """
    Load a trained classifier from saved files.
    
    Args:
        model_path: Path to saved model (.pth file)
        target_grid_path: Path to saved target grid (.npy file)
        class_names_path: Path to saved class names (.json file)
        device: Device to load on
        preprocessing_config: Preprocessing configuration
    
    Returns:
        RamanSpectrumClassifier instance
    """
    # Load model
    from model import Raman1DCNN
    
    # Load class names to determine n_classes
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    n_classes = len(class_names)
    
    # Load target grid to determine input length
    target_grid = np.load(target_grid_path)
    input_length = len(target_grid)
    
    # Create model architecture (you may need to match training config)
    model = Raman1DCNN(input_length=input_length, n_classes=n_classes)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create classifier
    classifier = RamanSpectrumClassifier(
        model=model,
        target_grid=target_grid,
        class_names=class_names,
        device=device,
        preprocessing_config=preprocessing_config
    )
    
    return classifier


def save_classifier(
    classifier: RamanSpectrumClassifier,
    model_path: str,
    target_grid_path: str,
    class_names_path: str
) -> None:
    """
    Save a classifier to disk.
    
    Args:
        classifier: Classifier to save
        model_path: Path to save model
        target_grid_path: Path to save target grid
        class_names_path: Path to save class names
    """
    # Save model
    torch.save(classifier.model.state_dict(), model_path)
    
    # Save target grid
    np.save(target_grid_path, classifier.target_grid)
    
    # Save class names
    with open(class_names_path, 'w') as f:
        json.dump(classifier.class_names, f, indent=2)


def format_prediction_report(
    prediction: Dict,
    include_ratios: bool = False,
    peak_ratios: Optional[Dict] = None
) -> str:
    """
    Format prediction results as a human-readable report.
    
    Args:
        prediction: Prediction dictionary
        include_ratios: Whether to include peak ratios
        peak_ratios: Optional peak ratios dictionary
    
    Returns:
        Formatted report string
    """
    report = f"Predicted Class: {prediction['predicted_class']}\n"
    report += f"Confidence: {prediction['confidence']:.3f}\n"
    
    if 'class_probabilities' in prediction:
        report += "\nClass Probabilities:\n"
        for class_name, prob in sorted(
            prediction['class_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report += f"  {class_name}: {prob:.3f}\n"
    
    if include_ratios and peak_ratios is not None:
        report += f"\nPeak Ratios:\n"
        report += f"  D/G: {peak_ratios['D/G']:.3f}\n"
        report += f"  2D/G: {peak_ratios['2D/G']:.3f}\n"
    
    return report

