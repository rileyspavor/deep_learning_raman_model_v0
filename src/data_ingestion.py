"""
Data Ingestion Module for Raman Spectroscopy Classification

This module handles loading real Raman spectra and generating synthetic spectra
for training. All functions are independent and reusable.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json


def load_raman_spectrum(
    file_path: Union[str, Path],
    wavenumber_col: str = "wavenumber",
    intensity_col: str = "intensity",
    delimiter: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single Raman spectrum from file.
    
    Args:
        file_path: Path to spectrum file (.txt, .csv, etc.)
        wavenumber_col: Column name or index for wavenumber data
        intensity_col: Column name or index for intensity data
        delimiter: File delimiter (None for auto-detect)
    
    Returns:
        Tuple of (wavenumbers, intensities) as numpy arrays
    """
    file_path = Path(file_path)
    
    # Auto-detect delimiter if not provided
    if delimiter is None:
        if file_path.suffix == '.csv':
            delimiter = ','
        else:
            delimiter = '\t'
    
    # Try to load as CSV/TSV with pandas first
    try:
        # Try reading with header first
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # Check if first row looks like data (numeric) or header (string)
        first_row_numeric = True
        try:
            pd.to_numeric(df.iloc[0, 0])
            pd.to_numeric(df.iloc[0, 1])
        except (ValueError, TypeError):
            first_row_numeric = False
        
        # If first row is not numeric, it's likely a header - try to use it
        if not first_row_numeric and len(df) > 1:
            # Try to find wavenumber column by common names
            wavenumber_cols = ['wavenumber', 'wavenumbers', 'raman_shift', 'raman_shift_cm-1', 
                              'raman_shift_cm', 'cm-1', 'cm^-1', 'shift', 'x', 'wavenumber (cm-1)']
            intensity_cols = ['intensity', 'intensities', 'y', 'counts', 'signal', 'raman_intensity']
            
            # Find matching column names (case-insensitive)
            df_lower = df.columns.str.lower()
            wavenumber_col_found = None
            intensity_col_found = None
            
            for col_name in wavenumber_cols:
                matches = [i for i, col in enumerate(df_lower) if col_name.lower() in col.lower()]
                if matches:
                    wavenumber_col_found = df.columns[matches[0]]
                    break
            
            for col_name in intensity_cols:
                matches = [i for i, col in enumerate(df_lower) if col_name.lower() in col.lower()]
                if matches:
                    intensity_col_found = df.columns[matches[0]]
                    break
            
            # If found, use them; otherwise use first two columns
            if wavenumber_col_found and intensity_col_found:
                wavenumbers = pd.to_numeric(df[wavenumber_col_found], errors='coerce').values
                intensities = pd.to_numeric(df[intensity_col_found], errors='coerce').values
            else:
                # Use first two columns, skip header row
                wavenumbers = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                intensities = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        else:
            # No header, try to use specified column names or indices
            if isinstance(wavenumber_col, str):
                if wavenumber_col in df.columns:
                    wavenumbers = pd.to_numeric(df[wavenumber_col], errors='coerce').values
                else:
                    wavenumbers = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
            else:
                wavenumbers = pd.to_numeric(df.iloc[:, wavenumber_col], errors='coerce').values
                
            if isinstance(intensity_col, str):
                if intensity_col in df.columns:
                    intensities = pd.to_numeric(df[intensity_col], errors='coerce').values
                else:
                    intensities = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
            else:
                intensities = pd.to_numeric(df.iloc[:, intensity_col], errors='coerce').values
        
        # Remove NaN values
        mask = np.isfinite(wavenumbers) & np.isfinite(intensities)
        wavenumbers = wavenumbers[mask]
        intensities = intensities[mask]
            
    except Exception as e:
        # Fallback: assume space/tab-separated two-column format, try to skip header
        try:
            # Try with skiprows=1 first (in case there's a header)
            data = np.loadtxt(file_path, skiprows=1)
            if data.ndim == 1:
                # Single row, try without skiprows
                data = np.loadtxt(file_path)
            wavenumbers = data[:, 0]
            intensities = data[:, 1]
        except (ValueError, IndexError):
            # If that fails, try reading line by line to skip non-numeric header
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            data_rows = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        float(parts[0])
                        float(parts[1])
                        data_rows.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue  # Skip non-numeric rows (headers)
            
            if not data_rows:
                raise ValueError(f"Could not parse numeric data from {file_path}")
            
            data = np.array(data_rows)
            wavenumbers = data[:, 0]
            intensities = data[:, 1]
    
    # Sort by wavenumber if needed
    sort_idx = np.argsort(wavenumbers)
    wavenumbers = wavenumbers[sort_idx]
    intensities = intensities[sort_idx]
    
    return wavenumbers, intensities


def load_raman_dataset(
    data_dir: Union[str, Path],
    label_mapping: Optional[Dict[str, int]] = None,
    file_pattern: str = "*.txt",
    wavenumber_col: str = "wavenumber",
    intensity_col: str = "intensity"
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[str]]:
    """
    Load multiple Raman spectra from a directory with labels.
    
    Args:
        data_dir: Directory containing spectrum files
        label_mapping: Dict mapping file prefixes/patterns to class labels
                      If None, assumes subdirectories are class names
        file_pattern: Glob pattern for spectrum files
        wavenumber_col: Column name for wavenumber data
        intensity_col: Column name for intensity data
    
    Returns:
        Tuple of (wavenumbers_list, intensities_list, labels, filenames)
    """
    data_dir = Path(data_dir)
    wavenumbers_list = []
    intensities_list = []
    labels = []
    filenames = []
    
    if label_mapping is None:
        # Assume subdirectories are class names
        for class_dir in sorted(data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            class_label = hash(class_name) % 1000  # Simple hash if no mapping
            
            for spectrum_file in class_dir.glob(file_pattern):
                w, i = load_raman_spectrum(
                    spectrum_file, wavenumber_col, intensity_col
                )
                wavenumbers_list.append(w)
                intensities_list.append(i)
                labels.append(class_label)
                filenames.append(str(spectrum_file))
    else:
        # Use provided label mapping
        for spectrum_file in data_dir.rglob(file_pattern):
            # Determine label from file path or name
            file_str = str(spectrum_file)
            label = None
            
            for pattern, class_label in label_mapping.items():
                if pattern in file_str:
                    label = class_label
                    break
            
            if label is not None:
                w, i = load_raman_spectrum(
                    spectrum_file, wavenumber_col, intensity_col
                )
                wavenumbers_list.append(w)
                intensities_list.append(i)
                labels.append(label)
                filenames.append(file_str)
    
    return wavenumbers_list, intensities_list, labels, filenames


def generate_synthetic_raman_spectrum(
    wavenumber_grid: np.ndarray,
    material_type: str = "graphene",
    noise_level: float = 0.02,
    baseline_level: float = 0.1,
    peak_variation: float = 0.1
) -> np.ndarray:
    """
    Generate a synthetic Raman spectrum with typical graphene peaks.
    
    Args:
        wavenumber_grid: Array of wavenumbers (cm^-1) to generate spectrum for
        material_type: Type of material ('graphite', 'graphene', 'GO', 'rGO', 'GNP')
        noise_level: Standard deviation of Gaussian noise relative to max intensity
        baseline_level: Baseline fluorescence level (relative to max)
        peak_variation: Random variation in peak positions/intensities (fraction)
    
    Returns:
        Synthetic intensity array
    """
    # Typical peak positions for graphene materials (cm^-1)
    peak_params = {
        "graphite": {
            "D": (1350, 50, 0.3),  # (position, width, intensity)
            "G": (1580, 20, 1.0),
            "2D": (2700, 40, 0.8)
        },
        "graphene": {
            "D": (1350, 30, 0.1),
            "G": (1580, 15, 1.0),
            "2D": (2700, 30, 2.0)
        },
        "GO": {
            "D": (1350, 60, 1.2),
            "G": (1590, 25, 1.0),
            "2D": (2700, 50, 0.2)
        },
        "rGO": {
            "D": (1350, 50, 0.8),
            "G": (1585, 20, 1.0),
            "2D": (2700, 40, 0.5)
        },
        "GNP": {
            "D": (1350, 45, 0.5),
            "G": (1580, 22, 1.0),
            "2D": (2700, 35, 1.2)
        }
    }
    
    # Default to not graphene if material type not found
    if material_type not in peak_params:
        material_type = "not graphene"
    
    params = peak_params[material_type]
    spectrum = np.zeros_like(wavenumber_grid)
    
    # Generate each peak as a Lorentzian
    for peak_name, (pos, width, intensity) in params.items():
        # Add random variation
        pos_var = pos * (1 + np.random.uniform(-peak_variation, peak_variation))
        width_var = width * (1 + np.random.uniform(-peak_variation, peak_variation))
        intensity_var = intensity * (1 + np.random.uniform(-peak_variation, peak_variation))
        
        # Lorentzian peak shape
        lorentzian = intensity_var / (1 + ((wavenumber_grid - pos_var) / (width_var / 2))**2)
        spectrum += lorentzian
    
    # Add baseline (exponential decay from low wavenumber)
    baseline = baseline_level * np.exp(-(wavenumber_grid - wavenumber_grid.min()) / 1000)
    spectrum += baseline
    
    # Add noise
    noise = np.random.normal(0, noise_level * spectrum.max(), size=spectrum.shape)
    spectrum += noise
    
    # Ensure non-negative
    spectrum = np.maximum(spectrum, 0)
    
    return spectrum


def generate_synthetic_dataset(
    wavenumber_grid: np.ndarray,
    n_samples_per_class: int,
    class_labels: Dict[str, int],
    noise_level: float = 0.02,
    baseline_level: float = 0.1,
    peak_variation: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset of Raman spectra.
    
    Args:
        wavenumber_grid: Array of wavenumbers (cm^-1)
        n_samples_per_class: Number of samples to generate per class
        class_labels: Dict mapping material type strings to integer labels
        noise_level: Standard deviation of noise
        baseline_level: Baseline fluorescence level
        peak_variation: Random variation in peak parameters
    
    Returns:
        Tuple of (spectra_array, labels_array)
        spectra_array: Shape (n_samples, n_wavenumbers)
        labels_array: Shape (n_samples,)
    """
    all_spectra = []
    all_labels = []
    
    for material_type, label in class_labels.items():
        for _ in range(n_samples_per_class):
            spectrum = generate_synthetic_raman_spectrum(
                wavenumber_grid,
                material_type=material_type,
                noise_level=noise_level,
                baseline_level=baseline_level,
                peak_variation=peak_variation
            )
            all_spectra.append(spectrum)
            all_labels.append(label)
    
    return np.array(all_spectra), np.array(all_labels)


def save_spectrum(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    file_path: Union[str, Path],
    label: Optional[str] = None
) -> None:
    """
    Save a Raman spectrum to file.
    
    Args:
        wavenumbers: Wavenumber array
        intensities: Intensity array
        file_path: Output file path
        label: Optional label/metadata to include
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        df = pd.DataFrame({
            'wavenumber': wavenumbers,
            'intensity': intensities
        })
        df.to_csv(file_path, index=False)
    else:
        # Save as space-separated text
        data = np.column_stack([wavenumbers, intensities])
        np.savetxt(file_path, data, fmt='%.6f', delimiter='\t')
    
    # Save metadata if label provided
    if label is not None:
        metadata_path = file_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({'label': label}, f)


def load_label_mapping(file_path: Union[str, Path]) -> Dict[str, int]:
    """
    Load label mapping from JSON file.
    
    Args:
        file_path: Path to JSON file with label mapping
    
    Returns:
        Dictionary mapping material names to integer labels
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_label_mapping(label_mapping: Dict[str, int], file_path: Union[str, Path]) -> None:
    """
    Save label mapping to JSON file.
    
    Args:
        label_mapping: Dictionary mapping material names to integer labels
        file_path: Output JSON file path
    """
    with open(file_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)


def load_npz_dataset(
    file_path: Union[str, Path],
    spectra_key: str = "spectra",
    wavenumbers_key: str = "wavenumbers",
    labels_key: str = "y",
    label_names_key: str = "label_names"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Load Raman spectroscopy dataset from .npz file.
    
    Args:
        file_path: Path to .npz file
        spectra_key: Key for spectra array in .npz file
        wavenumbers_key: Key for wavenumbers array in .npz file
        labels_key: Key for integer labels array in .npz file
        label_names_key: Key for label names array in .npz file
    
    Returns:
        Tuple of (spectra, wavenumbers, labels, label_names, metadata_dict)
        - spectra: Array of shape (n_samples, n_wavenumbers)
        - wavenumbers: Array of shape (n_wavenumbers,) or (n_samples, n_wavenumbers)
        - labels: Array of integer labels of shape (n_samples,)
        - label_names: Array of label name strings
        - metadata_dict: Dictionary with additional data (id_ig, i2d_ig, etc.)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    
    # Load main arrays
    spectra = data[spectra_key]
    wavenumbers = data[wavenumbers_key]
    labels = data[labels_key]
    
    # Load label names if available
    if label_names_key in data:
        label_names = data[label_names_key]
        # Handle case where label_names might be stored as object array
        if label_names.dtype == object:
            label_names = np.array([str(name) for name in label_names])
    else:
        label_names = None
    
    # Collect all other keys as metadata
    metadata = {}
    expected_keys = {spectra_key, wavenumbers_key, labels_key, label_names_key}
    
    for key in data.keys():
        if key not in expected_keys:
            metadata[key] = data[key]
    
    # Handle wavenumbers shape - if it's 2D, take first row (assuming all same)
    if wavenumbers.ndim == 2:
        if wavenumbers.shape[0] == spectra.shape[0]:
            # Each spectrum has its own wavenumber array
            # Check if they're all the same
            if np.allclose(wavenumbers[0], wavenumbers[-1]):
                wavenumbers = wavenumbers[0]
            else:
                # Different wavenumber grids - will need per-spectrum handling
                pass  # Keep as 2D for now
        else:
            wavenumbers = wavenumbers[0]
    
    # Ensure spectra is 2D
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)
    
    # Ensure labels is 1D
    if labels.ndim > 1:
        labels = labels.flatten()
    
    return spectra, wavenumbers, labels, label_names, metadata


def convert_npz_to_list_format(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    labels: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Convert numpy arrays from .npz format to list format expected by preprocessing.
    
    Args:
        spectra: Array of shape (n_samples, n_wavenumbers)
        wavenumbers: Array of shape (n_wavenumbers,) or (n_samples, n_wavenumbers)
        labels: Array of integer labels of shape (n_samples,)
    
    Returns:
        Tuple of (wavenumbers_list, intensities_list, labels_list)
    """
    n_samples = spectra.shape[0]
    
    # Handle wavenumbers
    if wavenumbers.ndim == 1:
        # Same wavenumber grid for all spectra
        wavenumbers_list = [wavenumbers.copy() for _ in range(n_samples)]
    else:
        # Different wavenumber grid per spectrum
        wavenumbers_list = [wavenumbers[i].copy() for i in range(n_samples)]
    
    # Convert spectra to list
    intensities_list = [spectra[i].copy() for i in range(n_samples)]
    
    # Convert labels to list
    labels_list = labels.tolist()
    
    return wavenumbers_list, intensities_list, labels_list

