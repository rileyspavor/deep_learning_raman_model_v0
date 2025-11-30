"""
Preprocessing Module for Raman Spectroscopy Data

This module provides independent, reusable functions for preprocessing
Raman spectra: alignment, baseline correction, normalization, and smoothing.
"""

import numpy as np
from scipy import interpolate, signal
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple, Union, Dict


def align_spectrum(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    target_grid: np.ndarray,
    method: str = "linear"
) -> np.ndarray:
    """
    Align/resample a spectrum to a common wavenumber grid.
    
    Args:
        wavenumbers: Original wavenumber array
        intensities: Original intensity array
        target_grid: Target wavenumber grid to interpolate onto
        method: Interpolation method ('linear', 'cubic', 'nearest')
    
    Returns:
        Interpolated intensities on target_grid
    """
    # Remove any NaN or inf values
    valid_mask = np.isfinite(wavenumbers) & np.isfinite(intensities)
    wavenumbers = wavenumbers[valid_mask]
    intensities = intensities[valid_mask]
    
    if len(wavenumbers) == 0:
        raise ValueError("No valid data points in spectrum")
    
    # Ensure target grid is within data range
    min_w = max(wavenumbers.min(), target_grid.min())
    max_w = min(wavenumbers.max(), target_grid.max())
    target_grid = target_grid[(target_grid >= min_w) & (target_grid <= max_w)]
    
    # Interpolate
    if method == "linear":
        f = interpolate.interp1d(
            wavenumbers, intensities,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
    elif method == "cubic":
        f = interpolate.interp1d(
            wavenumbers, intensities,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )
    elif method == "nearest":
        f = interpolate.interp1d(
            wavenumbers, intensities,
            kind='nearest',
            bounds_error=False,
            fill_value=0.0
        )
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    aligned_intensities = f(target_grid)
    
    return aligned_intensities


def correct_baseline(
    intensities: np.ndarray,
    method: str = "als",
    lam: float = 1e4,
    p: float = 0.001,
    n_iter: int = 10,
    poly_order: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct baseline/fluorescence background from Raman spectrum.
    
    Args:
        intensities: Intensity array
        method: Method to use ('als', 'polynomial', 'rolling_ball', 'none')
        lam: Smoothness parameter for ALS (higher = smoother)
        p: Asymmetry parameter for ALS (0-1, lower = more asymmetric)
        n_iter: Number of iterations for ALS
        poly_order: Polynomial order for polynomial method
    
    Returns:
        Tuple of (corrected_intensities, baseline)
        If method='none', returns (original_intensities, zeros)
    """
    if method == "none":
        return intensities.copy(), np.zeros_like(intensities)
    
    if method == "als":
        # Asymmetric Least Squares baseline correction
        baseline = _als_baseline(intensities, lam=lam, p=p, n_iter=n_iter)
        corrected = intensities - baseline
        corrected = np.maximum(corrected, 0)  # Ensure non-negative
        return corrected, baseline
    
    elif method == "polynomial":
        # Polynomial baseline correction
        x = np.arange(len(intensities))
        coeffs = np.polyfit(x, intensities, poly_order)
        baseline = np.polyval(coeffs, x)
        corrected = intensities - baseline
        corrected = np.maximum(corrected, 0)
        return corrected, baseline
    
    elif method == "rolling_ball":
        # Rolling ball baseline correction
        baseline = _rolling_ball_baseline(intensities)
        corrected = intensities - baseline
        corrected = np.maximum(corrected, 0)
        return corrected, baseline
    
    else:
        raise ValueError(f"Unknown baseline correction method: {method}")


def _als_baseline(
    y: np.ndarray,
    lam: float = 1e4,
    p: float = 0.001,
    n_iter: int = 10
) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction (internal helper).
    """
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2), format='csr')
    w = np.ones(L)
    
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L), format='csr')
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    
    return z


def _rolling_ball_baseline(intensities: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Rolling ball baseline correction (internal helper).
    """
    # Simple min filter approach
    baseline = signal.savgol_filter(
        signal.medfilt(intensities, kernel_size=window),
        window_length=window,
        polyorder=3
    )
    return baseline


def normalize_spectrum(
    intensities: np.ndarray,
    method: str = "max",
    g_peak_index: Optional[int] = None,
    g_peak_range: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Normalize spectrum intensities.
    
    Args:
        intensities: Intensity array
        method: Normalization method:
            - 'max': Divide by maximum intensity
            - 'area': Divide by area under curve
            - 'g_peak': Divide by G-peak intensity (requires g_peak_index or g_peak_range)
            - 'minmax': Min-max scaling to [0, 1]
            - 'zscore': Z-score normalization
        g_peak_index: Index of G-peak for 'g_peak' method
        g_peak_range: (start, end) indices for G-peak region
    
    Returns:
        Normalized intensity array
    """
    intensities = intensities.astype(float)
    
    if method == "max":
        max_val = np.max(intensities)
        if max_val > 0:
            return intensities / max_val
        return intensities
    
    elif method == "area":
        area = np.trapz(intensities)
        if area > 0:
            return intensities / area
        return intensities
    
    elif method == "g_peak":
        if g_peak_index is not None:
            g_intensity = intensities[g_peak_index]
        elif g_peak_range is not None:
            g_intensity = np.max(intensities[g_peak_range[0]:g_peak_range[1]])
        else:
            raise ValueError("g_peak method requires g_peak_index or g_peak_range")
        
        if g_intensity > 0:
            return intensities / g_intensity
        return intensities
    
    elif method == "minmax":
        min_val = np.min(intensities)
        max_val = np.max(intensities)
        if max_val > min_val:
            return (intensities - min_val) / (max_val - min_val)
        return intensities
    
    elif method == "zscore":
        mean = np.mean(intensities)
        std = np.std(intensities)
        if std > 0:
            return (intensities - mean) / std
        return intensities - mean
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def smooth_spectrum(
    intensities: np.ndarray,
    method: str = "savgol",
    window_length: int = 5,
    polyorder: int = 3,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Smooth spectrum to reduce noise.
    
    Args:
        intensities: Intensity array
        method: Smoothing method ('savgol', 'gaussian', 'moving_average', 'none')
        window_length: Window length for savgol/moving_average (must be odd for savgol)
        polyorder: Polynomial order for savgol (must be < window_length)
        sigma: Standard deviation for Gaussian smoothing
    
    Returns:
        Smoothed intensity array
    """
    if method == "none":
        return intensities.copy()
    
    if method == "savgol":
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        # Ensure polyorder < window_length
        polyorder = min(polyorder, window_length - 1)
        
        if len(intensities) < window_length:
            return intensities.copy()
        
        return signal.savgol_filter(intensities, window_length, polyorder)
    
    elif method == "gaussian":
        return signal.gaussian_filter1d(intensities, sigma=sigma)
    
    elif method == "moving_average":
        return np.convolve(intensities, np.ones(window_length)/window_length, mode='same')
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def preprocess_spectrum(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    target_grid: Optional[np.ndarray] = None,
    align: bool = True,
    baseline_correct: bool = True,
    baseline_method: str = "als",
    normalize: bool = True,
    normalize_method: str = "max",
    smooth: bool = False,
    smooth_method: str = "savgol",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for a single spectrum.
    
    Args:
        wavenumbers: Original wavenumber array
        intensities: Original intensity array
        target_grid: Target wavenumber grid (if None, uses original)
        align: Whether to align to target_grid
        baseline_correct: Whether to correct baseline
        baseline_method: Baseline correction method
        normalize: Whether to normalize
        normalize_method: Normalization method
        smooth: Whether to smooth
        smooth_method: Smoothing method
        **kwargs: Additional arguments passed to individual functions
    
    Returns:
        Tuple of (processed_wavenumbers, processed_intensities)
    """
    processed_intensities = intensities.copy()
    processed_wavenumbers = wavenumbers.copy()
    
    # Step 1: Align to common grid
    if align and target_grid is not None:
        processed_intensities = align_spectrum(
            processed_wavenumbers,
            processed_intensities,
            target_grid,
            method=kwargs.get('align_method', 'linear')
        )
        processed_wavenumbers = target_grid
    
    # Step 2: Baseline correction
    if baseline_correct:
        processed_intensities, _ = correct_baseline(
            processed_intensities,
            method=baseline_method,
            lam=kwargs.get('baseline_lam', 1e4),
            p=kwargs.get('baseline_p', 0.001),
            n_iter=kwargs.get('baseline_n_iter', 10),
            poly_order=kwargs.get('baseline_poly_order', 3)
        )
    
    # Step 3: Smoothing (before normalization to preserve relative intensities)
    if smooth:
        processed_intensities = smooth_spectrum(
            processed_intensities,
            method=smooth_method,
            window_length=kwargs.get('smooth_window_length', 5),
            polyorder=kwargs.get('smooth_polyorder', 3),
            sigma=kwargs.get('smooth_sigma', 1.0)
        )
    
    # Step 4: Normalization
    if normalize:
        processed_intensities = normalize_spectrum(
            processed_intensities,
            method=normalize_method,
            g_peak_index=kwargs.get('g_peak_index', None),
            g_peak_range=kwargs.get('g_peak_range', None)
        )
    
    return processed_wavenumbers, processed_intensities


def preprocess_dataset(
    wavenumbers_list: list,
    intensities_list: list,
    target_grid: Optional[np.ndarray] = None,
    **preprocessing_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess a dataset of spectra.
    
    Args:
        wavenumbers_list: List of wavenumber arrays
        intensities_list: List of intensity arrays
        target_grid: Common wavenumber grid (if None, inferred from data)
        **preprocessing_kwargs: Arguments passed to preprocess_spectrum
    
    Returns:
        Tuple of (wavenumbers_grid, processed_spectra_array)
        processed_spectra_array has shape (n_samples, n_wavenumbers)
    """
    # Infer target grid if not provided
    if target_grid is None:
        all_wavenumbers = np.concatenate(wavenumbers_list)
        min_w = np.min(all_wavenumbers)
        max_w = np.max(all_wavenumbers)
        # Create a common grid (e.g., 1 cm^-1 resolution)
        target_grid = np.arange(min_w, max_w + 1, 1.0)
    
    processed_spectra = []
    
    for wavenumbers, intensities in zip(wavenumbers_list, intensities_list):
        _, processed = preprocess_spectrum(
            wavenumbers,
            intensities,
            target_grid=target_grid,
            **preprocessing_kwargs
        )
        processed_spectra.append(processed)
    
    return target_grid, np.array(processed_spectra)


def preprocess_aligned_spectra(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    align: bool = False,
    baseline_correct: bool = True,
    baseline_method: str = "als",
    normalize: bool = True,
    normalize_method: str = "max",
    smooth: bool = False,
    smooth_method: str = "savgol",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess already-aligned spectra (e.g., from .npz file).
    More efficient when spectra are already on the same grid.
    
    Args:
        spectra: Array of shape (n_samples, n_wavenumbers) - already aligned
        wavenumbers: Array of shape (n_wavenumbers,) - common wavenumber grid
        align: Whether to align (usually False if already aligned)
        baseline_correct: Whether to correct baseline
        baseline_method: Baseline correction method
        normalize: Whether to normalize
        normalize_method: Normalization method
        smooth: Whether to smooth
        smooth_method: Smoothing method
        **kwargs: Additional preprocessing arguments
    
    Returns:
        Tuple of (wavenumbers, processed_spectra)
        processed_spectra has shape (n_samples, n_wavenumbers)
    """
    processed_spectra = spectra.copy()
    
    # Baseline correction (applied to each spectrum)
    if baseline_correct:
        for i in range(processed_spectra.shape[0]):
            processed_spectra[i], _ = correct_baseline(
                processed_spectra[i],
                method=baseline_method,
                lam=kwargs.get('baseline_lam', 1e4),
                p=kwargs.get('baseline_p', 0.001),
                n_iter=kwargs.get('baseline_n_iter', 10),
                poly_order=kwargs.get('baseline_poly_order', 3)
            )
    
    # Smoothing (applied to each spectrum)
    if smooth:
        for i in range(processed_spectra.shape[0]):
            processed_spectra[i] = smooth_spectrum(
                processed_spectra[i],
                method=smooth_method,
                window_length=kwargs.get('smooth_window_length', 5),
                polyorder=kwargs.get('smooth_polyorder', 3),
                sigma=kwargs.get('smooth_sigma', 1.0)
            )
    
    # Normalization (applied to each spectrum)
    if normalize:
        for i in range(processed_spectra.shape[0]):
            processed_spectra[i] = normalize_spectrum(
                processed_spectra[i],
                method=normalize_method,
                g_peak_index=kwargs.get('g_peak_index', None),
                g_peak_range=kwargs.get('g_peak_range', None)
            )
    
    return wavenumbers, processed_spectra


def compute_peak_ratios(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    d_peak_range: Tuple[float, float] = (1300, 1400),
    g_peak_range: Tuple[float, float] = (1550, 1620),
    d2_peak_range: Tuple[float, float] = (2650, 2750)
) -> Dict[str, float]:
    """
    Compute D/G and 2D/G peak intensity ratios.
    
    Args:
        intensities: Intensity array
        wavenumbers: Wavenumber array
        d_peak_range: (min, max) wavenumber range for D peak
        g_peak_range: (min, max) wavenumber range for G peak
        d2_peak_range: (min, max) wavenumber range for 2D peak
    
    Returns:
        Dictionary with 'D/G' and '2D/G' ratios
    """
    def get_peak_intensity(w_range):
        mask = (wavenumbers >= w_range[0]) & (wavenumbers <= w_range[1])
        if np.any(mask):
            return np.max(intensities[mask])
        return 0.0
    
    d_intensity = get_peak_intensity(d_peak_range)
    g_intensity = get_peak_intensity(g_peak_range)
    d2_intensity = get_peak_intensity(d2_peak_range)
    
    ratios = {}
    if g_intensity > 0:
        ratios['D/G'] = d_intensity / g_intensity
        ratios['2D/G'] = d2_intensity / g_intensity
    else:
        ratios['D/G'] = 0.0
        ratios['2D/G'] = 0.0
    
    return ratios

