"""
PPG Signal Normalization
========================
Standardization and scaling utilities for PPG signals to ensure invariance to
skin pigmentation, optical contact pressure, and device sensor gains.
"""

from typing import Literal
import numpy as np


def normalize_signal(
    signal: np.ndarray,
    method: Literal["z_score", "min_max", "robust"] = "z_score",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize 1D PPG signal.

    Parameters
    ----------
    signal : np.ndarray
        Input 1D signal.
    method : str
        - 'z_score': Zero mean, unit variance (StandardScaler equivalent)
        - 'min_max': Scale into [0, 1] range
        - 'robust': Median subtraction divided by IQR (resilient to motion spikes)
    eps : float
        Numerical stability epsilon to prevent division by zero.

    Returns
    -------
    np.ndarray : Normalized signal.
    """
    sig = np.asarray(signal, dtype=np.float32)
    if sig.size == 0:
        return sig

    if method == "z_score":
        mean = np.mean(sig)
        std = np.std(sig)
        if std < eps:
            return np.zeros_like(sig)
        return (sig - mean) / (std + eps)

    elif method == "min_max":
        min_val = np.min(sig)
        max_val = np.max(sig)
        diff = max_val - min_val
        if diff < eps:
            return np.zeros_like(sig)
        return (sig - min_val) / (diff + eps)

    elif method == "robust":
        median = np.median(sig)
        q75, q25 = np.percentile(sig, [75, 25])
        iqr = q75 - q25
        if iqr < eps:
            return sig - median
        return (sig - median) / (iqr + eps)

    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'z_score', 'min_max', or 'robust'.")
