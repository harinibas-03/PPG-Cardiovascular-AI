"""
PPG Signal Filtering and Denoising
===================================
Implementation of zero-phase Butterworth filtering, baseline wander removal,
and powerline noise attenuation for optical PPG signals.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal as sp_signal


def bandpass_filter(
    raw_signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 8.0,
    order: int = 3,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter.

    PPG frequency content:
    - Heart rate fundamental: 0.5 - 3.5 Hz (30 - 210 bpm)
    - Dicrotic notch and harmonic morphology: up to ~8.0 Hz
    - Frequencies <0.5 Hz: respiratory baseline wander
    - Frequencies >8.0 Hz: high-frequency sensor / motion noise

    Parameters
    ----------
    raw_signal : np.ndarray
        1D raw PPG amplitude array.
    fs : float
        Sampling frequency in Hz.
    lowcut : float
        Low cutoff frequency in Hz (default 0.5 Hz).
    highcut : float
        High cutoff frequency in Hz (default 8.0 Hz).
    order : int
        Filter order (default 3, 2-pass = effective 6th order).

    Returns
    -------
    np.ndarray : Filtered zero-phase signal.
    """
    sig = np.asarray(raw_signal, dtype=np.float64)
    if sig.size < 10:
        return sig

    nyquist = 0.5 * fs
    # Ensure highcut is strictly less than nyquist
    safe_highcut = min(highcut, nyquist * 0.95)
    safe_lowcut = max(lowcut, 0.05)

    if safe_lowcut >= safe_highcut:
        safe_lowcut = safe_highcut * 0.1

    low = safe_lowcut / nyquist
    high = safe_highcut / nyquist

    b, a = sp_signal.butter(order, [low, high], btype="bandpass")
    # Use zero-phase forward-backward filter to preserve peak locations
    padlen = min(sig.size - 1, 3 * max(len(a), len(b)))
    filtered = sp_signal.filtfilt(b, a, sig, padlen=padlen)
    return filtered.astype(np.float32)


def remove_baseline_wander(
    raw_signal: np.ndarray,
    fs: float,
    cutoff: float = 0.5,
    order: int = 2,
) -> np.ndarray:
    """
    Remove low-frequency respiration baseline wander using Butterworth high-pass filter.
    """
    sig = np.asarray(raw_signal, dtype=np.float64)
    if sig.size < 10:
        return sig

    nyquist = 0.5 * fs
    safe_cutoff = min(cutoff, nyquist * 0.9) / nyquist
    b, a = sp_signal.butter(order, safe_cutoff, btype="highpass")
    padlen = min(sig.size - 1, 3 * max(len(a), len(b)))
    detrended = sp_signal.filtfilt(b, a, sig, padlen=padlen)
    return detrended.astype(np.float32)


def notch_filter(
    raw_signal: np.ndarray,
    fs: float,
    freq: float = 50.0,
    q: float = 30.0,
) -> np.ndarray:
    """
    Attenuate powerline interference (50 Hz or 60 Hz) when sampling frequency allows it (fs > 2 * freq).
    """
    sig = np.asarray(raw_signal, dtype=np.float64)
    nyquist = 0.5 * fs
    if freq >= nyquist or sig.size < 10:
        # Not applicable if fs is too low (e.g., 30 Hz smartphone PPG)
        return sig.astype(np.float32)

    w0 = freq / nyquist
    b, a = sp_signal.iirnotch(w0, q)
    padlen = min(sig.size - 1, 3 * max(len(a), len(b)))
    filtered = sp_signal.filtfilt(b, a, sig, padlen=padlen)
    return filtered.astype(np.float32)


def clean_ppg_signal(
    raw_signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 8.0,
) -> np.ndarray:
    """
    Complete signal conditioning pipeline:
    1. Attenuates powerline interference if fs > 110 Hz.
    2. Zero-phase bandpass filter (0.5 - 8.0 Hz).
    3. Handles NaN or Inf gracefully.
    """
    sig = np.asarray(raw_signal, dtype=np.float32)
    # Replace NaNs/Infs with interpolation
    if np.any(np.isnan(sig)) or np.any(np.isinf(sig)):
        nans = np.isnan(sig) | np.isinf(sig)
        x = np.arange(len(sig))
        sig[nans] = np.interp(x[nans], x[~nans], sig[~nans])

    if fs > 110.0:
        sig = notch_filter(sig, fs=fs, freq=50.0)

    clean = bandpass_filter(sig, fs=fs, lowcut=lowcut, highcut=highcut)
    return clean
