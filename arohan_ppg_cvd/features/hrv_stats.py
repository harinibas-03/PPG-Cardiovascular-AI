"""
PPG Heart Rate Variability (PRV) and Statistical Features
=========================================================
Computes:
1. Heart rate (bpm) and peak-to-peak interval (PPI) statistics (Mean PPI, SDNN, RMSSD, pNN50, CV).
2. Frequency domain power bands (LF, HF, LF/HF ratio) via Welch PSD.
3. Statistical distribution features (mean, std, skewness, kurtosis, spectral entropy).
"""

from typing import Dict, Any, List
import numpy as np
from scipy import stats, signal as sp_signal


def extract_hrv_features(peaks: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract Pulse Rate Variability (PRV) metrics from detected systolic peak sample indices.

    Parameters
    ----------
    peaks : np.ndarray
        Array of systolic peak indices.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict: PRV / HRV metrics.
    """
    if len(peaks) < 2:
        return {
            "hr_bpm": 0.0,
            "mean_ppi_ms": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50": 0.0,
            "cv_ppi": 0.0,
        }

    # Pulse-to-Pulse intervals in milliseconds
    ppi_samples = np.diff(peaks)
    ppi_ms = (ppi_samples / fs) * 1000.0

    # Filter out unphysiological PPI values (outside 300ms to 2000ms, i.e. 30 to 200 bpm)
    valid_ppi = ppi_ms[(ppi_ms >= 285.0) & (ppi_ms <= 2000.0)]
    if len(valid_ppi) < 2:
        valid_ppi = ppi_ms

    mean_ppi = float(np.mean(valid_ppi))
    hr_bpm = float(60000.0 / mean_ppi) if mean_ppi > 0 else 0.0
    sdnn = float(np.std(valid_ppi, ddof=1)) if len(valid_ppi) > 1 else 0.0

    # Successive differences
    successive_diffs = np.diff(valid_ppi)
    if len(successive_diffs) > 0:
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
        nn50 = np.sum(np.abs(successive_diffs) > 50.0)
        pnn50 = float((nn50 / len(successive_diffs)) * 100.0)
    else:
        rmssd = 0.0
        pnn50 = 0.0

    cv = float(sdnn / mean_ppi) if mean_ppi > 0 else 0.0

    return {
        "hr_bpm": round(hr_bpm, 2),
        "mean_ppi_ms": round(mean_ppi, 2),
        "sdnn_ms": round(sdnn, 2),
        "rmssd_ms": round(rmssd, 2),
        "pnn50": round(pnn50, 2),
        "cv_ppi": round(cv, 4),
    }


def extract_statistical_and_spectral_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract statistical moments and spectral power bands from conditioned PPG segment.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.size < 10:
        return {}

    mean_val = float(np.mean(sig))
    std_val = float(np.std(sig))
    var_val = float(np.var(sig))
    skew_val = float(stats.skew(sig))
    kurt_val = float(stats.kurtosis(sig))

    # Welch power spectral density
    nperseg = min(len(sig), max(16, int(fs * 4.0)))
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg)

    # Power in LF (0.04 - 0.15 Hz) and HF (0.15 - 0.4 Hz)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    cardiac_mask = (freqs >= 0.5) & (freqs <= 4.0)

    lf_power = float(np.sum(psd[lf_mask])) if np.any(lf_mask) else 0.0
    hf_power = float(np.sum(psd[hf_mask])) if np.any(hf_mask) else 0.0
    cardiac_power = float(np.sum(psd[cardiac_mask])) if np.any(cardiac_mask) else 0.0

    lf_hf_ratio = float(lf_power / (hf_power + 1e-8)) if hf_power > 0 else 0.0

    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-12)
    psd_norm = psd_norm[psd_norm > 0]
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm))) if len(psd_norm) > 0 else 0.0

    # Zero-crossing rate of centered signal
    centered = sig - mean_val
    zero_crossings = np.sum(np.diff(np.signbit(centered)) != 0)
    zcr = float(zero_crossings / len(sig))

    return {
        "stat_mean": round(mean_val, 5),
        "stat_std": round(std_val, 5),
        "stat_variance": round(var_val, 5),
        "stat_skewness": round(skew_val, 5),
        "stat_kurtosis": round(kurt_val, 5),
        "stat_zero_crossing_rate": round(zcr, 5),
        "spec_lf_power": round(lf_power, 6),
        "spec_hf_power": round(hf_power, 6),
        "spec_lf_hf_ratio": round(lf_hf_ratio, 4),
        "spec_cardiac_band_power": round(cardiac_power, 6),
        "spec_spectral_entropy": round(spectral_entropy, 4),
    }
