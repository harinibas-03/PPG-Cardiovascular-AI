"""
PPG Signal Quality Index (SQI) Assessment
=========================================
Automated artifact detection and signal quality classification to prevent
corrupt/noisy segments from entering feature extraction and machine learning pipelines.
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats, signal as sp_signal


def compute_skewness_sqi(signal: np.ndarray) -> float:
    """
    Skewness SQI (sSQI). Clean PPG signals exhibit a positive skewness due to the
    sharp systolic upstroke and slower diastolic runoff.
    """
    if len(signal) < 10 or np.all(signal == signal[0]):
        return 0.0
    return float(stats.skew(signal, bias=False))


def compute_kurtosis_sqi(signal: np.ndarray) -> float:
    """
    Kurtosis SQI (kSQI). Reflects the 'peakedness' of the pulse waveform.
    Clean physiological PPG typically exhibits kurtosis > 2.0.
    """
    if len(signal) < 10 or np.all(signal == signal[0]):
        return 0.0
    return float(stats.kurtosis(signal, fisher=False, bias=False))


def compute_power_sqi(signal: np.ndarray, fs: float, low_band: Tuple[float, float] = (0.7, 3.5)) -> float:
    """
    Relative Power SQI (pSQI). Ratio of power in the fundamental heart rate band (0.7 - 3.5 Hz)
    to total power in the cardiac-relevant band (0.5 - 8.0 Hz).
    """
    if len(signal) < int(fs * 2.0):
        return 0.0

    freqs, psd = sp_signal.welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 4.0)))
    total_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 8.0)])
    band_power = np.sum(psd[(freqs >= low_band[0]) & (freqs <= low_band[1])])

    if total_power <= 1e-12:
        return 0.0
    return float(np.clip(band_power / total_power, 0.0, 1.0))


def compute_perfusion_index(raw_signal: np.ndarray) -> float:
    """
    Perfusion Index (PI). Ratio of pulsatile blood flow (AC component) to non-pulsatile
    static blood volume (DC component), expressed as a percentage: (AC / DC) * 100.
    """
    sig = np.asarray(raw_signal, dtype=np.float64)
    if sig.size == 0:
        return 0.0
    dc = np.abs(np.mean(sig))
    if dc < 1e-6:
        return 0.0
    ac = (np.max(sig) - np.min(sig)) / 2.0
    return float((ac / dc) * 100.0)


def evaluate_signal_quality(
    signal: np.ndarray,
    fs: float,
    min_power_sqi: float = 0.40,
    min_kurtosis: float = 1.8,
) -> Tuple[bool, Dict[str, float]]:
    """
    Assess overall acceptability of a PPG segment.

    Parameters
    ----------
    signal : np.ndarray
        Conditioned 1D PPG segment.
    fs : float
        Sampling frequency in Hz.
    min_power_sqi : float
        Threshold for cardiac band power concentration.
    min_kurtosis : float
        Threshold for peakiness.

    Returns
    -------
    is_acceptable : bool
        True if signal quality is adequate for clinical feature extraction.
    sqi_metrics : dict
        Calculated SQI values.
    """
    ssqi = compute_skewness_sqi(signal)
    ksqi = compute_kurtosis_sqi(signal)
    psqi = compute_power_sqi(signal, fs)
    pi = compute_perfusion_index(signal)

    # Heuristic clinical acceptability check
    # 1. Sufficient cardiac band power
    # 2. Kurtosis not flat/degenerate
    # 3. Non-zero variance
    is_acceptable = (psqi >= min_power_sqi) and (ksqi >= min_kurtosis) and (np.std(signal) > 1e-4)

    metrics = {
        "sSQI": round(ssqi, 4),
        "kSQI": round(ksqi, 4),
        "pSQI": round(psqi, 4),
        "perfusion_index": round(pi, 4),
        "is_acceptable": float(is_acceptable),
    }

    return bool(is_acceptable), metrics
