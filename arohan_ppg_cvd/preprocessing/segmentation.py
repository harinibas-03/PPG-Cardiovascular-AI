"""
PPG Pulse Segmentation and Peak Detection
=========================================
Physiologically constrained peak detection, individual pulse-beat extraction,
and fixed-window sliding segmentation for continuous PPG recordings.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import signal as sp_signal


def detect_systolic_peaks(
    signal: np.ndarray,
    fs: float,
    min_bpm: float = 35.0,
    max_bpm: float = 210.0,
    prominence_ratio: float = 0.25,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect systolic peaks in a conditioned PPG signal.

    Parameters
    ----------
    signal : np.ndarray
        1D clean/normalized PPG signal.
    fs : float
        Sampling rate in Hz.
    min_bpm : float
        Minimum physiological heart rate (default 35 bpm).
    max_bpm : float
        Maximum physiological heart rate (default 210 bpm).
    prominence_ratio : float
        Fraction of peak-to-peak amplitude required for peak prominence.

    Returns
    -------
    peaks : np.ndarray
        Sample indices of detected systolic peaks.
    properties : dict
        Peak properties dictionary from scipy.signal.find_peaks.
    """
    sig = np.asarray(signal, dtype=np.float32)
    if sig.size < int(fs):
        return np.array([], dtype=np.int64), {}

    # Distance constraint: at max_bpm, peak interval in seconds is 60 / max_bpm
    min_distance = max(1, int(fs * (60.0 / max_bpm)))

    # Prominence threshold:
    # Use robust amplitude estimation (10th to 90th percentile) to avoid being distorted
    # by initial sensor settling transients, motion spikes, or DC offset steps.
    p10, p90 = np.percentile(sig, [10, 90])
    robust_range = float(p90 - p10)
    if robust_range < 1e-4:
        robust_range = float(np.ptp(sig))

    prominence = max(0.01, robust_range * prominence_ratio)

    peaks, properties = sp_signal.find_peaks(
        sig,
        distance=min_distance,
        prominence=prominence,
    )

    return peaks, properties


def detect_pulse_valleys(
    signal: np.ndarray,
    peaks: np.ndarray,
) -> np.ndarray:
    """
    Detect diastolic pulse onsets (valleys/troughs) located between consecutive systolic peaks.

    Parameters
    ----------
    signal : np.ndarray
        1D PPG signal.
    peaks : np.ndarray
        Indices of detected systolic peaks.

    Returns
    -------
    valleys : np.ndarray
        Indices of valleys.
    """
    if len(peaks) < 2:
        return np.array([], dtype=np.int64)

    valleys = []
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        if end > start + 1:
            segment = signal[start:end]
            valley_idx = start + np.argmin(segment)
            valleys.append(valley_idx)

    return np.asarray(valleys, dtype=np.int64)


def segment_pulse_waves(
    signal: np.ndarray,
    fs: float,
    min_bpm: float = 35.0,
    max_bpm: float = 210.0,
) -> List[Dict[str, Any]]:
    """
    Segment PPG into individual single-beat pulse waves (from pulse onset to pulse onset).

    Returns
    -------
    List of beat dictionaries:
        - 'beat': np.ndarray of beat samples
        - 'start_idx': int
        - 'peak_idx': int
        - 'end_idx': int
        - 'duration_sec': float
    """
    peaks, _ = detect_systolic_peaks(signal, fs, min_bpm=min_bpm, max_bpm=max_bpm)
    if len(peaks) < 3:
        return []

    valleys = detect_pulse_valleys(signal, peaks)
    if len(valleys) < 2:
        return []

    min_samples = max(1, int(fs * (60.0 / max_bpm)))
    max_samples = max(min_samples + 1, int(fs * (60.0 / min_bpm)))

    beats = []
    for i in range(len(valleys) - 1):
        v_start = valleys[i]
        v_end = valleys[i + 1]

        # Find peak between these valleys
        candidates = peaks[(peaks > v_start) & (peaks < v_end)]
        if len(candidates) == 1:
            peak = candidates[0]
            beat_samples = signal[v_start:v_end]
            if min_samples <= len(beat_samples) <= max_samples:
                beats.append({
                    "beat": beat_samples,
                    "start_idx": int(v_start),
                    "peak_idx": int(peak),
                    "end_idx": int(v_end),
                    "duration_sec": float((v_end - v_start) / fs),
                })

    return beats


def segment_fixed_windows(
    signal: np.ndarray,
    fs: float,
    window_sec: float = 5.0,
    overlap_sec: float = 2.5,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Segment signal into fixed-length sliding windows for batch feature extraction.

    Parameters
    ----------
    signal : np.ndarray
        1D PPG signal.
    fs : float
        Sampling rate in Hz.
    window_sec : float
        Window length in seconds (default 5.0s, standard for short-term PPG analysis).
    overlap_sec : float
        Overlap duration in seconds (default 2.5s, 50% overlap).

    Returns
    -------
    List of tuples: (start_sample, end_sample, window_samples)
    """
    win_len = int(window_sec * fs)
    step_len = max(1, int((window_sec - overlap_sec) * fs))

    windows = []
    n_samples = len(signal)
    if n_samples < win_len:
        if n_samples > int(fs * 2.0):  # At least 2 seconds
            windows.append((0, n_samples, signal))
        return windows

    for start in range(0, n_samples - win_len + 1, step_len):
        end = start + win_len
        windows.append((start, end, signal[start:end]))

    return windows
