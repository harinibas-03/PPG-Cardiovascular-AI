"""
PPG Derivative Feature Extraction: VPG & APG
============================================
Computes:
1. Velocity Plethysmogram (VPG - 1st derivative): maximum velocity slope.
2. Acceleration Plethysmogram (APG - 2nd derivative):
   - Fiducial points (a, b, c, d, e waves)
   - Vascular aging & stiffness indices: b/a, c/a, d/a, e/a, and Aging Index (b - c - d - e)/a.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import signal as sp_signal


def compute_derivatives(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1st derivative (VPG) and 2nd derivative (APG) with zero-phase smoothing
    to suppress numerical differentiation noise.
    """
    dt = 1.0 / fs
    # 1st derivative (VPG)
    vpg = np.gradient(signal, dt)

    # Low-pass filter VPG gently before 2nd derivative if fs allows
    nyquist = 0.5 * fs
    cut = min(12.0, nyquist * 0.9)
    if cut > 1.0:
        b, a = sp_signal.butter(2, cut / nyquist, btype="low")
        padlen = min(len(vpg) - 1, 6)
        vpg_smooth = sp_signal.filtfilt(b, a, vpg, padlen=padlen) if padlen > 0 else vpg
    else:
        vpg_smooth = vpg

    # 2nd derivative (APG)
    apg = np.gradient(vpg_smooth, dt)
    return vpg_smooth.astype(np.float32), apg.astype(np.float32)


def extract_single_beat_apg(beat: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract classical APG fiducial waves (a, b, c, d, e) and arterial stiffness ratios.

    Fiducial wave definitions in early beat cycle:
    - a: initial positive peak (systolic acceleration)
    - b: early negative trough (systolic deceleration)
    - c: re-acceleration local peak
    - d: late systolic deceleration trough
    - e: early diastolic peak
    """
    if len(beat) < int(fs * 0.3):
        return {}

    vpg, apg = compute_derivatives(beat, fs)

    # Focus on the first 65% of the beat cycle where systolic APG complexes occur
    systolic_window_len = int(len(beat) * 0.65)
    apg_sys = apg[:systolic_window_len]

    # Find peaks and valleys in APG
    peaks, _ = sp_signal.find_peaks(apg_sys)
    valleys, _ = sp_signal.find_peaks(-apg_sys)

    if len(peaks) == 0:
        return {}

    # Wave 'a': the initial prominent positive peak
    a_idx = peaks[0]
    a_amp = float(apg_sys[a_idx])

    if a_amp <= 1e-6:
        # Fallback to max value
        a_idx = int(np.argmax(apg_sys))
        a_amp = float(apg_sys[a_idx])
        if a_amp <= 1e-6:
            return {}

    # Wave 'b': lowest trough immediately following 'a'
    valleys_after_a = valleys[valleys > a_idx]
    if len(valleys_after_a) > 0:
        b_idx = valleys_after_a[0]
        b_amp = float(apg_sys[b_idx])
    else:
        b_amp = float(np.min(apg_sys[a_idx:])) if a_idx < len(apg_sys) - 1 else 0.0
        b_idx = a_idx + 1

    # Wave 'c': next peak following 'b'
    peaks_after_b = peaks[peaks > b_idx]
    if len(peaks_after_b) > 0:
        c_idx = peaks_after_b[0]
        c_amp = float(apg_sys[c_idx])
    else:
        c_amp = 0.0
        c_idx = b_idx + 1

    # Wave 'd': trough following 'c'
    valleys_after_c = valleys[valleys > c_idx]
    if len(valleys_after_c) > 0:
        d_idx = valleys_after_c[0]
        d_amp = float(apg_sys[d_idx])
    else:
        d_amp = 0.0
        d_idx = c_idx + 1

    # Wave 'e': positive peak following 'd'
    peaks_after_d = peaks[peaks > d_idx]
    if len(peaks_after_d) > 0:
        e_idx = peaks_after_d[0]
        e_amp = float(apg_sys[e_idx])
    else:
        e_amp = 0.0

    # Clinical ratios normalized by 'a'
    b_div_a = b_amp / (a_amp + 1e-6)
    c_div_a = c_amp / (a_amp + 1e-6)
    d_div_a = d_amp / (a_amp + 1e-6)
    e_div_a = e_amp / (a_amp + 1e-6)

    # Vascular Aging Index: (b - c - d - e) / a
    aging_index = (b_amp - c_amp - d_amp - e_amp) / (a_amp + 1e-6)

    # Maximum velocity from VPG
    vpg_max = float(np.max(vpg))
    vpg_min = float(np.min(vpg))

    return {
        "apg_a_amp": round(a_amp, 5),
        "apg_b_amp": round(b_amp, 5),
        "apg_c_amp": round(c_amp, 5),
        "apg_d_amp": round(d_amp, 5),
        "apg_e_amp": round(e_amp, 5),
        "apg_b_a_ratio": round(float(b_div_a), 5),
        "apg_c_a_ratio": round(float(c_div_a), 5),
        "apg_d_a_ratio": round(float(d_div_a), 5),
        "apg_e_a_ratio": round(float(e_div_a), 5),
        "apg_aging_index": round(float(aging_index), 5),
        "vpg_max_slope": round(vpg_max, 5),
        "vpg_min_slope": round(vpg_min, 5),
    }


def extract_derivative_features(beats: List[Dict[str, Any]], fs: float) -> Dict[str, float]:
    """
    Extract aggregated APG and VPG features across all beats.
    """
    if not beats:
        return {
            "mean_apg_b_a_ratio": 0.0,
            "mean_apg_c_a_ratio": 0.0,
            "mean_apg_d_a_ratio": 0.0,
            "mean_apg_e_a_ratio": 0.0,
            "mean_apg_aging_index": 0.0,
            "mean_vpg_max_slope": 0.0,
        }

    beat_metrics = []
    for b in beats:
        m = extract_single_beat_apg(b["beat"], fs)
        if m:
            beat_metrics.append(m)

    if not beat_metrics:
        return {
            "mean_apg_b_a_ratio": 0.0,
            "mean_apg_c_a_ratio": 0.0,
            "mean_apg_d_a_ratio": 0.0,
            "mean_apg_e_a_ratio": 0.0,
            "mean_apg_aging_index": 0.0,
            "mean_vpg_max_slope": 0.0,
        }

    keys = beat_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [bm[k] for bm in beat_metrics if k in bm and not np.isnan(bm[k])]
        agg[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
        agg[f"std_{k}"] = float(np.std(vals)) if vals else 0.0

    return agg
