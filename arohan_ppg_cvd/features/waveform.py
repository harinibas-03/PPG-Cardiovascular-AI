"""
PPG Waveform Morphology Feature Extraction
==========================================
Extracts clinical pulse wave morphology parameters:
- Systolic peak, dicrotic notch, and diastolic peak
- Systolic rise time (crest time) and total beat duration
- Pulse width at 50% amplitude (PW50)
- Augmentation Index (AI) and Reflection Index (RI)
- Pulse area partitioning (systolic vs diastolic area ratio)
"""

from typing import Dict, Any, List, Optional
import numpy as np
from scipy import signal as sp_signal


def extract_single_beat_morphology(beat: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Extract morphological landmarks from a single segmented pulse wave beat.
    The beat is expected to start at pulse onset (valley) and end at the next pulse onset.

    Parameters
    ----------
    beat : np.ndarray
        1D array of single-beat PPG values.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict: Morphological features.
    """
    n = len(beat)
    if n < int(fs * 0.25):  # Shorter than 250ms is unphysiological
        return {}

    t_sec = np.arange(n) / fs
    total_time = float(t_sec[-1])

    # Baseline adjustment: subtract minimum so pulse starts at 0
    beat_base = beat - np.min(beat)
    systolic_idx = int(np.argmax(beat_base))
    systolic_amp = float(beat_base[systolic_idx])
    systolic_time = float(t_sec[systolic_idx])

    if systolic_amp <= 1e-6 or systolic_idx <= 1 or systolic_idx >= n - 2:
        return {}

    # Crest time (systolic rise time ratio)
    crest_time_ratio = systolic_time / total_time if total_time > 0 else 0.0

    # Pulse Width at 50% amplitude (PW50)
    half_amp = 0.5 * systolic_amp
    left_candidates = np.where(beat_base[:systolic_idx] <= half_amp)[0]
    right_candidates = np.where(beat_base[systolic_idx:] <= half_amp)[0]

    if len(left_candidates) > 0 and len(right_candidates) > 0:
        t_left = t_sec[left_candidates[-1]]
        t_right = t_sec[systolic_idx + right_candidates[0]]
        pw50 = float(t_right - t_left)
    else:
        pw50 = float(systolic_time)

    # Search for dicrotic notch & diastolic peak in downstroke
    downstroke = beat_base[systolic_idx:]
    down_t = t_sec[systolic_idx:]

    notch_amp = 0.0
    notch_time = 0.0
    diastolic_amp = 0.0
    diastolic_time = 0.0

    if len(downstroke) > 4:
        # First derivative of downstroke
        d1 = np.gradient(downstroke, 1.0 / fs)
        # In a classical PPG, dicrotic notch corresponds to a local minimum of downstroke
        # or inflection point where d1 reaches a local maximum
        valleys_in_down, _ = sp_signal.find_peaks(-downstroke)
        peaks_in_down, _ = sp_signal.find_peaks(downstroke)

        if len(valleys_in_down) > 0 and len(peaks_in_down) > 0 and peaks_in_down[0] > valleys_in_down[0]:
            notch_idx = valleys_in_down[0]
            dia_idx = peaks_in_down[0]
            notch_amp = float(downstroke[notch_idx])
            notch_time = float(down_t[notch_idx])
            diastolic_amp = float(downstroke[dia_idx])
            diastolic_time = float(down_t[dia_idx])
        else:
            # Look at inflection points (zero crossings of d2 or peaks of d1)
            d2 = np.gradient(d1, 1.0 / fs)
            d2_peaks, _ = sp_signal.find_peaks(d2)
            if len(d2_peaks) > 0:
                inflection_idx = d2_peaks[0]
                notch_amp = float(downstroke[inflection_idx])
                notch_time = float(down_t[inflection_idx])
                diastolic_amp = notch_amp * 1.05
                diastolic_time = notch_time + (0.05)

    # Augmentation Index (AI = (Diastolic Amp - Systolic Amp) / Systolic Amp)
    # Reflection Index (RI = Diastolic Amp / Systolic Amp)
    if diastolic_amp > 0 and systolic_amp > 0:
        augmentation_index = float((diastolic_amp - systolic_amp) / systolic_amp)
        reflection_index = float(diastolic_amp / systolic_amp)
    else:
        augmentation_index = 0.0
        reflection_index = 0.0

    # Pulse area ratio: Area under systolic phase vs diastolic phase
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    systolic_area = float(trapz_fn(beat_base[:systolic_idx], dx=1.0 / fs))
    diastolic_area = float(trapz_fn(beat_base[systolic_idx:], dx=1.0 / fs))
    area_ratio = float(systolic_area / (diastolic_area + 1e-6))

    return {
        "systolic_amplitude": round(systolic_amp, 5),
        "systolic_time": round(systolic_time, 5),
        "pulse_duration": round(total_time, 5),
        "crest_time_ratio": round(crest_time_ratio, 5),
        "pw50": round(pw50, 5),
        "notch_amplitude": round(notch_amp, 5),
        "notch_time": round(notch_time, 5),
        "diastolic_amplitude": round(diastolic_amp, 5),
        "diastolic_time": round(diastolic_time, 5),
        "augmentation_index": round(augmentation_index, 5),
        "reflection_index": round(reflection_index, 5),
        "systolic_diastolic_area_ratio": round(area_ratio, 5),
    }


def extract_waveform_features(beats: List[Dict[str, Any]], fs: float) -> Dict[str, float]:
    """
    Extract aggregated morphological features across all valid segmented beats.
    """
    if not beats:
        return {
            "mean_systolic_amplitude": 0.0,
            "mean_pulse_duration": 0.0,
            "mean_crest_time_ratio": 0.0,
            "mean_pw50": 0.0,
            "mean_augmentation_index": 0.0,
            "mean_reflection_index": 0.0,
            "mean_area_ratio": 0.0,
        }

    beat_metrics = []
    for b in beats:
        m = extract_single_beat_morphology(b["beat"], fs)
        if m:
            beat_metrics.append(m)

    if not beat_metrics:
        return {
            "mean_systolic_amplitude": 0.0,
            "mean_pulse_duration": 0.0,
            "mean_crest_time_ratio": 0.0,
            "mean_pw50": 0.0,
            "mean_augmentation_index": 0.0,
            "mean_reflection_index": 0.0,
            "mean_area_ratio": 0.0,
        }

    keys = beat_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [bm[k] for bm in beat_metrics if k in bm and not np.isnan(bm[k])]
        agg[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
        agg[f"std_{k}"] = float(np.std(vals)) if vals else 0.0

    return agg
