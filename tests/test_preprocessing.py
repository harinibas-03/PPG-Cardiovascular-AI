"""
Unit tests for PPG preprocessing: filtering, normalization, segmentation, and SQI.
"""

import pytest
import numpy as np
from arohan_ppg_cvd.preprocessing.filters import (
    bandpass_filter,
    remove_baseline_wander,
    notch_filter,
    clean_ppg_signal,
)
from arohan_ppg_cvd.preprocessing.normalization import normalize_signal
from arohan_ppg_cvd.preprocessing.segmentation import (
    detect_systolic_peaks,
    detect_pulse_valleys,
    segment_pulse_waves,
    segment_fixed_windows,
)
from arohan_ppg_cvd.preprocessing.sqi import (
    compute_skewness_sqi,
    compute_kurtosis_sqi,
    compute_power_sqi,
    compute_perfusion_index,
    evaluate_signal_quality,
)


@pytest.fixture
def synthetic_ppg():
    """Generates 5 seconds of synthetic PPG at 30 Hz."""
    fs = 30.0
    t = np.linspace(0, 5.0, int(fs * 5.0))
    beat_freq = 1.2  # 72 bpm
    # Pulse waveform with systolic and dicrotic components
    phase = (t * beat_freq) % 1.0
    sig = np.exp(-((phase - 0.20) ** 2) / 0.008) + 0.35 * np.exp(-((phase - 0.42) ** 2) / 0.015)
    wander = 0.2 * np.sin(2 * np.pi * 0.2 * t)
    return (sig + wander).astype(np.float32), fs


def test_bandpass_filter(synthetic_ppg):
    sig, fs = synthetic_ppg
    filtered = bandpass_filter(sig, fs=fs, lowcut=0.5, highcut=8.0)
    assert len(filtered) == len(sig)
    assert not np.any(np.isnan(filtered))
    # Baseline wander (mean offset) should be attenuated
    assert np.abs(np.mean(filtered)) < np.abs(np.mean(sig)) + 0.1


def test_remove_baseline_wander(synthetic_ppg):
    sig, fs = synthetic_ppg
    detrended = remove_baseline_wander(sig, fs=fs, cutoff=0.5)
    assert len(detrended) == len(sig)
    assert not np.any(np.isnan(detrended))


def test_notch_filter():
    fs = 150.0  # High enough to notch 50 Hz
    t = np.linspace(0, 1.0, int(fs))
    clean = np.sin(2 * np.pi * 1.5 * t)
    noise = 0.5 * np.sin(2 * np.pi * 50.0 * t)
    sig = clean + noise
    notched = notch_filter(sig, fs=fs, freq=50.0)
    assert len(notched) == len(sig)


def test_normalize_signal():
    arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)

    # Z-score: mean 0, std 1
    z = normalize_signal(arr, method="z_score")
    assert np.isclose(np.mean(z), 0.0, atol=1e-5)
    assert np.isclose(np.std(z), 1.0, atol=1e-5)

    # Min-max: [0, 1]
    mm = normalize_signal(arr, method="min_max")
    assert np.isclose(np.min(mm), 0.0, atol=1e-5)
    assert np.isclose(np.max(mm), 1.0, atol=1e-5)

    # Robust scaling
    rob = normalize_signal(arr, method="robust")
    assert len(rob) == len(arr)


def test_detect_systolic_peaks(synthetic_ppg):
    sig, fs = synthetic_ppg
    clean = clean_ppg_signal(sig, fs=fs)
    peaks, _ = detect_systolic_peaks(clean, fs=fs, min_bpm=40.0, max_bpm=180.0)
    # 5 seconds at 1.2 Hz should have ~5-6 peaks
    assert 4 <= len(peaks) <= 8


def test_segment_pulse_waves(synthetic_ppg):
    sig, fs = synthetic_ppg
    clean = clean_ppg_signal(sig, fs=fs)
    beats = segment_pulse_waves(clean, fs=fs)
    assert len(beats) >= 2
    for b in beats:
        assert "beat" in b
        assert b["duration_sec"] > 0.3
        assert b["peak_idx"] > b["start_idx"]


def test_segment_fixed_windows(synthetic_ppg):
    sig, fs = synthetic_ppg
    windows = segment_fixed_windows(sig, fs=fs, window_sec=2.0, overlap_sec=1.0)
    assert len(windows) >= 3
    for start, end, w in windows:
        assert len(w) == int(2.0 * fs)


def test_sqi_metrics(synthetic_ppg):
    sig, fs = synthetic_ppg
    clean = clean_ppg_signal(sig, fs=fs)
    norm = normalize_signal(clean, method="z_score")

    ssqi = compute_skewness_sqi(norm)
    ksqi = compute_kurtosis_sqi(norm)
    psqi = compute_power_sqi(norm, fs=fs)
    pi = compute_perfusion_index(sig)

    assert isinstance(ssqi, float)
    assert isinstance(ksqi, float)
    assert 0.0 <= psqi <= 1.0
    assert pi >= 0.0

    is_acc, details = evaluate_signal_quality(norm, fs=fs)
    assert isinstance(is_acc, bool)
    assert "sSQI" in details
    assert "kSQI" in details
