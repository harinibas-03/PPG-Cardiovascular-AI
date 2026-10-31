"""
Unit tests for PPG feature extraction (waveform morphology, APG/VPG derivatives, HRV, unified extractor).
"""

import pytest
import numpy as np
from arohan_ppg_cvd.preprocessing.filters import clean_ppg_signal
from arohan_ppg_cvd.preprocessing.segmentation import detect_systolic_peaks, segment_pulse_waves
from arohan_ppg_cvd.features.waveform import extract_single_beat_morphology, extract_waveform_features
from arohan_ppg_cvd.features.derivatives import compute_derivatives, extract_single_beat_apg, extract_derivative_features
from arohan_ppg_cvd.features.hrv_stats import extract_hrv_features, extract_statistical_and_spectral_features
from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor


@pytest.fixture
def clean_ppg_and_beats():
    fs = 30.0
    t = np.linspace(0, 8.0, int(fs * 8.0))
    beat_freq = 1.2  # 72 bpm
    phase = (t * beat_freq) % 1.0
    sig = np.exp(-((phase - 0.20) ** 2) / 0.008) + 0.35 * np.exp(-((phase - 0.42) ** 2) / 0.015)
    clean = clean_ppg_signal(sig, fs=fs)
    peaks, _ = detect_systolic_peaks(clean, fs=fs)
    beats = segment_pulse_waves(clean, fs=fs)
    return clean, peaks, beats, fs


def test_waveform_single_beat(clean_ppg_and_beats):
    _, _, beats, fs = clean_ppg_and_beats
    assert len(beats) > 0
    morph = extract_single_beat_morphology(beats[0]["beat"], fs=fs)
    assert "systolic_amplitude" in morph
    assert "crest_time_ratio" in morph
    assert "pw50" in morph
    assert morph["systolic_amplitude"] > 0
    assert 0.0 < morph["crest_time_ratio"] < 1.0


def test_waveform_aggregated(clean_ppg_and_beats):
    _, _, beats, fs = clean_ppg_and_beats
    agg = extract_waveform_features(beats, fs=fs)
    assert "mean_systolic_amplitude" in agg
    assert "mean_crest_time_ratio" in agg
    assert "mean_pw50" in agg


def test_derivatives_apg(clean_ppg_and_beats):
    clean, _, beats, fs = clean_ppg_and_beats
    vpg, apg = compute_derivatives(clean, fs=fs)
    assert len(vpg) == len(clean)
    assert len(apg) == len(clean)

    apg_metrics = extract_single_beat_apg(beats[0]["beat"], fs=fs)
    assert "apg_a_amp" in apg_metrics
    assert "apg_aging_index" in apg_metrics

    agg_deriv = extract_derivative_features(beats, fs=fs)
    assert "mean_apg_aging_index" in agg_deriv


def test_hrv_features(clean_ppg_and_beats):
    _, peaks, _, fs = clean_ppg_and_beats
    hrv = extract_hrv_features(peaks, fs=fs)
    assert "hr_bpm" in hrv
    assert 60.0 <= hrv["hr_bpm"] <= 85.0
    assert "mean_ppi_ms" in hrv
    assert "sdnn_ms" in hrv
    assert "rmssd_ms" in hrv


def test_statistical_and_spectral(clean_ppg_and_beats):
    clean, _, _, fs = clean_ppg_and_beats
    stats = extract_statistical_and_spectral_features(clean, fs=fs)
    assert "stat_mean" in stats
    assert "stat_skewness" in stats
    assert "stat_kurtosis" in stats
    assert "spec_cardiac_band_power" in stats


def test_unified_feature_extractor(clean_ppg_and_beats):
    clean, _, _, fs = clean_ppg_and_beats
    extractor = PPGFeatureExtractor(default_fs=fs)
    feat = extractor.extract_from_signal(clean, fs=fs, file_id="patient_001", bpm_reference=72.0)

    assert feat is not None
    assert feat["file_id"] == "patient_001"
    assert feat["device_bpm_ref"] == 72.0
    assert "hr_bpm" in feat
    assert "mean_apg_aging_index" in feat
    assert "sSQI" in feat
    assert "mean_crest_time_ratio" in feat
