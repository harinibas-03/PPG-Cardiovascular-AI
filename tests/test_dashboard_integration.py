"""
Integration test suite for the Arohan PPG-CVD Dashboard logic and pipeline components.
Verifies all 4 dashboard tabs, real Spandan data, uploads, model loading, and supervised integration.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from arohan_ppg_cvd.preprocessing.filters import clean_ppg_signal
from arohan_ppg_cvd.preprocessing.normalization import normalize_signal
from arohan_ppg_cvd.preprocessing.segmentation import detect_systolic_peaks, segment_pulse_waves
from arohan_ppg_cvd.preprocessing.sqi import evaluate_signal_quality
from arohan_ppg_cvd.features.derivatives import compute_derivatives
from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor
from arohan_ppg_cvd.inference.predictor import ArohanPredictor
from arohan_ppg_cvd.adapters.upload_parser import parse_and_validate_upload, UploadValidationError


@pytest.fixture
def real_spandan_sample():
    assert os.path.exists("data/sample_ppg.csv")
    df = pd.read_csv("data/sample_ppg.csv")
    sig = df["signal"].to_numpy(dtype=np.float32)
    fs = float(df["sampling_rate"].iloc[0])
    fid = str(df["file_id"].iloc[0])
    return sig, fs, fid


def test_tab1_signal_conditioning_and_peaks(real_spandan_sample):
    """Tab 1: Conditioned signal, peak detection, and SQI metrics on real Spandan data."""
    sig, fs, fid = real_spandan_sample
    clean = clean_ppg_signal(sig, fs=fs, lowcut=0.5, highcut=8.0)
    norm = normalize_signal(clean, method="z_score")
    is_acc, sqi = evaluate_signal_quality(norm, fs=fs)
    peaks, _ = detect_systolic_peaks(norm, fs=fs)

    assert len(clean) == len(sig)
    assert len(norm) == len(sig)
    # Systolic peaks must be distributed across the recording (> 20 peaks for 60s at ~62 bpm)
    assert len(peaks) >= 20
    # Peak detection must not be concentrated only in the first 2 seconds
    late_peaks = peaks[peaks > int(fs * 5.0)]
    assert len(late_peaks) > 15
    # SQI indices must be present
    assert "sSQI" in sqi and "kSQI" in sqi and "pSQI" in sqi and "perfusion_index" in sqi


def test_tab2_morphology_and_derivatives(real_spandan_sample):
    """Tab 2: Pulse wave segmentation, average beat profile, and VPG/APG derivatives."""
    sig, fs, fid = real_spandan_sample
    clean = clean_ppg_signal(sig, fs=fs)
    norm = normalize_signal(clean, method="z_score")
    beats = segment_pulse_waves(norm, fs=fs)
    vpg, apg = compute_derivatives(norm, fs)

    # Valid segmented cardiac beats
    assert len(beats) >= 15
    for b in beats:
        assert len(b["beat"]) > 0
        assert 0.25 <= b["duration_sec"] <= 2.0

    # Derivatives
    assert len(vpg) == len(norm)
    assert len(apg) == len(norm)
    assert not np.any(np.isnan(vpg))
    assert not np.any(np.isnan(apg))


def test_tab3_feature_vector_extraction(real_spandan_sample):
    """Tab 3: Feature vector computation, matching schema, and exportable DataFrame."""
    sig, fs, fid = real_spandan_sample
    extractor = PPGFeatureExtractor(default_fs=fs)
    features = extractor.extract_from_signal(sig, fs=fs, file_id=fid)

    assert features is not None
    assert features["file_id"] == fid
    assert features["detected_peaks_count"] >= 20
    assert features["segmented_beats_count"] >= 15
    # Heart rate must be realistic (50 to 90 bpm for this patient)
    assert 50.0 <= features["hr_bpm"] <= 90.0

    df_feat = pd.DataFrame([features])
    csv_bytes = df_feat.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 200


def test_tab4_kmeans_model_loading_and_inference(real_spandan_sample):
    """Tab 4: Loading models/unsupervised_kmeans.joblib and predicting cluster assignment."""
    model_path = "models/unsupervised_kmeans.joblib"
    assert os.path.exists(model_path)

    predictor = ArohanPredictor(model_path)
    assert predictor.model is not None
    assert predictor.model.is_unsupervised_fitted is True

    sig, fs, fid = real_spandan_sample
    extractor = PPGFeatureExtractor(default_fs=fs)
    features = extractor.extract_from_signal(sig, fs=fs, file_id=fid)
    df_feat = pd.DataFrame([features])

    cluster_id = int(predictor.model.clusterer.predict_clusters(df_feat)[0])
    assert 0 <= cluster_id < predictor.model.clusterer.n_clusters

    dists = predictor.model.clusterer.transform_to_cluster_distances(df_feat)[0]
    assert len(dists) == predictor.model.clusterer.n_clusters
    assert dists[cluster_id] >= 0.0


def test_tab4_supervised_model_safety_boundary():
    """Tab 4: Ensures supervised CAD/CVD prediction reports unavailable when not trained."""
    predictor = ArohanPredictor("models/unsupervised_kmeans.joblib")
    assert predictor.is_ready_for_clinical_inference is False

    # Calling analyze_signal safely reports unavailable rather than fabricating predictions
    sig = np.sin(np.linspace(0, 20, 300)).astype(np.float32) + 1.0
    res = predictor.analyze_signal(sig, fs=30.0)
    pred_info = res["clinical_prediction"]
    assert pred_info["is_available"] is False
    assert "supervised model" in pred_info["message"].lower() or "not been loaded" in pred_info["message"].lower()


def test_upload_and_pipeline_with_real_spandan_subset():
    """Tests loading a JSON record directly from spandan_subset.json as if uploaded by user."""
    raw_path = "data/raw/spandan_subset.json"
    assert os.path.exists(raw_path)
    with open(raw_path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    # Test with first 3 records in dataset
    for r in recs[:3]:
        raw_bytes = json.dumps(r).encode("utf-8")
        parsed = parse_and_validate_upload(raw_bytes, r.get("file_id", "test.json"))

        assert parsed["fs_auto_detected"] is True
        assert parsed["duration_sec"] > 10.0
        assert len(parsed["signal"]) > 200

        extractor = PPGFeatureExtractor(default_fs=parsed["sampling_rate"])
        feats = extractor.extract_from_signal(parsed["signal"], fs=parsed["sampling_rate"])
        assert feats is not None
        assert feats["detected_peaks_count"] > 10

