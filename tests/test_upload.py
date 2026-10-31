"""
Unit tests for upload parsing, validation, and pipeline integration.
"""

import io
import json
import pytest
import numpy as np
import pandas as pd
import joblib

from arohan_ppg_cvd.adapters.upload_parser import parse_and_validate_upload, UploadValidationError
from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor
from arohan_ppg_cvd.inference.predictor import ArohanPredictor


def _make_clean_ppg(fs: float = 30.0, duration: float = 5.0, bpm: float = 72.0) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration))
    freq = bpm / 60.0
    phase = (t * freq) % 1.0
    pulse = np.exp(-((phase - 0.20) ** 2) / 0.008) + 0.35 * np.exp(-((phase - 0.42) ** 2) / 0.015)
    return pulse.astype(np.float32)


def test_upload_csv_standard():
    sig = _make_clean_ppg(fs=30.0, duration=4.0)
    df = pd.DataFrame({
        "signal": sig,
        "sampling_rate": [30.0] * len(sig),
        "file_id": ["rec_001.csv"] * len(sig),
    })
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    res = parse_and_validate_upload(csv_bytes, "rec_001.csv")
    assert len(res["signal"]) == len(sig)
    assert np.isclose(res["sampling_rate"], 30.0)
    assert res["fs_auto_detected"] is True
    assert res["file_id"] == "rec_001.csv"
    assert res["duration_sec"] >= 3.9


def test_upload_csv_with_timestamps():
    fs = 50.0
    sig = _make_clean_ppg(fs=fs, duration=4.0)
    t = np.arange(len(sig)) / fs
    df = pd.DataFrame({"t": t, "ppg": sig})
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    res = parse_and_validate_upload(csv_bytes, "wearable_ppg.csv", manual_fs=30.0)
    assert np.isclose(res["sampling_rate"], 50.0, atol=0.5)
    assert res["fs_auto_detected"] is True
    assert len(res["signal"]) == len(sig)


def test_upload_csv_single_column_manual_fs():
    sig = _make_clean_ppg(fs=25.0, duration=3.0)
    df = pd.DataFrame({"amplitude": sig})
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    res = parse_and_validate_upload(csv_bytes, "single_col.csv", manual_fs=25.0)
    assert res["sampling_rate"] == 25.0
    assert res["fs_auto_detected"] is False
    assert len(res["signal"]) == len(sig)


def test_upload_json_spandan_subset():
    sig = _make_clean_ppg(fs=30.0, duration=5.0)
    record = {
        "file_id": "spandan_sub_123",
        "sampling_rate": 30.0,
        "bpm": 72.0,
        "signal": sig.tolist(),
    }
    json_bytes = json.dumps(record).encode("utf-8")

    res = parse_and_validate_upload(json_bytes, "spandan_sub_123.json")
    assert res["fs_auto_detected"] is True
    assert res["sampling_rate"] == 30.0
    assert res["file_id"] == "spandan_sub_123"
    assert res["device_bpm"] == 72.0
    assert len(res["signal"]) == len(sig)


def test_upload_json_raw_spandan_format():
    fs = 30.0
    n = 90
    t_ms = (np.arange(n) / fs) * 1000.0
    sig = _make_clean_ppg(fs=fs, duration=3.0)
    ppg_dicts = [
        {"r": float(s * 0.8), "g": float(s), "b": float(s * 0.5), "t": float(ti)}
        for s, ti in zip(sig, t_ms)
    ]
    record = {
        "file_id": "spandan_raw_456",
        "bpm": 75.0,
        "ppg": ppg_dicts,
    }
    json_bytes = json.dumps(record).encode("utf-8")

    res = parse_and_validate_upload(json_bytes, "spandan_raw_456.json")
    assert np.isclose(res["sampling_rate"], 30.0, atol=1.0)
    assert res["fs_auto_detected"] is True
    assert res["file_id"] == "spandan_raw_456"
    assert len(res["signal"]) == n


def test_upload_empty_file():
    with pytest.raises(UploadValidationError, match="empty"):
        parse_and_validate_upload(b"", "empty.csv")


def test_upload_unsupported_extension():
    with pytest.raises(UploadValidationError, match="Unsupported file format"):
        parse_and_validate_upload(b"some,data\n1,2", "file.pdf")


def test_upload_insufficient_duration():
    sig = _make_clean_ppg(fs=30.0, duration=1.0)  # Only 1.0s, minimum is 2.0s
    csv_bytes = pd.DataFrame({"signal": sig}).to_csv(index=False).encode("utf-8")
    with pytest.raises(UploadValidationError, match="duration is too short"):
        parse_and_validate_upload(csv_bytes, "short.csv", manual_fs=30.0)


def test_upload_flatline_signal():
    flat_sig = np.full(100, 42.0, dtype=np.float32)  # constant zero variance
    csv_bytes = pd.DataFrame({"signal": flat_sig}).to_csv(index=False).encode("utf-8")
    with pytest.raises(UploadValidationError, match="flatlined or has near-zero amplitude variance"):
        parse_and_validate_upload(csv_bytes, "flatline.csv", manual_fs=30.0)


def test_upload_all_nan():
    nan_sig = np.full(100, np.nan, dtype=np.float32)
    csv_bytes = pd.DataFrame({"signal": nan_sig}).to_csv(index=False).encode("utf-8")
    with pytest.raises(UploadValidationError, match="only NaN or Inf"):
        parse_and_validate_upload(csv_bytes, "all_nan.csv", manual_fs=30.0)


def test_upload_excessive_nan():
    sig = _make_clean_ppg(fs=30.0, duration=4.0)
    sig[: int(len(sig) * 0.25)] = np.nan  # 25% NaNs (> 10% limit)
    csv_bytes = pd.DataFrame({"signal": sig}).to_csv(index=False).encode("utf-8")
    with pytest.raises(UploadValidationError, match="excessive missing"):
        parse_and_validate_upload(csv_bytes, "excessive_nan.csv", manual_fs=30.0)


def test_upload_minor_nan_interpolation():
    sig = _make_clean_ppg(fs=30.0, duration=4.0)
    sig[10:13] = np.nan  # 3 missing samples out of 120 (2.5%)
    csv_bytes = pd.DataFrame({"signal": sig}).to_csv(index=False).encode("utf-8")
    res = parse_and_validate_upload(csv_bytes, "minor_nan.csv", manual_fs=30.0)
    assert not np.any(np.isnan(res["signal"]))
    assert len(res["warnings"]) > 0
    assert "Interpolated" in res["warnings"][0]


def test_upload_end_to_end_pipeline():
    """Verify that parsed upload feeds smoothly into feature extractor and kmeans model."""
    sig = _make_clean_ppg(fs=30.0, duration=10.0, bpm=72.0)
    df = pd.DataFrame({"signal": sig, "sampling_rate": [30.0] * len(sig)})
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    parsed = parse_and_validate_upload(csv_bytes, "pipeline_test.csv")
    extractor = PPGFeatureExtractor(default_fs=parsed["sampling_rate"])
    features = extractor.extract_from_signal(parsed["signal"], fs=parsed["sampling_rate"])
    assert features is not None
    assert "hr_bpm" in features
    assert features["detected_peaks_count"] > 5

    # Check KMeans model prediction
    pred = ArohanPredictor("models/unsupervised_kmeans.joblib")
    df_feat = pd.DataFrame([features])
    cluster_id = pred.model.clusterer.predict_clusters(df_feat)[0]
    assert isinstance(int(cluster_id), int)
    assert 0 <= int(cluster_id) < pred.model.clusterer.n_clusters


def test_upload_real_sample_ppg_csv():
    with open("data/sample_ppg.csv", "rb") as f:
        content = f.read()
    res = parse_and_validate_upload(content, "sample_ppg.csv")
    assert res["fs_auto_detected"] is True
    assert np.isclose(res["sampling_rate"], 26.04, atol=0.1)
    assert res["duration_sec"] > 50.0
    assert len(res["signal"]) > 1000


def test_upload_real_spandan_subset_json():
    with open("data/raw/spandan_subset.json", "r", encoding="utf-8") as f:
        recs = json.load(f)
    first_rec_bytes = json.dumps(recs[0]).encode("utf-8")
    res = parse_and_validate_upload(first_rec_bytes, "first_rec.json")
    assert res["fs_auto_detected"] is True
    assert res["duration_sec"] > 50.0
    assert len(res["signal"]) > 1000

