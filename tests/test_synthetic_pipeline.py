"""
Unit tests for the synthetic dataset generation, supervised modeling, and hybrid inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from arohan_ppg_cvd.models.hybrid_pipeline import HybridPPGClassifier
from arohan_ppg_cvd.models.persistence import load_model_bundle
from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor
from scripts.generate_synthetic_dataset import generate_single_pulse_wave
from scripts.train_synthetic_hybrid import split_by_subject, extract_feature_matrix


def test_synthetic_pulse_generation():
    """Verify synthetic pulse generation generates valid signals without NaNs."""
    t = np.linspace(0, 5.0, 150)
    sig_a = generate_single_pulse_wave(t, bpm=72.0, pattern="A", amplitude=1.0)
    sig_b = generate_single_pulse_wave(t, bpm=84.0, pattern="B", amplitude=0.9)

    assert len(sig_a) == 150
    assert len(sig_b) == 150
    assert not np.any(np.isnan(sig_a))
    assert not np.any(np.isnan(sig_b))
    assert np.ptp(sig_a) > 0.5
    assert np.ptp(sig_b) > 0.5


def test_synthetic_dataset_files_exist():
    """Ensure generated synthetic data files exist and have correct shape and distribution."""
    feat_path = "data/synthetic/synthetic_features.csv"
    meta_path = "data/synthetic/synthetic_metadata.csv"
    assert os.path.exists(feat_path), f"Missing {feat_path}"
    assert os.path.exists(meta_path), f"Missing {meta_path}"

    df = pd.read_csv(feat_path)
    assert len(df) == 1000
    assert df["label"].value_counts()[0] == 500
    assert df["label"].value_counts()[1] == 500
    assert df["subject_id"].nunique() == 100
    assert df.isnull().sum().sum() == 0


def test_subject_level_splitting_zero_leakage():
    """Verify that split_by_subject produces strictly zero subject overlap."""
    df = pd.read_csv("data/synthetic/synthetic_features.csv")
    train_df, test_df = split_by_subject(df, test_ratio=0.20, random_seed=42)

    train_subs = set(train_df["subject_id"].unique())
    test_subs = set(test_df["subject_id"].unique())

    # Zero intersection between subjects
    assert len(train_subs.intersection(test_subs)) == 0
    assert len(train_subs) == 80
    assert len(test_subs) == 20
    assert len(train_df) == 800
    assert len(test_df) == 200


def test_trained_synthetic_hybrid_model_loading_and_inference():
    """Test loading and inference with models/synthetic_hybrid_model.joblib."""
    model_path = "models/synthetic_hybrid_model.joblib"
    assert os.path.exists(model_path), f"Missing {model_path}"

    bundle = load_model_bundle(model_path)
    assert isinstance(bundle, HybridPPGClassifier)
    assert bundle.is_unsupervised_fitted is True
    assert bundle.is_supervised_fitted is True

    # Test prediction on a sample
    df = pd.read_csv("data/synthetic/synthetic_features.csv")
    X, _, _ = extract_feature_matrix(df.iloc[:5])
    probs = bundle.predict_proba(X)
    preds = bundle.predict(X)

    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert len(preds) == 5
    assert set(np.unique(preds)).issubset({0, 1})


def test_real_kmeans_model_not_overwritten():
    """Safety check: Ensure models/unsupervised_kmeans.joblib was not modified or overwritten."""
    spandan_kmeans_path = "models/unsupervised_kmeans.joblib"
    assert os.path.exists(spandan_kmeans_path)

    bundle = joblib.load(spandan_kmeans_path)
    model = bundle["model"]
    # Real Spandan model has unsupervised fitted but supervised NOT fitted
    assert model.is_unsupervised_fitted is True
    assert model.is_supervised_fitted is False

