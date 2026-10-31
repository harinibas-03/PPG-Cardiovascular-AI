"""
Unit tests for Machine Learning Pipeline: Unsupervised, Supervised, Hybrid, and Predictor.
"""

import pytest
import os
import numpy as np
import pandas as pd
from arohan_ppg_cvd.models.unsupervised import PPGUnsupervisedClusterer
from arohan_ppg_cvd.models.supervised import PPGSupervisedClassifier
from arohan_ppg_cvd.models.hybrid_pipeline import HybridPPGClassifier
from arohan_ppg_cvd.models.persistence import save_model_bundle, load_model_bundle
from arohan_ppg_cvd.inference.predictor import ArohanPredictor


@pytest.fixture
def dummy_dataset():
    """Generates synthetic tabular feature matrix for testing ML components."""
    np.random.seed(42)
    n_samples = 40
    n_feats = 10
    X = np.random.randn(n_samples, n_feats).astype(np.float32)
    y = np.random.choice([0, 1], size=n_samples)
    patients = np.repeat(np.arange(10), 4)  # 10 patients, 4 windows each

    cols = [f"feat_{i}" for i in range(n_feats)]
    df_X = pd.DataFrame(X, columns=cols)
    return df_X, y, patients


def test_unsupervised_clusterer(dummy_dataset):
    df_X, _, _ = dummy_dataset
    clusterer = PPGUnsupervisedClusterer(n_clusters=3, use_pca=True, n_components=4)
    clusterer.fit(df_X)

    assert clusterer.is_fitted
    clusters = clusterer.predict_clusters(df_X)
    assert len(clusters) == len(df_X)
    assert set(np.unique(clusters)).issubset({0, 1, 2})

    dists = clusterer.transform_to_cluster_distances(df_X)
    assert dists.shape == (len(df_X), 3)


def test_supervised_classifier_patient_cv(dummy_dataset):
    df_X, y, patients = dummy_dataset
    clf = PPGSupervisedClassifier(n_estimators=20, max_depth=4)
    cv_res = clf.cross_validate_patient_level(df_X, y, groups=patients, n_splits=3)

    assert clf.is_fitted
    assert "mean_roc_auc" in cv_res
    assert "mean_f1" in cv_res

    probs = clf.predict_proba(df_X)
    assert probs.shape == (len(df_X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_hybrid_pipeline(dummy_dataset):
    df_X, y, patients = dummy_dataset
    hybrid = HybridPPGClassifier(n_clusters=3, n_pca_components=4, n_estimators=20)
    hybrid.fit_supervised(df_X, y, groups=patients, cv_splits=3)

    assert hybrid.is_unsupervised_fitted
    assert hybrid.is_supervised_fitted

    probs = hybrid.predict_proba(df_X)
    assert probs.shape == (len(df_X), 2)
    preds = hybrid.predict(df_X)
    assert len(preds) == len(df_X)


def test_model_persistence(tmp_path, dummy_dataset):
    df_X, y, _ = dummy_dataset
    hybrid = HybridPPGClassifier(n_clusters=2, n_estimators=10)
    hybrid.fit_supervised(df_X, y)

    save_path = str(tmp_path / "test_model.joblib")
    save_model_bundle(hybrid, save_path, metadata={"test": True})
    assert os.path.exists(save_path)

    loaded = load_model_bundle(save_path)
    assert loaded.is_supervised_fitted
    probs_orig = hybrid.predict_proba(df_X)
    probs_loaded = loaded.predict_proba(df_X)
    assert np.allclose(probs_orig, probs_loaded)


def test_arohan_predictor_safety(tmp_path, dummy_dataset):
    # Case 1: Untrained predictor without model bundle
    predictor = ArohanPredictor()
    assert not predictor.is_ready_for_clinical_inference

    dummy_signal = np.sin(np.linspace(0, 10, 300)).astype(np.float32)
    res = predictor.analyze_signal(dummy_signal, fs=30.0)

    # Should not produce clinical predictions
    assert res["clinical_prediction"]["is_available"] is False
    assert "medical_disclaimer" in res

    # Case 2: Trained predictor
    df_X, y, _ = dummy_dataset
    hybrid = HybridPPGClassifier(n_clusters=2, n_estimators=10)
    hybrid.fit_supervised(df_X, y)

    model_file = str(tmp_path / "trained_screening.joblib")
    save_model_bundle(hybrid, model_file)

    predictor_trained = ArohanPredictor(model_file)
    assert predictor_trained.is_ready_for_clinical_inference
