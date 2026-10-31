"""Unit & Integration Tests for Clinical Dataset Loading, Deep 1D CNN Waveform Encoder,

Multi-Branch Feature Fusion, and Multimodal Readiness Schema.
"""

import os
import pytest
import numpy as np
import pandas as pd

from arohan_ppg_cvd.adapters.clinical_loader import (
    load_clinical_ppg_bp_dataset,
    split_clinical_by_subject,
)
from arohan_ppg_cvd.models.deep_waveform import DeepWaveformEncoder
from arohan_ppg_cvd.models.fusion import (
    MultimodalCVDClassifier,
    evaluate_predictions,
)
from arohan_ppg_cvd.models.multimodal_schema import (
    ClinicalLabBiomarkers,
    MultimodalPatientRecord,
    MultimodalFusionCoordinator,
)
from arohan_ppg_cvd.inference.predictor import ArohanPredictor

CLINICAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clinical")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

class TestClinicalDatasetLoader:
    """Validate PPG-BP clinical dataset ingestion and zero patient leakage."""

    @pytest.mark.skipif(not os.path.exists(CLINICAL_DIR), reason="Clinical dataset not downloaded")
    def test_load_clinical_dataset(self):
        df = load_clinical_ppg_bp_dataset(CLINICAL_DIR)
        assert len(df) > 0
        assert "subject_id" in df.columns
        assert "binary_target" in df.columns
        assert "raw_signal" in df.columns
        assert df["sampling_rate"].iloc[0] == 1000.0

        # Check binary labels are strictly 0 and 1
        unique_targets = set(df["binary_target"].unique())
        assert unique_targets.issubset({0, 1})

    @pytest.mark.skipif(not os.path.exists(CLINICAL_DIR), reason="Clinical dataset not downloaded")
    def test_zero_patient_leakage_split(self):
        df = load_clinical_ppg_bp_dataset(CLINICAL_DIR)
        train_df, test_df = split_clinical_by_subject(df, test_size=0.25, random_state=42)

        train_subjects = set(train_df["subject_id"].unique())
        test_subjects = set(test_df["subject_id"].unique())

        assert len(train_subjects.intersection(test_subjects)) == 0, "Patient leakage detected!"
        assert len(train_df) + len(test_df) == len(df)


class TestDeepWaveformEncoder:
    """Validate 1D CNN waveform encoder (Branch B)."""

    def test_encoder_output_shape(self):
        encoder = DeepWaveformEncoder(target_length=256, embedding_dim=32, random_state=42)
        signals = [
            np.sin(np.linspace(0, 10, 500)),
            np.cos(np.linspace(0, 15, 800)),
            np.random.randn(300),
        ]
        embeddings = encoder.transform(signals)

        assert embeddings.shape == (3, 32)
        # Check normalized embeddings (L2 norm ~ 1)
        for i in range(3):
            assert np.isclose(np.linalg.norm(embeddings[i]), 1.0, atol=1e-3)

    def test_encoder_handles_zero_signal(self):
        encoder = DeepWaveformEncoder(target_length=256, embedding_dim=32)
        emb = encoder.transform([np.array([])])
        assert emb.shape == (1, 32)


class TestMultimodalCVDClassifier:
    """Validate multi-branch fusion architecture."""

    def test_fused_training_and_inference(self):
        np.random.seed(42)
        N = 20
        X_struct = np.random.randn(N, 75).astype(np.float32)
        raw_signals = [np.random.randn(300) for _ in range(N)]
        y = np.random.randint(0, 2, size=N)

        clf = MultimodalCVDClassifier(mode="fused", n_estimators=10, random_state=42)
        clf.fit(X_structured=X_struct, raw_signals=raw_signals, y=y)

        preds = clf.predict(X_structured=X_struct, raw_signals=raw_signals)
        probs = clf.predict_proba(X_structured=X_struct, raw_signals=raw_signals)

        assert len(preds) == N
        assert probs.shape == (N, 2)
        assert np.allclose(np.sum(probs, axis=1), 1.0)

    def test_evaluation_metrics_computation(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0, 0])
        y_prob = np.array([
            [0.8, 0.2],
            [0.2, 0.8],
            [0.9, 0.1],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.7, 0.3],
        ])

        metrics = evaluate_predictions(y_true, y_pred, y_prob)
        assert "accuracy" in metrics
        assert "recall_sensitivity" in metrics
        assert "specificity" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert metrics["accuracy"] > 0.0


class TestMultimodalSchema:
    """Validate future clinical multimodal integration interface."""

    def test_patient_record_serialization(self):
        labs = ClinicalLabBiomarkers(
            systolic_bp_mmhg=135.0,
            diastolic_bp_mmhg=88.0,
            total_cholesterol_mg_dl=210.0,
            fasting_glucose_mg_dl=105.0,
        )
        rec = MultimodalPatientRecord(
            patient_id="IND_PAT_001",
            age=52.0,
            sex="M",
            bmi=26.4,
            wearable_ppg_features=np.zeros(75, dtype=np.float32),
            wearable_waveform_embedding=np.zeros(32, dtype=np.float32),
            clinical_labs=labs,
            clinical_text_embedding=np.zeros(128, dtype=np.float32),
        )

        d = rec.to_dict()
        assert d["patient_id"] == "IND_PAT_001"
        assert d["wearable_ppg_features_dim"] == 75
        assert d["wearable_waveform_embedding_dim"] == 32
        assert d["has_clinical_text_embedding"] is True

        fused_vec = rec.get_fused_feature_vector()
        # 3 (demo) + 10 (labs) + 75 (ppg) + 32 (emb) + 128 (text) = 248 dimensions
        assert len(fused_vec) == 248

    def test_multimodal_coordinator_late_fusion(self):
        coord = MultimodalFusionCoordinator(
            weight_wearable_ppg=0.5,
            weight_clinical_labs=0.3,
            weight_clinical_text=0.2,
        )
        res = coord.compute_late_fusion_risk(
            ppg_risk_score=0.75,
            labs_risk_score=0.60,
            text_risk_score=0.50,
        )
        assert 0.0 <= res["fused_cardiovascular_risk_score"] <= 1.0
        assert res["risk_category"] in ["Low Risk", "Moderate Risk", "High Risk"]


class TestPredictorWithClinicalModel:
    """Verify ArohanPredictor behavior with trained clinical model bundle."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(MODELS_DIR, "clinical_cvd_model.joblib")),
        reason="Clinical model bundle not found",
    )
    def test_predictor_clinical_bundle_inference(self):
        bundle_path = os.path.join(MODELS_DIR, "clinical_cvd_model.joblib")
        predictor = ArohanPredictor(bundle_path)
        assert predictor.is_ready_for_clinical_inference is True

        sig = np.sin(np.linspace(0, 10, 1000)) + 0.1 * np.random.randn(1000)
        res = predictor.analyze_signal(sig, fs=100.0)

        assert res["status"] == "success"
        pred = res["clinical_prediction"]
        assert pred["is_available"] is True
        assert 0.0 <= pred["probability_cvd"] <= 1.0
        assert pred["risk_tier"] is not None
        assert "External Clinical Benchmark" in pred["population_provenance"]

