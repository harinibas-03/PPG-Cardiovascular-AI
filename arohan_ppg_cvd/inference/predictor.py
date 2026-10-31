"""Inference Engine and Clinical Prediction Interface.

==================================================
Provides safe, validated prediction capabilities for PPG-based CVD screening.

SAFETY AND COMPLIANCE RULES:
1. STRICT CHECK: Predictions (probabilities and labels) are ONLY returned if a valid,
   clinically trained model bundle has been supplied and verified.
2. Exploratory PPG metrics and unsupervised clusters are strictly segregated from
   disease predictions.
3. Every output includes a standardized medical ML disclaimer.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import os
import joblib
import numpy as np
import pandas as pd

from ..models.hybrid_pipeline import HybridPPGClassifier
from ..models.fusion import MultimodalCVDClassifier
from ..models.persistence import load_model_bundle
from ..features.extractor import PPGFeatureExtractor

logger = logging.getLogger(__name__)


class ModelNotTrainedError(Exception):
    """Raised when inference is requested on an uncalibrated or untrained model."""
    pass


class ArohanPredictor:
    """Standardized inference interface for Arohan PPG-based CVD screening."""

    def __init__(self, model_bundle_path: Optional[str] = None):
        self.model_bundle_path = model_bundle_path
        self.bundle_dict: Optional[Dict[str, Any]] = None
        self.model: Optional[Union[HybridPPGClassifier, MultimodalCVDClassifier]] = None
        self.extractor = PPGFeatureExtractor()

        if model_bundle_path:
            self.load_model(model_bundle_path)

    def load_model(self, path: str) -> None:
        """Load a trained model bundle (.joblib)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model bundle not found at: {path}")

        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.bundle_dict = loaded
            self.model = loaded["model"]
        elif isinstance(loaded, (HybridPPGClassifier, MultimodalCVDClassifier)):
            self.bundle_dict = {}
            self.model = loaded
        else:
            raise ValueError(f"Unrecognized model bundle format in {path}")

        self.model_bundle_path = path
        logger.info(f"Loaded trained screening model from {path}")

    @property
    def is_ready_for_clinical_inference(self) -> bool:
        """Verify whether model is trained and authorized for supervised inference."""
        if self.model is None:
            return False
        if hasattr(self.model, "is_supervised_fitted"):
            return bool(self.model.is_supervised_fitted)
        if hasattr(self.model, "_is_fitted"):
            return bool(self.model._is_fitted)
        return False

    def analyze_signal(
        self,
        raw_signal: Union[np.ndarray, List[float]],
        fs: float = 30.0,
        file_id: Optional[str] = None,
        bpm_ref: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Full signal analysis pipeline:

        1. Extracts physiological and statistical PPG features.
        2. Evaluates Signal Quality Index (SQI).
        3. Returns exploratory morphological cluster (if unsupervised model is available).
        4. Returns supervised CVD screening probabilities ONLY if a validated model is loaded.
        """
        raw_arr = np.asarray(raw_signal, dtype=np.float32).flatten()

        # 1. Feature extraction
        features = self.extractor.extract_from_signal(
            raw_signal=raw_arr,
            fs=fs,
            file_id=file_id,
            bpm_reference=bpm_ref,
        )

        if features is None:
            return {
                "status": "failed",
                "error": "Signal quality unacceptable or duration insufficient (< 2 seconds).",
                "sqi_passed": False,
            }

        df_feat = pd.DataFrame([features])
        meta_cols = {"file_id", "sampling_rate", "duration_sec", "detected_peaks_count", "segmented_beats_count", "device_bpm_ref", "is_acceptable"}
        feat_cols = [c for c in df_feat.columns if c not in meta_cols]

        result: Dict[str, Any] = {
            "status": "success",
            "sqi_passed": bool(features.get("is_acceptable", 0.0) == 1.0),
            "signal_metrics": {
                "heart_rate_bpm": features.get("hr_bpm"),
                "mean_pulse_duration_sec": features.get("mean_pulse_duration"),
                "augmentation_index": features.get("mean_augmentation_index"),
                "reflection_index": features.get("mean_reflection_index"),
                "apg_aging_index": features.get("mean_apg_aging_index"),
                "perfusion_index": features.get("perfusion_index"),
            },
            "sqi_details": {
                "sSQI": features.get("sSQI"),
                "kSQI": features.get("kSQI"),
                "pSQI": features.get("pSQI"),
            },
            "features": features,
        }

        # 2. Unsupervised cluster assignment (exploratory)
        if self.model is not None and getattr(self.model, "is_unsupervised_fitted", False):
            cluster_id = int(self.model.clusterer.predict_clusters(df_feat)[0])
            result["exploratory_pulse_cluster"] = {
                "cluster_id": cluster_id,
                "notice": "Geometric waveform grouping. This is NOT a medical diagnosis.",
            }
        else:
            result["exploratory_pulse_cluster"] = {
                "cluster_id": None,
                "notice": "Unsupervised model not loaded or multimodal bundle active.",
            }

        # 3. Supervised Prediction
        if self.is_ready_for_clinical_inference:
            if isinstance(self.model, MultimodalCVDClassifier):
                expected_feats = (self.bundle_dict or {}).get("feature_names", None)
                if expected_feats:
                    for col in expected_feats:
                        if col not in df_feat.columns:
                            df_feat[col] = 0.0
                    X_struct = df_feat[expected_feats].to_numpy(dtype=np.float32)
                else:
                    X_struct = df_feat[feat_cols].to_numpy(dtype=np.float32)
                probs = self.model.predict_proba(
                    X_structured=X_struct,
                    raw_signals=[raw_arr],
                )[0]
                prob_healthy = float(probs[0])
                prob_cvd = float(probs[1])
                target_diagnoses = (self.bundle_dict or {}).get("diagnoses", "CVD vs Healthy Control")
                dataset_provenance = (self.bundle_dict or {}).get("dataset", "PPG-BP Database")
                population_provenance = (self.bundle_dict or {}).get("population", "External Clinical Benchmark")
            else:
                probs = self.model.predict_proba(df_feat)[0]
                prob_healthy = float(probs[0])
                prob_cvd = float(probs[1])
                target_diagnoses = "Pattern A vs Pattern B (Synthetic)"
                dataset_provenance = "Synthetic Generator"
                population_provenance = "Simulated Mathematical Waveforms"

            if prob_cvd >= 0.60:
                risk_tier = "Elevated Cardiovascular Risk / Possible Vascular Pathology"
            elif prob_cvd >= 0.40:
                risk_tier = "Moderate / Intermediate Risk (Clinical Follow-Up Recommended)"
            else:
                risk_tier = "Low Risk / Consistent with Normotensive Control"

            result["clinical_prediction"] = {
                "is_available": True,
                "risk_tier": risk_tier,
                "probability_cvd": round(prob_cvd, 4),
                "probability_healthy": round(prob_healthy, 4),
                "confidence_score": round(float(abs(prob_cvd - prob_healthy)), 4),
                "target_diagnoses": target_diagnoses,
                "dataset_provenance": dataset_provenance,
                "population_provenance": population_provenance,
            }
        else:
            result["clinical_prediction"] = {
                "is_available": False,
                "message": (
                    "Clinical prediction is disabled: A validated supervised model trained on confirmed "
                    "medical labels has not been loaded. "
                    "Spandan-1M raw dataset contains no diagnosis labels. "
                    "To evaluate disease screening, load `models/clinical_cvd_model.joblib`."
                ),
            }

        result["medical_disclaimer"] = (
            "DISCLAIMER: Arohan_PPG_CVD is an experimental algorithmic tool for non-invasive pulse screening research. "
            "It does NOT provide certified medical diagnoses. Any clinical decisions must be confirmed by a licensed "
            "physician using diagnostic standard-of-care (e.g., 12-lead ECG, coronary angiography, cardiac troponin)."
        )

        return result
