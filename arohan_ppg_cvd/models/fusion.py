"""Multimodal & Multi-Branch Fusion Architecture for Cardiovascular Disease AI.

This module unifies:
- Branch A: Structured 75-D physiological feature vector (HR, PPI, morphology, crest time, SQI, VPG, APG, spectral).
- Branch B: Unstructured 32-D raw waveform embedding from the 1D CNN waveform encoder.
- Branch C: Unsupervised cluster soft-affinities from K-Means clustering.

Provides model training, evaluation, comparison, and unified clinical inference.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from arohan_ppg_cvd.models.deep_waveform import DeepWaveformEncoder

class MultimodalCVDClassifier(BaseEstimator, ClassifierMixin):
    """Unified Multimodal Classifier combining structured features, deep waveform embeddings,

    and optional unsupervised cluster features.
    """

    def __init__(
        self,
        mode: str = "fused",  # 'structured_only', 'waveform_only', 'fused', 'hybrid_kmeans'
        n_estimators: int = 150,
        max_depth: Optional[int] = 10,
        random_state: int = 42,
    ):
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.scaler_struct = StandardScaler()
        self.scaler_fused = StandardScaler()
        self.waveform_encoder = DeepWaveformEncoder(random_state=random_state)
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
        )
        self.feature_names_: List[str] = []
        self._is_fitted = False

    def _prepare_features(
        self,
        X_structured: Optional[np.ndarray] = None,
        raw_signals: Optional[List[np.ndarray]] = None,
        cluster_distances: Optional[np.ndarray] = None,
        fit: bool = False,
    ) -> np.ndarray:
        """Combine modalities according to the configured mode."""
        parts = []

        if self.mode in ["structured_only", "fused", "hybrid_kmeans"]:
            if X_structured is None:
                raise ValueError("X_structured is required for mode=" + self.mode)
            X_struct_clean = np.nan_to_num(X_structured, nan=0.0, posinf=0.0, neginf=0.0)
            if fit:
                X_struct_scaled = self.scaler_struct.fit_transform(X_struct_clean)
            else:
                X_struct_scaled = self.scaler_struct.transform(X_struct_clean)
            parts.append(X_struct_scaled)

        if self.mode in ["waveform_only", "fused", "hybrid_kmeans"]:
            if raw_signals is None:
                raise ValueError("raw_signals is required for mode=" + self.mode)
            # Encode raw waveforms
            emb = self.waveform_encoder.transform(raw_signals)
            parts.append(emb)

        if self.mode == "hybrid_kmeans":
            if cluster_distances is not None:
                parts.append(cluster_distances)

        if len(parts) == 1:
            X_comb = parts[0]
        else:
            X_comb = np.hstack(parts)

        return X_comb.astype(np.float32)

    def fit(
        self,
        X_structured: Optional[np.ndarray] = None,
        raw_signals: Optional[List[np.ndarray]] = None,
        y: Optional[np.ndarray] = None,
        cluster_distances: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """Fit multimodal pipeline."""
        if y is None:
            raise ValueError("Target labels y must be provided.")
        
        self.waveform_encoder.fit(raw_signals if raw_signals is not None else [])
        X_mat = self._prepare_features(
            X_structured=X_structured,
            raw_signals=raw_signals,
            cluster_distances=cluster_distances,
            fit=True,
        )

        if self.mode in ["fused", "hybrid_kmeans"]:
            X_mat = self.scaler_fused.fit_transform(X_mat)

        self.classifier.fit(X_mat, y)
        self._is_fitted = True
        return self

    def predict(
        self,
        X_structured: Optional[np.ndarray] = None,
        raw_signals: Optional[List[np.ndarray]] = None,
        cluster_distances: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict binary disease label (0: Healthy, 1: CVD)."""
        X_mat = self._prepare_features(
            X_structured=X_structured,
            raw_signals=raw_signals,
            cluster_distances=cluster_distances,
            fit=False,
        )
        if self.mode in ["fused", "hybrid_kmeans"]:
            X_mat = self.scaler_fused.transform(X_mat)
        return self.classifier.predict(X_mat)

    def predict_proba(
        self,
        X_structured: Optional[np.ndarray] = None,
        raw_signals: Optional[List[np.ndarray]] = None,
        cluster_distances: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict disease probability distribution [P(Healthy), P(CVD)]."""
        X_mat = self._prepare_features(
            X_structured=X_structured,
            raw_signals=raw_signals,
            cluster_distances=cluster_distances,
            fit=False,
        )
        if self.mode in ["fused", "hybrid_kmeans"]:
            X_mat = self.scaler_fused.transform(X_mat)
        return self.classifier.predict_proba(X_mat)

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive clinical classification metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    roc_auc = float(roc_auc_score(y_true, y_proba[:, 1])) if y_proba is not None else None
    pr_auc = float(average_precision_score(y_true, y_proba[:, 1])) if y_proba is not None else None

    return {
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

