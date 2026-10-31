"""
Hybrid Supervised + Unsupervised AI Pipeline
============================================
Combines unsupervised morphological clustering (K-Means + PCA) with supervised
Random Forest classification for cardiovascular risk screening.

The unsupervised stage discovers geometric pulse clusters from PPG features.
Soft distances to cluster centroids (and one-hot cluster representations)
are passed to the supervised stage as augmented representation features.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd

from .unsupervised import PPGUnsupervisedClusterer
from .supervised import PPGSupervisedClassifier
from ..features.extractor import PPGFeatureExtractor

logger = logging.getLogger(__name__)


class HybridPPGClassifier:
    """
    Hybrid Supervised + Unsupervised Architecture for PPG-based CVD Screening.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        use_pca: bool = True,
        n_pca_components: int = 6,
        n_estimators: int = 150,
        max_depth: Optional[int] = 12,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.use_pca = use_pca
        self.n_pca_components = n_pca_components
        self.random_state = random_state

        self.clusterer = PPGUnsupervisedClusterer(
            n_clusters=n_clusters,
            use_pca=use_pca,
            n_components=n_pca_components,
            random_state=random_state,
        )

        self.classifier = PPGSupervisedClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
        )

        self.feature_extractor = PPGFeatureExtractor()
        self.feature_names: List[str] = []
        self.augmented_feature_names: List[str] = []
        self.is_unsupervised_fitted: bool = False
        self.is_supervised_fitted: bool = False

    def fit_unsupervised(self, X: Union[pd.DataFrame, np.ndarray]) -> "HybridPPGClassifier":
        """
        Stage 1: Fit unsupervised K-Means representation learner on PPG features.
        Can be trained on unlabeled data (e.g. Spandan-1M exploratory subset).
        """
        logger.info(f"Fitting unsupervised clusterer (k={self.n_clusters})...")
        self.clusterer.fit(X)
        self.feature_names = self.clusterer.feature_names
        self.is_unsupervised_fitted = True
        return self

    def augment_features(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Transform base features into hybrid feature matrix by augmenting with:
        - Euclidean distances to each cluster centroid (continuous cluster affinities)
        - One-hot encoded cluster assignment
        """
        if not self.is_unsupervised_fitted:
            raise RuntimeError("Unsupervised clusterer must be fitted before feature augmentation.")

        # Extract base array
        if isinstance(X, pd.DataFrame):
            meta_cols = {"file_id", "patient_id", "label", "target", "diagnosis"}
            cols = [c for c in self.feature_names if c in X.columns] if self.feature_names else [
                c for c in X.columns if c not in meta_cols
            ]
            X_base = X[cols].to_numpy(dtype=np.float32)
            base_names = cols
        else:
            X_base = np.asarray(X, dtype=np.float32)
            base_names = self.feature_names if self.feature_names else [f"feat_{i}" for i in range(X_base.shape[1])]

        X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0)

        # Cluster distances shape: (N, n_clusters)
        distances = self.clusterer.transform_to_cluster_distances(X_base)
        cluster_labels = self.clusterer.predict_clusters(X_base)

        # One-hot encoded clusters shape: (N, n_clusters)
        n_samples = len(X_base)
        one_hot_clusters = np.zeros((n_samples, self.n_clusters), dtype=np.float32)
        for i, c in enumerate(cluster_labels):
            one_hot_clusters[i, c] = 1.0

        # Concatenate: [base_features, distances, one_hot]
        X_augmented = np.hstack([X_base, distances, one_hot_clusters])

        aug_names = list(base_names)
        aug_names += [f"cluster_dist_{k}" for k in range(self.n_clusters)]
        aug_names += [f"cluster_onehot_{k}" for k in range(self.n_clusters)]

        return X_augmented, aug_names

    def fit_supervised(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        groups: Optional[Union[pd.Series, np.ndarray]] = None,
        cv_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        Stage 2: Fit supervised classifier using confirmed ground-truth diagnosis labels.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Base feature matrix.
        y : array-like
            Ground truth binary clinical labels (0 = Healthy, 1 = CVD / CAD).
        groups : Optional array-like
            Patient / recording IDs for patient-level cross validation.
        cv_splits : int
            Number of folds for validation.
        """
        if not self.is_unsupervised_fitted:
            # Auto-fit unsupervised stage on training features if not already fitted
            logger.info("Auto-fitting unsupervised clustering on training set...")
            self.fit_unsupervised(X)

        X_aug, aug_names = self.augment_features(X)
        self.augmented_feature_names = aug_names

        cv_results = {}
        if groups is not None and len(np.unique(groups)) >= 2:
            df_aug = pd.DataFrame(X_aug, columns=aug_names)
            cv_results = self.classifier.cross_validate_patient_level(
                df_aug, y, groups=groups, n_splits=cv_splits
            )
        else:
            self.classifier.fit(X_aug, y, feature_names=aug_names)

        self.is_supervised_fitted = True
        logger.info("Hybrid pipeline training complete.")
        return cv_results

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for Healthy vs Possible CVD.
        Returns shape (N, 2): [P(Healthy), P(CVD)].
        """
        if not self.is_supervised_fitted:
            raise RuntimeError(
                "Supervised classifier is NOT fitted yet. "
                "The pipeline requires training on a medically labeled dataset before generating predictions."
            )

        X_aug, _ = self.augment_features(X)
        return self.classifier.predict_proba(X_aug)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict binary labels (0 = Healthy, 1 = Possible CVD).
        """
        if not self.is_supervised_fitted:
            raise RuntimeError(
                "Supervised classifier is NOT fitted yet. "
                "The pipeline requires training on a medically labeled dataset before generating predictions."
            )

        X_aug, _ = self.augment_features(X)
        return self.classifier.predict(X_aug)

    def predict_from_raw_signal(
        self,
        raw_signal: Union[np.ndarray, List[float]],
        fs: float = 30.0,
    ) -> Dict[str, Any]:
        """
        End-to-end inference for a single raw PPG recording.
        """
        if not self.is_supervised_fitted:
            raise RuntimeError(
                "Cannot perform clinical prediction: model has not been trained on a labeled medical dataset."
            )

        feat_dict = self.feature_extractor.extract_from_signal(raw_signal, fs=fs)
        if feat_dict is None:
            return {
                "status": "error",
                "message": "Signal quality check failed or signal too short.",
            }

        df_feat = pd.DataFrame([feat_dict])
        probs = self.predict_proba(df_feat)[0]
        cluster_id = int(self.clusterer.predict_clusters(df_feat)[0])

        prob_cvd = float(probs[1])
        prediction_label = "Possible CVD / CAD" if prob_cvd >= 0.5 else "Healthy"

        return {
            "status": "success",
            "prediction": prediction_label,
            "probability_healthy": round(float(probs[0]), 4),
            "probability_cvd": round(prob_cvd, 4),
            "exploratory_cluster_id": cluster_id,
            "signal_quality": {
                "sSQI": feat_dict.get("sSQI"),
                "kSQI": feat_dict.get("kSQI"),
                "pSQI": feat_dict.get("pSQI"),
                "perfusion_index": feat_dict.get("perfusion_index"),
            },
            "features": feat_dict,
            "disclaimer": (
                "MEDICAL RESEARCH NOTICE: This prediction is generated for research and educational purposes. "
                "It is not a certified diagnostic device and does not substitute for a clinical 12-lead ECG, "
                "troponin biomarkers, or cardiologist evaluation."
            ),
        }
