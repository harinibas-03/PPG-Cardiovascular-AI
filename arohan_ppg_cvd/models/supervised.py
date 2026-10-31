"""
Supervised Random Forest Classifier for CVD Screening
=====================================================
Trains a Random Forest classifier on confirmed medical diagnosis labels
(0 = Healthy / Control, 1 = Possible CVD / CAD).

Enforces strictly patient-level (or recording-level) split validation via
GroupKFold / GroupShuffleSplit to prevent data leakage across beats/windows.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class PPGSupervisedClassifier:
    """
    Supervised classifier for PPG cardiovascular risk classification.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: Optional[int] = 12,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Number of decision trees.
        max_depth : Optional[int]
            Max depth of trees to prevent overfitting.
        class_weight : str
            Class weight balancing strategy (default 'balanced').
        random_state : int
            Reproducibility seed.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state

        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    class_weight=class_weight,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ])

        self.feature_names: List[str] = []
        self.classes_: np.ndarray = np.array([0, 1])
        self.is_fitted: bool = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> "PPGSupervisedClassifier":
        """
        Fit the supervised classifier on confirmed ground-truth medical labels.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Features matrix.
        y : array-like
            Binary labels: 0 for Healthy, 1 for CVD / CAD.
        """
        X_arr, names = self._prepare_features(X, feature_names)
        y_arr = np.asarray(y, dtype=np.int64)

        if len(np.unique(y_arr)) < 2:
            raise ValueError(
                f"Training requires at least two classes (Healthy and CVD). Found classes: {np.unique(y_arr)}"
            )

        self.feature_names = names
        self.pipeline.fit(X_arr, y_arr)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        self.is_fitted = True
        logger.info(f"Successfully fitted PPGSupervisedClassifier on {len(y_arr)} samples.")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict binary class (0: Healthy, 1: Possible CVD).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Cannot predict without a trained model.")
        X_arr, _ = self._prepare_features(X, self.feature_names)
        return self.pipeline.predict(X_arr)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities. Output shape: (N, 2).
        Column 0: P(Healthy), Column 1: P(CVD / CAD).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Cannot predict without a trained model.")
        X_arr, _ = self._prepare_features(X, self.feature_names)
        return self.pipeline.predict_proba(X_arr)

    def cross_validate_patient_level(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        groups: Union[pd.Series, np.ndarray],
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform patient-level GroupKFold cross-validation to prevent intra-patient data leakage.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : array-like
            Ground truth binary diagnosis labels.
        groups : array-like
            Patient IDs or Recording IDs (`file_id`).
        n_splits : int
            Number of folds (default 5).

        Returns
        -------
        dict with cross-validation performance metrics (mean and std across folds).
        """
        X_arr, names = self._prepare_features(X)
        self.feature_names = names
        y_arr = np.asarray(y, dtype=np.int64)
        groups_arr = np.asarray(groups)

        unique_groups = len(np.unique(groups_arr))
        actual_splits = min(n_splits, unique_groups)
        if actual_splits < 2:
            raise ValueError(f"Patient-level CV requires at least 2 unique patients. Found: {unique_groups}")

        gkf = GroupKFold(n_splits=actual_splits)
        fold_metrics = []
        oof_preds = np.zeros(len(y_arr), dtype=np.int64)
        oof_probs = np.zeros(len(y_arr), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_arr, y_arr, groups=groups_arr)):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_va, y_va = X_arr[val_idx], y_arr[val_idx]

            self.pipeline.fit(X_tr, y_tr)
            probs = self.pipeline.predict_proba(X_va)[:, 1]
            preds = (probs >= 0.5).astype(np.int64)

            oof_preds[val_idx] = preds
            oof_probs[val_idx] = probs

            # Metrics
            roc_auc = roc_auc_score(y_va, probs) if len(np.unique(y_va)) > 1 else np.nan
            pr_auc = average_precision_score(y_va, probs) if len(np.unique(y_va)) > 1 else np.nan

            fold_metrics.append({
                "fold": fold + 1,
                "accuracy": accuracy_score(y_va, preds),
                "precision": precision_score(y_va, preds, zero_division=0),
                "recall": recall_score(y_va, preds, zero_division=0),
                "f1": f1_score(y_va, preds, zero_division=0),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            })

        # Fit on full training set after CV
        self.fit(X, y)

        df_metrics = pd.DataFrame(fold_metrics)
        summary = {
            "folds": fold_metrics,
            "mean_roc_auc": float(df_metrics["roc_auc"].mean()),
            "std_roc_auc": float(df_metrics["roc_auc"].std()),
            "mean_f1": float(df_metrics["f1"].mean()),
            "std_f1": float(df_metrics["f1"].std()),
            "mean_sensitivity": float(df_metrics["recall"].mean()),
            "mean_precision": float(df_metrics["precision"].mean()),
            "mean_accuracy": float(df_metrics["accuracy"].mean()),
            "oof_confusion_matrix": confusion_matrix(y_arr, oof_preds).tolist(),
        }

        return summary

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Return ranked feature importances from the fitted Random Forest.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        clf: RandomForestClassifier = self.pipeline.named_steps["clf"]
        importances = clf.feature_importances_
        names = self.feature_names if self.feature_names else [f"feat_{i}" for i in range(len(importances))]

        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    def _prepare_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        if isinstance(X, pd.DataFrame):
            meta_cols = {"file_id", "patient_id", "label", "target", "diagnosis"}
            if feature_names:
                cols = [c for c in feature_names if c in X.columns]
            else:
                cols = [c for c in X.columns if c not in meta_cols]
            X_arr = X[cols].to_numpy(dtype=np.float32)
            names = cols
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            names = feature_names if feature_names else [f"feat_{i}" for i in range(X_arr.shape[1])]

        return np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0), names
