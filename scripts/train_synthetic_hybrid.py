"""
Synthetic Hybrid Machine Learning Benchmark
===========================================
Trains and compares:
1. Baseline Random Forest classifier on base 75-dimensional PPG features.
2. Hybrid Random Forest classifier integrating K-Means unsupervised cluster representations
   (cluster centroid distances and one-hot cluster affinities).

Evaluates both models on held-out synthetic subjects with strictly zero intra-subject data leakage.

SAFETY AND RESEARCH NOTICE:
This training workflow uses computer-generated synthetic PPG data (Pattern A vs Pattern B)
for educational and pipeline-demonstration purposes. The performance numbers DO NOT represent
clinical CAD/CVD diagnostic accuracy.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arohan_ppg_cvd.models.hybrid_pipeline import HybridPPGClassifier
from arohan_ppg_cvd.models.persistence import save_model_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_synthetic_hybrid")


def split_by_subject(
    df: pd.DataFrame,
    test_ratio: float = 0.20,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset strictly at the subject level so that no subject's recordings
    appear in both train and test partitions.
    """
    rng = np.random.RandomState(random_seed)
    subjects_class0 = df[df["label"] == 0]["subject_id"].unique()
    subjects_class1 = df[df["label"] == 1]["subject_id"].unique()

    n_test_c0 = max(1, int(len(subjects_class0) * test_ratio))
    n_test_c1 = max(1, int(len(subjects_class1) * test_ratio))

    test_subs_c0 = rng.choice(subjects_class0, size=n_test_c0, replace=False)
    test_subs_c1 = rng.choice(subjects_class1, size=n_test_c1, replace=False)
    test_subs = set(test_subs_c0).union(set(test_subs_c1))

    train_df = df[~df["subject_id"].isin(test_subs)].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subs)].reset_index(drop=True)

    logger.info(
        f"Subject split complete: {len(train_df['subject_id'].unique())} train subjects ({len(train_df)} recs), "
        f"{len(test_df['subject_id'].unique())} test subjects ({len(test_df)} recs)."
    )
    return train_df, test_df


def extract_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Extract numeric features, excluding identifiers and ground-truth metadata."""
    meta_cols = {"file_id", "subject_id", "label", "pattern", "device_bpm_ref", "record_id"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].copy()
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y, feat_cols


def train_and_evaluate(
    data_path: str = "data/synthetic/synthetic_features.csv",
    output_dir: str = "models",
    reports_dir: str = "reports",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Train and evaluate both Baseline RF and Hybrid RF models on synthetic PPG features.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    logger.info(f"Loading synthetic features from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Subject-level Train / Test Split
    train_df, test_df = split_by_subject(df, test_ratio=0.20, random_seed=random_seed)
    X_train_df, y_train, feature_names = extract_feature_matrix(train_df)
    X_test_df, y_test, _ = extract_feature_matrix(test_df)

    logger.info(f"Number of base input features: {len(feature_names)}")

    # ----------------------------------------------------
    # 2. Model 1: Baseline Random Forest
    # ----------------------------------------------------
    logger.info("Training Model 1: Baseline Random Forest (Original Features)...")
    baseline_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=150, max_depth=12, random_state=random_seed, n_jobs=-1)),
    ])
    baseline_pipe.fit(X_train_df, y_train)

    y_pred_base = baseline_pipe.predict(X_test_df)
    y_prob_base = baseline_pipe.predict_proba(X_test_df)[:, 1]

    cm_base = confusion_matrix(y_test, y_pred_base)
    base_metrics = {
        "model_type": "Baseline Random Forest",
        "feature_count": len(feature_names),
        "accuracy": round(float(accuracy_score(y_test, y_pred_base)), 4),
        "precision": round(float(precision_score(y_test, y_pred_base, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_base, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred_base, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_base)), 4),
        "confusion_matrix": {
            "tn": int(cm_base[0, 0]),
            "fp": int(cm_base[0, 1]),
            "fn": int(cm_base[1, 0]),
            "tp": int(cm_base[1, 1]),
        },
    }

    # ----------------------------------------------------
    # 3. Model 2: Hybrid K-Means + Random Forest
    # ----------------------------------------------------
    logger.info("Training Model 2: Hybrid Pipeline (K-Means Representation + Random Forest)...")
    hybrid = HybridPPGClassifier(
        n_clusters=4,
        use_pca=True,
        n_pca_components=6,
        n_estimators=150,
        max_depth=12,
        random_state=random_seed,
    )

    # Fit unsupervised stage strictly on training features
    hybrid.fit_unsupervised(X_train_df)

    # Augment features: base + centroid distances + cluster one-hot
    X_train_aug, aug_names = hybrid.augment_features(X_train_df)
    X_test_aug, _ = hybrid.augment_features(X_test_df)

    # Train supervised classifier on augmented training features
    hybrid.classifier.fit(X_train_aug, y_train, feature_names=aug_names)
    hybrid.is_supervised_fitted = True
    hybrid.feature_names = feature_names
    hybrid.augmented_feature_names = aug_names

    y_pred_hyb = hybrid.classifier.predict(X_test_aug)
    y_prob_hyb = hybrid.classifier.predict_proba(X_test_aug)[:, 1]

    cm_hyb = confusion_matrix(y_test, y_pred_hyb)
    hybrid_metrics = {
        "model_type": "Hybrid (K-Means + Random Forest)",
        "base_feature_count": len(feature_names),
        "augmented_feature_count": len(aug_names),
        "k_clusters": hybrid.n_clusters,
        "accuracy": round(float(accuracy_score(y_test, y_pred_hyb)), 4),
        "precision": round(float(precision_score(y_test, y_pred_hyb, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_hyb, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred_hyb, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_hyb)), 4),
        "confusion_matrix": {
            "tn": int(cm_hyb[0, 0]),
            "fp": int(cm_hyb[0, 1]),
            "fn": int(cm_hyb[1, 0]),
            "tp": int(cm_hyb[1, 1]),
        },
    }

    # ----------------------------------------------------
    # 4. Save Models (Without Touching Real Spandan Models)
    # ----------------------------------------------------
    base_model_path = os.path.join(output_dir, "synthetic_baseline_rf.joblib")
    joblib.dump({
        "pipeline": baseline_pipe,
        "feature_names": feature_names,
        "metrics": base_metrics,
        "dataset_type": "synthetic_benchmark",
    }, base_model_path)
    logger.info(f"Saved baseline model to {base_model_path}")

    hybrid_model_path = os.path.join(output_dir, "synthetic_hybrid_model.joblib")
    save_model_bundle(
        model=hybrid,
        save_path=hybrid_model_path,
        metadata={
            "metrics": {
                "test_accuracy": hybrid_metrics["accuracy"],
                "test_f1": hybrid_metrics["f1_score"],
                "test_roc_auc": hybrid_metrics["roc_auc"],
            },
            "dataset_name": "synthetic_ppg_1000",
            "notes": "Proof-of-concept hybrid K-Means + Random Forest model trained on simulated PPG patterns.",
        },
    )
    logger.info(f"Saved hybrid model bundle to {hybrid_model_path}")

    # ----------------------------------------------------
    # 5. Generate Comparison Plots & Reports
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Confusion Matrix: Baseline
    sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=["Pattern A", "Pattern B"], yticklabels=["Pattern A", "Pattern B"])
    axes[0].set_title(f"Baseline RF (Acc: {base_metrics['accuracy']*100:.1f}%)", fontweight="bold")
    axes[0].set_ylabel("True Pattern")
    axes[0].set_xlabel("Predicted Pattern")

    # Confusion Matrix: Hybrid
    sns.heatmap(cm_hyb, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1],
                xticklabels=["Pattern A", "Pattern B"], yticklabels=["Pattern A", "Pattern B"])
    axes[1].set_title(f"Hybrid K-Means + RF (Acc: {hybrid_metrics['accuracy']*100:.1f}%)", fontweight="bold")
    axes[1].set_ylabel("True Pattern")
    axes[1].set_xlabel("Predicted Pattern")

    # ROC Curves Comparison
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    fpr_hyb, tpr_hyb, _ = roc_curve(y_test, y_prob_hyb)

    axes[2].plot(fpr_base, tpr_base, color="#2563eb", lw=2, label=f"Baseline (AUC = {base_metrics['roc_auc']:.3f})")
    axes[2].plot(fpr_hyb, tpr_hyb, color="#059669", lw=2.5, linestyle="--", label=f"Hybrid (AUC = {hybrid_metrics['roc_auc']:.3f})")
    axes[2].plot([0, 1], [0, 1], color="#94a3b8", linestyle=":")
    axes[2].set_title("ROC Curves on Held-Out Subjects", fontweight="bold")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].legend(loc="lower right")
    axes[2].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(reports_dir, "synthetic_evaluation_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved evaluation plot to {plot_path}")

    # Comprehensive summary report
    report_data = {
        "dataset": {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_subjects": int(train_df["subject_id"].nunique()),
            "test_subjects": int(test_df["subject_id"].nunique()),
            "evaluation_strategy": "Strict Subject-Level Held-Out Test Split (Zero Data Leakage)",
        },
        "baseline_model": base_metrics,
        "hybrid_model": hybrid_metrics,
        "performance_delta": {
            "accuracy_change": round(hybrid_metrics["accuracy"] - base_metrics["accuracy"], 4),
            "f1_change": round(hybrid_metrics["f1_score"] - base_metrics["f1_score"], 4),
            "roc_auc_change": round(hybrid_metrics["roc_auc"] - base_metrics["roc_auc"], 4),
        },
        "disclaimer": (
            "RESEARCH DEMONSTRATION NOTICE: Results are calculated strictly on computer-generated synthetic PPG data "
            "(Pattern A vs Pattern B). These metrics DO NOT represent clinical Coronary Artery Disease (CAD) or "
            "Cardiovascular Disease (CVD) screening performance."
        ),
    }

    report_json_path = os.path.join(reports_dir, "synthetic_hybrid_evaluation.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info(f"Saved evaluation JSON report to {report_json_path}")

    # Markdown report
    md_content = f"""# Synthetic PPG Hybrid Machine Learning Evaluation Report

> **DISCLAIMER:**  
> This benchmark evaluates a proof-of-concept machine learning pipeline on **1,000 computer-generated synthetic PPG recordings** (Pattern A vs Pattern B).  
> These results **DO NOT** claim or demonstrate clinical diagnostic accuracy for Coronary Artery Disease (CAD) or Cardiovascular Disease (CVD).

---

## 1. Experimental Setup & Leakage-Free Design
- **Total Dataset**: 1,000 recordings (100 synthetic subjects, 10 recordings each, 6.0 seconds @ 30 Hz).
- **Split Strategy**: **Subject-Level Held-Out Split** (80 training subjects / 800 recordings vs 20 held-out test subjects / 200 recordings).
- **Pre-processing**: Scaler, Imputer, and K-Means representations fitted **strictly on training subjects**. Zero test data was observed during representation learning or scaling.

---

## 2. Performance Comparison on Held-Out Test Subjects

| Metric | Baseline Random Forest | Hybrid (K-Means + RF) | Relative Change |
| :--- | :---: | :---: | :---: |
| **Input Features** | {base_metrics['feature_count']} base features | {hybrid_metrics['augmented_feature_count']} features ({base_metrics['feature_count']} base + 8 K-Means) | +8 features |
| **Test Accuracy** | **{base_metrics['accuracy']*100:.2f}%** | **{hybrid_metrics['accuracy']*100:.2f}%** | **{report_data['performance_delta']['accuracy_change']*100:+.2f}%** |
| **Precision** | {base_metrics['precision']*100:.2f}% | {hybrid_metrics['precision']*100:.2f}% | {report_data['performance_delta']['accuracy_change']*100:+.2f}% |
| **Recall (Sensitivity)** | {base_metrics['recall']*100:.2f}% | {hybrid_metrics['recall']*100:.2f}% | {(hybrid_metrics['recall']-base_metrics['recall'])*100:+.2f}% |
| **F1-Score** | {base_metrics['f1_score']:.4f} | {hybrid_metrics['f1_score']:.4f} | {report_data['performance_delta']['f1_change']:+.4f} |
| **ROC-AUC** | **{base_metrics['roc_auc']:.4f}** | **{hybrid_metrics['roc_auc']:.4f}** | **{report_data['performance_delta']['roc_auc_change']:+.4f}** |

---

## 3. Confusion Matrix Breakdown (200 Held-Out Test Recordings)

### Baseline Model
- **True Pattern A (TN)**: {base_metrics['confusion_matrix']['tn']}
- **False Pattern B (FP)**: {base_metrics['confusion_matrix']['fp']}
- **False Pattern A (FN)**: {base_metrics['confusion_matrix']['fn']}
- **True Pattern B (TP)**: {base_metrics['confusion_matrix']['tp']}

### Hybrid Model
- **True Pattern A (TN)**: {hybrid_metrics['confusion_matrix']['tn']}
- **False Pattern B (FP)**: {hybrid_metrics['confusion_matrix']['fp']}
- **False Pattern A (FN)**: {hybrid_metrics['confusion_matrix']['fn']}
- **True Pattern B (TP)**: {hybrid_metrics['confusion_matrix']['tp']}

---

## 4. Key Engineering Takeaways
1. **Unsupervised Representation Value**: Integrating unsupervised K-Means cluster distance features provides geometric clustering affinities directly to the decision trees, demonstrating the complete hybrid workflow.
2. **Clinical Translation Readiness**: When clinically annotated CAD/CVD data becomes available, the exact same pipeline (`HybridPPGClassifier`) can be trained by running `scripts/train_supervised.py --data <clinical_dataset.csv>`.
"""
    report_md_path = os.path.join(reports_dir, "synthetic_evaluation_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved evaluation Markdown report to {report_md_path}")

    return report_data


if __name__ == "__main__":
    train_and_evaluate()
