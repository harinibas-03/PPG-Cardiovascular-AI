"""Clinical Cardiovascular Disease AI Training & Multi-Branch Evaluation Pipeline.

Dataset: PPG-BP Database (Guilin People's Hospital, Liang et al. 2018)
Target: Confirmed Cardiovascular Disease (Hypertension Stages 1-2, Stroke, Cerebrovascular)
vs Strict Healthy Normotensive Controls.

Evaluates 4 architectures:
1. Structured-Only Classifier (75-D Physiological Features)
2. Waveform-Only Deep Classifier (32-D 1D CNN Embeddings)
3. Fused Deep + Structured Classifier (107-D Multimodal Vector)
4. Hybrid Fused + K-Means Classifier (112-D Representation)

Guarantees 100% zero patient leakage via GroupShuffleSplit on subject_id.
"""

import os
import json
from typing import Tuple, List, Dict, Optional, Any
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from arohan_ppg_cvd.adapters.clinical_loader import (
    load_clinical_ppg_bp_dataset,
    split_clinical_by_subject,
)
from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor
from arohan_ppg_cvd.models.deep_waveform import DeepWaveformEncoder
from arohan_ppg_cvd.models.fusion import MultimodalCVDClassifier, evaluate_predictions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clinical")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
CACHE_PATH = os.path.join(DATA_DIR, "clinical_features_cache.csv")

def extract_or_load_features(df_clinical: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Extract 75-D physiological features or load from local cache."""
    extractor = PPGFeatureExtractor(default_fs=1000.0)
    
    feature_records = []
    print(f"Extracting physiological features for {len(df_clinical)} clinical segments...")
    for idx, row in df_clinical.iterrows():
        raw_sig = row["raw_signal"]
        feats = extractor.extract_from_signal(raw_sig, fs=1000.0)
        feats["subject_id"] = row["subject_id"]
        feats["segment_idx"] = row["segment_idx"]
        feats["binary_target"] = row["binary_target"]
        feature_records.append(feats)

    df_feats = pd.DataFrame(feature_records)
    feat_cols = [c for c in df_feats.columns if c not in ["subject_id", "segment_idx", "binary_target"]]
    
    # Fill NAs with median
    for c in feat_cols:
        df_feats[c] = pd.to_numeric(df_feats[c], errors="coerce")
        median_val = df_feats[c].median()
        df_feats[c] = df_feats[c].fillna(median_val if pd.notna(median_val) else 0.0)

    X_mat = df_feats[feat_cols].to_numpy(dtype=np.float32)
    return df_feats, X_mat, feat_cols

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("Step 1: Ingesting PPG-BP clinical benchmark dataset...")
    df_raw = load_clinical_ppg_bp_dataset()
    print(f"Loaded {len(df_raw)} recordings across {df_raw['subject_id'].nunique()} subjects.")
    print(f"Cohort Breakdown: {df_raw['target_name'].value_counts().to_dict()}")

    print("\nStep 2: Subject-grouped Train/Test Splitting (Zero Patient Leakage)...")
    train_df, test_df = split_clinical_by_subject(df_raw, test_size=0.25, random_state=42)
    print(f"Train subjects: {train_df['subject_id'].nunique()}, Segments: {len(train_df)}")
    print(f"Test subjects:  {test_df['subject_id'].nunique()}, Segments: {len(test_df)}")

    print("\nStep 3: Extracting structured features...")
    train_feats_df, X_train_struct, feat_names = extract_or_load_features(train_df)
    test_feats_df, X_test_struct, _ = extract_or_load_features(test_df)

    y_train = train_df["binary_target"].to_numpy(dtype=int)
    y_test = test_df["binary_target"].to_numpy(dtype=int)
    raw_train = list(train_df["raw_signal"].values)
    raw_test = list(test_df["raw_signal"].values)

    # Load unsupervised K-Means model to compute cluster features
    kmeans_bundle_path = os.path.join(MODELS_DIR, "unsupervised_kmeans.joblib")
    kmeans_pipeline = None
    train_cluster_dist = None
    test_cluster_dist = None
    if os.path.exists(kmeans_bundle_path):
        try:
            km_bundle = joblib.load(kmeans_bundle_path)
            kmeans_pipeline = km_bundle["model"]
            print("Loaded real-data K-Means model for hybrid representation.")
            # Use cluster soft affinities
            train_X_aug, aug_names = kmeans_pipeline.augment_features(train_feats_df[feat_names])
            test_X_aug, _ = kmeans_pipeline.augment_features(test_feats_df[feat_names])
            n_base = len(feat_names)
            train_cluster_dist = train_X_aug[:, n_base:]
            test_cluster_dist = test_X_aug[:, n_base:]
        except Exception as e:
            print(f"Notice: K-Means feature augmentation skipped ({e})")

    results = {}

    # Architecture 1: Structured-Only Baseline RF
    print("\nTraining Architecture 1: Structured-Only (75-D Physiological Features)...")
    clf_struct = MultimodalCVDClassifier(mode="structured_only", n_estimators=200, random_state=42)
    clf_struct.fit(X_structured=X_train_struct, y=y_train)
    y_pred_s = clf_struct.predict(X_structured=X_test_struct)
    y_prob_s = clf_struct.predict_proba(X_structured=X_test_struct)
    metrics_s = evaluate_predictions(y_test, y_pred_s, y_prob_s)
    results["Structured_Only"] = metrics_s
    print(f"Structured-Only Accuracy: {metrics_s['accuracy']:.4f}, F1: {metrics_s['f1_score']:.4f}, ROC-AUC: {metrics_s['roc_auc']:.4f}")

    # Architecture 2: Waveform-Only 1D CNN Encoder
    print("\nTraining Architecture 2: Waveform-Only Deep 1D CNN (32-D Latent Embeddings)...")
    clf_wave = MultimodalCVDClassifier(mode="waveform_only", n_estimators=200, random_state=42)
    clf_wave.fit(raw_signals=raw_train, y=y_train)
    y_pred_w = clf_wave.predict(raw_signals=raw_test)
    y_prob_w = clf_wave.predict_proba(raw_signals=raw_test)
    metrics_w = evaluate_predictions(y_test, y_pred_w, y_prob_w)
    results["Waveform_Only"] = metrics_w
    print(f"Waveform-Only Accuracy: {metrics_w['accuracy']:.4f}, F1: {metrics_w['f1_score']:.4f}, ROC-AUC: {metrics_w['roc_auc']:.4f}")

    # Architecture 3: Fused Deep + Structured Multi-Branch Classifier
    print("\nTraining Architecture 3: Fused Deep + Structured (107-D Multimodal Vector)...")
    clf_fused = MultimodalCVDClassifier(mode="fused", n_estimators=200, random_state=42)
    clf_fused.fit(X_structured=X_train_struct, raw_signals=raw_train, y=y_train)
    y_pred_f = clf_fused.predict(X_structured=X_test_struct, raw_signals=raw_test)
    y_prob_f = clf_fused.predict_proba(X_structured=X_test_struct, raw_signals=raw_test)
    metrics_f = evaluate_predictions(y_test, y_pred_f, y_prob_f)
    results["Fused_Multimodal"] = metrics_f
    print(f"Fused Multimodal Accuracy: {metrics_f['accuracy']:.4f}, F1: {metrics_f['f1_score']:.4f}, ROC-AUC: {metrics_f['roc_auc']:.4f}")

    # Architecture 4: Hybrid Fused + K-Means Classifier
    if train_cluster_dist is not None:
        print("\nTraining Architecture 4: Hybrid Fused + Unsupervised K-Means Clusters...")
        clf_hybrid = MultimodalCVDClassifier(mode="hybrid_kmeans", n_estimators=200, random_state=42)
        clf_hybrid.fit(
            X_structured=X_train_struct,
            raw_signals=raw_train,
            cluster_distances=train_cluster_dist,
            y=y_train,
        )
        y_pred_h = clf_hybrid.predict(
            X_structured=X_test_struct,
            raw_signals=raw_test,
            cluster_distances=test_cluster_dist,
        )
        y_prob_h = clf_hybrid.predict_proba(
            X_structured=X_test_struct,
            raw_signals=raw_test,
            cluster_distances=test_cluster_dist,
        )
        metrics_h = evaluate_predictions(y_test, y_pred_h, y_prob_h)
        results["Hybrid_KMeans_Fused"] = metrics_h
        print(f"Hybrid K-Means Fused Accuracy: {metrics_h['accuracy']:.4f}, F1: {metrics_h['f1_score']:.4f}, ROC-AUC: {metrics_h['roc_auc']:.4f}")

    # Save comprehensive evaluation report to JSON
    json_path = os.path.join(REPORTS_DIR, "clinical_model_evaluation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved clinical evaluation metrics to {json_path}")

    # Serialize Best Model Bundle
    best_model = clf_fused
    model_bundle = {
        "model": best_model,
        "feature_names": feat_names,
        "mode": "fused",
        "dataset": "PPG-BP Database (Guilin People's Hospital)",
        "population": "Guilin, China (External Clinical Benchmark - NOT Indian population)",
        "diagnoses": "CVD (Hypertension Stages 1-2, Stroke, Cerebrovascular) vs Healthy Normotensive Control",
        "held_out_metrics": metrics_f,
        "comparison_results": results,
        "clinical_disclaimer": (
            "Medical ML Safety Notice: This model is trained on the open Guilin Hospital benchmark "
            "to validate multi-branch structured + deep waveform fusion. It has not been clinically "
            "validated on Indian patient populations and is not a standalone diagnostic medical device."
        ),
    }
    bundle_path = os.path.join(MODELS_DIR, "clinical_cvd_model.joblib")
    joblib.dump(model_bundle, bundle_path)
    print(f"Saved trained clinical model bundle to {bundle_path}")

    # Generate Evaluation Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Model Comparison Bar Chart
    models = list(results.keys())
    accs = [results[m]["accuracy"] for m in models]
    f1s = [results[m]["f1_score"] for m in models]
    rocs = [results[m]["roc_auc"] for m in models]

    x = np.arange(len(models))
    width = 0.25
    axes[0].bar(x - width, accs, width, label="Accuracy", color="#1f77b4")
    axes[0].bar(x, f1s, width, label="F1-Score", color="#2ca02c")
    axes[0].bar(x + width, rocs, width, label="ROC-AUC", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylim([0.5, 1.05])
    axes[0].set_ylabel("Score")
    axes[0].set_title("Clinical Model Architecture Comparison (Held-Out Test Subjects)", fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Right: Fused Model Confusion Matrix
    cm = np.array(metrics_f["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[1],
        xticklabels=["Healthy Control", "CVD Patient"],
        yticklabels=["Healthy Control", "CVD Patient"],
    )
    axes[1].set_title(f"Fused Model Confusion Matrix (Acc={metrics_f['accuracy']:.1%})", fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Clinical EMR Diagnosis")

    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "clinical_evaluation_metrics.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved evaluation figure to {plot_path}")

if __name__ == "__main__":
    main()
