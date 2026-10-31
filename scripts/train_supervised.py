"""
Supervised Training Script: Hybrid CVD / CAD Classifier
=======================================================
Trains the Hybrid (K-Means + Random Forest) model on a CONFIRMED MEDICALLY LABELED dataset.

SAFETY NOTICE:
Do not run this script on Spandan-1M raw data or datasets lacking verified clinical diagnoses.
The script requires a labeled target column (e.g. 'label' or 'diagnosis': 0 for Healthy, 1 for CVD/CAD)
and a patient/recording identifier column to strictly prevent intra-patient data leakage.

Usage:
    py scripts/train_supervised.py --data data/labeled_clinical_ppg.csv --target-col label --patient-col patient_id --output models/arohan_hybrid_cvd_model.joblib
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor
from arohan_ppg_cvd.models.hybrid_pipeline import HybridPPGClassifier
from arohan_ppg_cvd.models.persistence import save_model_bundle
from arohan_ppg_cvd.evaluation.metrics import compute_classification_metrics, plot_evaluation_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_supervised")


def main():
    parser = argparse.ArgumentParser(description="Train hybrid supervised classifier on confirmed medical labels.")
    parser.add_argument("--data", type=str, required=True, help="Path to medically labeled CSV or Parquet file")
    parser.add_argument("--target-col", type=str, default="label", help="Ground truth column (0: Healthy, 1: CVD/CAD)")
    parser.add_argument("--patient-col", type=str, default="patient_id", help="Patient ID column for leakage-free splitting")
    parser.add_argument("--n-clusters", type=int, default=4, help="K-Means clusters for unsupervised stage")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold cross-validation splits")
    parser.add_argument("--output", type=str, default="models/arohan_hybrid_cvd_model.joblib", help="Output model path")
    parser.add_argument("--eval-plot", type=str, default="reports/roc_pr_curve.png", help="Path to save evaluation plot")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        logger.error(f"Dataset file not found: {args.data}")
        logger.error("A confirmed medically labeled dataset is required to train the supervised classifier.")
        sys.exit(1)

    logger.info(f"Loading medically labeled dataset: {args.data}")
    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)

    if args.target_col not in df.columns:
        raise ValueError(
            f"Target column '{args.target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}. "
            "Please ensure you provide verified medical diagnosis labels (0: Healthy, 1: CVD/CAD)."
        )

    # Check class distribution
    y = df[args.target_col].to_numpy()
    unique_classes, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique_classes, counts))}")

    if len(unique_classes) < 2:
        raise ValueError("Dataset must contain both positive (CVD) and negative (Healthy) classes.")

    # Patient identifiers to prevent data leakage
    if args.patient_col in df.columns:
        groups = df[args.patient_col].to_numpy()
        logger.info(f"Patient grouping column: '{args.patient_col}' ({len(np.unique(groups))} unique patients)")
    else:
        logger.warning(
            f"Patient column '{args.patient_col}' not found. Using individual sample indices. "
            "WARNING: May lead to data leakage if multiple beats originate from the same patient."
        )
        groups = np.arange(len(df))

    # Feature extraction if input contains raw PPG
    if "signal" in df.columns or "ppg" in df.columns:
        logger.info("Extracting features from raw signal column...")
        extractor = PPGFeatureExtractor()
        sig_col = "signal" if "signal" in df.columns else "ppg"
        records = []
        for idx, row in df.iterrows():
            sig = row[sig_col]
            if isinstance(sig, str):
                sig = json.loads(sig)
            records.append({
                "signal": sig,
                "file_id": str(row.get(args.patient_col, idx)),
                "sampling_rate": float(row.get("sampling_rate", 30.0)),
            })
        df_feats = extractor.extract_batch(records)
        df_feats[args.target_col] = df[args.target_col].values[:len(df_feats)]
        df_feats["patient_id"] = groups[:len(df_feats)]
        df = df_feats
        y = df[args.target_col].to_numpy()
        groups = df["patient_id"].to_numpy()

    meta_cols = {args.target_col, args.patient_col, "file_id", "patient_id"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols]

    logger.info(f"Feature matrix shape: {X.shape} with {len(feat_cols)} engineered features.")

    # Initialize Hybrid Pipeline
    hybrid = HybridPPGClassifier(
        n_clusters=args.n_clusters,
        use_pca=True,
        n_pca_components=min(6, len(feat_cols)),
        n_estimators=150,
        class_weight="balanced",
    )

    logger.info(f"Training hybrid pipeline with {args.n_splits}-fold Patient GroupKFold...")
    cv_summary = hybrid.fit_supervised(X, y, groups=groups, cv_splits=args.n_splits)

    logger.info("===== Cross-Validation Performance =====")
    logger.info(f"Mean ROC-AUC:    {cv_summary.get('mean_roc_auc', 0.0):.4f} +/- {cv_summary.get('std_roc_auc', 0.0):.4f}")
    logger.info(f"Mean F1-Score:   {cv_summary.get('mean_f1', 0.0):.4f} +/- {cv_summary.get('std_f1', 0.0):.4f}")
    logger.info(f"Mean Sensitivity:{cv_summary.get('mean_sensitivity', 0.0):.4f}")
    logger.info(f"Mean Specificity:{cv_summary.get('mean_precision', 0.0):.4f}")

    # Feature importances
    top_feats = hybrid.classifier.get_feature_importances().head(10)
    logger.info("\nTop 10 Important Features:\n" + top_feats.to_string(index=False))

    # Save model bundle
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_model_bundle(
        model=hybrid,
        save_path=args.output,
        metadata={
            "dataset_file": args.data,
            "target_col": args.target_col,
            "patient_col": args.patient_col,
            "cv_metrics": cv_summary,
            "top_features": top_feats.to_dict(orient="records"),
        },
    )

    # Generate evaluation plots
    if args.eval_plot:
        os.makedirs(os.path.dirname(os.path.abspath(args.eval_plot)), exist_ok=True)
        probs = hybrid.predict_proba(X)[:, 1]
        fig = plot_evaluation_curves(y, probs, output_path=args.eval_plot)
        logger.info(f"Evaluation curves saved to: {args.eval_plot}")

    logger.info(f"Model successfully saved to {args.output}")


if __name__ == "__main__":
    main()
