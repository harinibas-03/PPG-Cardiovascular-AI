"""
Unsupervised Training Script: PPG Morphological Clustering
==========================================================
Trains K-Means clustering and PCA on exploratory PPG data (e.g., Spandan-1M subset).

IMPORTANT:
This step discovers unsupervised pulse morphology groups. It DOES NOT classify
cardiovascular disease or infer coronary artery disease.

Usage:
    py scripts/train_unsupervised.py --input data/raw/spandan_subset.json --k 4 --output models/unsupervised_kmeans.joblib
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arohan_ppg_cvd.adapters.spandan_adapter import SpandanAdapter
from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor
from arohan_ppg_cvd.models.unsupervised import PPGUnsupervisedClusterer
from arohan_ppg_cvd.models.hybrid_pipeline import HybridPPGClassifier
from arohan_ppg_cvd.models.persistence import save_model_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_unsupervised")


def main():
    parser = argparse.ArgumentParser(description="Train unsupervised K-Means on PPG features.")
    parser.add_argument("--input", type=str, default="data/sample_ppg.csv", help="Input JSON or CSV file")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters (default: 4)")
    parser.add_argument("--pca-components", type=int, default=6, help="PCA components for projection")
    parser.add_argument("--output", type=str, default="models/unsupervised_kmeans.joblib", help="Output model path")
    args = parser.parse_args()

    logger.info(f"Loading data from: {args.input}")
    extractor = PPGFeatureExtractor(default_fs=30.0)

    # Detect file type
    if args.input.endswith(".json"):
        records = SpandanAdapter.load_from_json(args.input)
        df_features = extractor.extract_batch(records)
    elif args.input.endswith(".csv"):
        df_raw = pd.read_csv(args.input)
        # Check if already feature table or raw signal table
        if "hr_bpm" in df_raw.columns:
            df_features = df_raw
        elif "signal" in df_raw.columns or "ppg" in df_raw.columns:
            # Group by file_id if present
            if "file_id" in df_raw.columns:
                records = []
                for fid, grp in df_raw.groupby("file_id"):
                    sig_col = "signal" if "signal" in grp.columns else "ppg"
                    records.append({
                        "file_id": str(fid),
                        "signal": grp[sig_col].to_numpy(dtype=np.float32),
                        "sampling_rate": float(grp["sampling_rate"].iloc[0]) if "sampling_rate" in grp.columns else 30.0,
                    })
                df_features = extractor.extract_batch(records)
            else:
                sig_col = "signal" if "signal" in df_raw.columns else "ppg"
                feat = extractor.extract_from_signal(df_raw[sig_col].to_numpy(dtype=np.float32), fs=30.0)
                df_features = pd.DataFrame([feat]) if feat else pd.DataFrame()
        else:
            # Assume structured features
            df_features = df_raw
    else:
        raise ValueError(f"Unsupported format: {args.input}. Must be .json or .csv")

    logger.info(f"Extracted features for {len(df_features)} samples.")
    if len(df_features) < args.k:
        raise ValueError(f"Insufficient samples ({len(df_features)}) for k={args.k} clusters.")

    # Evaluate multiple k values
    logger.info("Evaluating clustering across k=2..6...")
    meta_cols = {"file_id", "patient_id", "label", "target"}
    feat_cols = [c for c in df_features.columns if c not in meta_cols]
    eval_results = PPGUnsupervisedClusterer.evaluate_clustering(df_features[feat_cols], k_range=(2, min(6, len(df_features)-1)))
    for res in eval_results:
        logger.info(f"k={res['k']}: Inertia={res['inertia']:.2f}, Silhouette={res['silhouette']:.4f}")

    # Build hybrid container with unsupervised stage
    hybrid = HybridPPGClassifier(
        n_clusters=args.k,
        use_pca=True,
        n_pca_components=args.pca_components,
    )
    hybrid.fit_unsupervised(df_features)

    # Save
    save_path = save_model_bundle(
        model=hybrid,
        save_path=args.output,
        metadata={
            "description": "Exploratory K-Means PPG morphology model",
            "k": args.k,
            "samples_count": len(df_features),
            "is_supervised_trained": False,
        },
    )
    logger.info(f"Model saved to {save_path}")
    logger.info("Notice: This unsupervised model is ready to provide cluster features to the supervised pipeline.")


if __name__ == "__main__":
    main()
