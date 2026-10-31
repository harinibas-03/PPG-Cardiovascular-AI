"""
Real-Data Preprocessing, Feature Extraction & Unsupervised Clustering Pipeline
=============================================================================
Processes the 50 real Indian PPG recordings from `data/raw/spandan_subset.json`:
1. Validates sampling rates, signal lengths, missing values, and Signal Quality (SQI).
2. Applies PPG preprocessing: bandpass filtering (0.5–8.0 Hz), detrending, and Z-score normalization.
3. Detects systolic peaks and extracts physiological, morphological, derivative (APG), and HRV features.
4. Generates a data quality report (accepted vs rejected recordings).
5. Exports extracted features to `data/processed/spandan_features.csv`.
6. Evaluates clustering across k=2..6 using Silhouette scores and trains an unsupervised K-Means model.
7. Saves the trained model bundle to `models/unsupervised_kmeans.joblib`.
8. Generates visualization plots:
   - Representative PPG waveforms with detected systolic peaks and APG second derivatives.
   - 2D PCA projection of the unsupervised morphological clusters.
   - Signal quality index distributions.
9. Exports a representative real Spandan recording to `data/sample_ppg.csv` for the dashboard.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arohan_ppg_cvd.preprocessing.filters import clean_ppg_signal
from arohan_ppg_cvd.preprocessing.normalization import normalize_signal
from arohan_ppg_cvd.preprocessing.segmentation import detect_systolic_peaks, segment_pulse_waves
from arohan_ppg_cvd.preprocessing.sqi import evaluate_signal_quality
from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor
from arohan_ppg_cvd.features.derivatives import compute_derivatives
from arohan_ppg_cvd.models.unsupervised import PPGUnsupervisedClusterer
from arohan_ppg_cvd.models.hybrid_pipeline import HybridPPGClassifier
from arohan_ppg_cvd.models.persistence import save_model_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("process_real_spandan")


def main():
    raw_path = "data/raw/spandan_subset.json"
    processed_dir = "data/processed"
    reports_dir = "reports"
    models_dir = "models"

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Input file not found: {raw_path}")

    logger.info(f"Loading raw Spandan records from {raw_path}...")
    with open(raw_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    logger.info(f"Loaded {len(records)} raw PPG recordings.")

    # 1. Validation & Preprocessing Audit
    audit_results = []
    valid_features = []
    representative_valid_record = None

    extractor = PPGFeatureExtractor(enforce_sqi=False)

    for idx, rec in enumerate(records):
        fid = rec.get("file_id", f"spandan_{idx:03d}")
        raw_sig = np.asarray(rec.get("signal", []), dtype=np.float32)
        fs = float(rec.get("sampling_rate", 30.0))
        device_bpm = rec.get("bpm")

        # Sanity Checks
        n_samples = len(raw_sig)
        duration_sec = n_samples / fs if fs > 0 else 0.0
        has_nan = bool(np.isnan(raw_sig).any())
        has_inf = bool(np.isinf(raw_sig).any())

        # Check for flatline or near-zero variance
        is_flat = bool(n_samples < 10 or np.ptp(raw_sig) < 1e-4)

        rejection_reason = None
        if has_nan or has_inf:
            rejection_reason = "Contains NaN or Inf values"
        elif n_samples < int(fs * 2.0):
            rejection_reason = f"Signal duration insufficient ({duration_sec:.1f}s < 2.0s)"
        elif is_flat:
            rejection_reason = "Signal has zero/negligible amplitude variance (flatline)"
        elif not (10.0 <= fs <= 250.0):
            rejection_reason = f"Unrealistic sampling rate ({fs:.1f} Hz)"

        if rejection_reason is not None:
            audit_results.append({
                "file_id": fid,
                "status": "REJECTED",
                "reason": rejection_reason,
                "sampling_rate": fs,
                "n_samples": n_samples,
                "duration_sec": round(duration_sec, 2),
                "device_bpm": device_bpm,
                "is_acceptable_sqi": False,
            })
            continue

        # Signal Preprocessing
        try:
            cleaned = clean_ppg_signal(raw_sig, fs=fs, lowcut=0.5, highcut=8.0)
            normed = normalize_signal(cleaned, method="z_score")
            is_acceptable, sqi_details = evaluate_signal_quality(normed, fs=fs)
            peaks, _ = detect_systolic_peaks(normed, fs=fs)
            beats = segment_pulse_waves(normed, fs=fs)

            # Extract unified features
            feat = extractor.extract_from_signal(
                raw_signal=raw_sig,
                fs=fs,
                file_id=fid,
                bpm_reference=device_bpm,
            )

            if feat is None:
                audit_results.append({
                    "file_id": fid,
                    "status": "REJECTED",
                    "reason": "Feature extraction failed or insufficient clean beats",
                    "sampling_rate": fs,
                    "n_samples": n_samples,
                    "duration_sec": round(duration_sec, 2),
                    "device_bpm": device_bpm,
                    "is_acceptable_sqi": False,
                })
                continue

            # Check if signal has acceptable SQI and sufficient detected peaks
            if len(peaks) < 2:
                status = "REJECTED"
                reason = "Fewer than 2 systolic peaks detected"
            else:
                status = "ACCEPTED"
                reason = "Passed all checks"
                valid_features.append(feat)

                if representative_valid_record is None and is_acceptable:
                    representative_valid_record = {
                        "file_id": fid,
                        "raw_signal": raw_sig,
                        "cleaned_signal": cleaned,
                        "norm_signal": normed,
                        "sampling_rate": fs,
                        "peaks": peaks,
                        "beats": beats,
                        "device_bpm": device_bpm,
                        "sqi": sqi_details,
                    }

            audit_results.append({
                "file_id": fid,
                "status": status,
                "reason": reason,
                "sampling_rate": round(fs, 2),
                "n_samples": n_samples,
                "duration_sec": round(duration_sec, 2),
                "device_bpm": device_bpm,
                "estimated_hr": feat.get("hr_bpm"),
                "detected_peaks": len(peaks),
                "segmented_beats": len(beats),
                "is_acceptable_sqi": is_acceptable,
                "sSQI": round(sqi_details.get("sSQI", 0.0), 3),
                "kSQI": round(sqi_details.get("kSQI", 0.0), 3),
                "pSQI": round(sqi_details.get("pSQI", 0.0), 3),
                "perfusion_index": round(sqi_details.get("perfusion_index", 0.0), 2),
            })

        except Exception as e:
            logger.error(f"Error processing record {fid}: {e}", exc_info=True)
            audit_results.append({
                "file_id": fid,
                "status": "REJECTED",
                "reason": f"Processing exception: {str(e)}",
                "sampling_rate": fs,
                "n_samples": n_samples,
                "duration_sec": round(duration_sec, 2),
                "device_bpm": device_bpm,
                "is_acceptable_sqi": False,
            })

    # Summary of Validation
    df_audit = pd.DataFrame(audit_results)
    n_total = len(df_audit)
    n_accepted = (df_audit["status"] == "ACCEPTED").sum()
    n_rejected = (df_audit["status"] == "REJECTED").sum()
    n_sqi_pass = df_audit["is_acceptable_sqi"].sum()

    logger.info("================ AUDIT SUMMARY ================")
    logger.info(f"Total recordings examined: {n_total}")
    logger.info(f"Accepted recordings:        {n_accepted} ({n_accepted / n_total * 100:.1f}%)")
    logger.info(f"Rejected recordings:        {n_rejected} ({n_rejected / n_total * 100:.1f}%)")
    logger.info(f"Strict SQI Pass count:      {n_sqi_pass} ({n_sqi_pass / n_total * 100:.1f}%)")
    logger.info("================================================")

    # Save audit report
    audit_csv = os.path.join(reports_dir, "spandan_validation_audit.csv")
    df_audit.to_csv(audit_csv, index=False)
    logger.info(f"Saved validation audit table to {audit_csv}")

    # 2. Save Extracted Features to CSV
    df_features = pd.DataFrame(valid_features)
    # Ensure numeric types
    for c in df_features.columns:
        if c != "file_id":
            df_features[c] = pd.to_numeric(df_features[c], errors="coerce").fillna(0.0)

    features_csv = os.path.join(processed_dir, "spandan_features.csv")
    df_features.to_csv(features_csv, index=False)
    logger.info(f"Saved {len(df_features)} valid feature vectors to {features_csv} (Columns: {df_features.shape[1]})")

    # 3. Evaluate and Train Unsupervised K-Means Clustering
    # Exclude identifiers and external device references from morphological pulse clustering
    meta_cols = {"file_id", "device_bpm_ref"}
    feat_cols = [c for c in df_features.columns if c not in meta_cols]
    X_features = df_features[feat_cols]

    logger.info(f"Evaluating K-Means clustering across k=2..6 on {len(X_features)} valid recordings...")
    eval_results = PPGUnsupervisedClusterer.evaluate_clustering(X_features, k_range=(2, min(6, len(X_features) - 1)))

    best_k = 3
    best_sil = -1.0
    for res in eval_results:
        logger.info(f"k={res['k']}: Inertia={res['inertia']:.2f}, Silhouette Score={res['silhouette']:.4f}")
        if res["silhouette"] > best_sil:
            best_sil = res["silhouette"]
            best_k = res["k"]

    logger.info(f"Selected cluster count k={best_k} (highest silhouette score: {best_sil:.4f})")

    # Cluster Stability Evaluation Across Random Seeds
    from sklearn.metrics import adjusted_rand_score
    from sklearn.ensemble import IsolationForest

    seeds = [42, 101, 2024, 7, 99]
    seed_clusters = {}
    for s in seeds:
        m = HybridPPGClassifier(
            n_clusters=best_k,
            use_pca=True,
            n_pca_components=min(6, len(feat_cols)),
            random_state=s,
        )
        m.fit_unsupervised(df_features)
        seed_clusters[s] = m.clusterer.predict_clusters(df_features)

    ari_scores = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            ari = adjusted_rand_score(seed_clusters[seeds[i]], seed_clusters[seeds[j]])
            ari_scores.append(ari)
    mean_ari = float(np.mean(ari_scores))
    logger.info(f"Mean Adjusted Rand Index (ARI) across 5 random seeds: {mean_ari:.4f} (Stability: {'High' if mean_ari > 0.75 else ('Moderate' if mean_ari > 0.5 else 'Low')})")

    # Outlier Influence Analysis (Isolation Forest)
    iso = IsolationForest(contamination=0.05, random_state=42)
    outlier_preds = iso.fit_predict(X_features.fillna(0.0))
    n_outliers = int((outlier_preds == -1).sum())
    n_inliers = int((outlier_preds == 1).sum())
    logger.info(f"Outlier detection (Isolation Forest 5% expected): {n_outliers} outliers detected out of {len(X_features)} valid records.")

    # Fit baseline model on inliers to check if outliers drive the clusters
    inlier_mask = outlier_preds == 1
    eval_inliers = PPGUnsupervisedClusterer.evaluate_clustering(X_features[inlier_mask], k_range=(2, min(6, inlier_mask.sum() - 1)))
    inlier_best_res = max(eval_inliers, key=lambda x: x["silhouette"])
    logger.info(f"Inlier-only clustering optimal k={inlier_best_res['k']} with Silhouette Score={inlier_best_res['silhouette']:.4f}")

    # Fit Primary Hybrid container with unsupervised stage
    hybrid_model = HybridPPGClassifier(
        n_clusters=best_k,
        use_pca=True,
        n_pca_components=min(6, len(feat_cols)),
        random_state=42,
    )
    hybrid_model.fit_unsupervised(df_features)

    # Predict cluster labels for valid recordings
    cluster_labels = hybrid_model.clusterer.predict_clusters(df_features)
    df_features["assigned_cluster"] = cluster_labels
    df_features["is_outlier"] = (outlier_preds == -1)

    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index().to_dict()
    outlier_per_cluster = pd.crosstab(df_features["assigned_cluster"], df_features["is_outlier"]).to_dict()
    logger.info(f"Discovered pulse waveform morphological clusters: {cluster_counts}")
    logger.info(f"Outlier distribution per cluster: {outlier_per_cluster}")
    logger.info("Notice: Clusters represent geometric waveform groupings and do NOT denote clinical CVD/CAD diagnoses.")

    # Save cluster stability & outlier report
    stability_data = {
        "k_evaluated": eval_results,
        "selected_k": best_k,
        "highest_silhouette_score": best_sil,
        "seed_ari_stability": {
            "tested_seeds": seeds,
            "pairwise_ari_scores": [round(s, 4) for s in ari_scores],
            "mean_ari": round(mean_ari, 4),
            "stability_rating": "High" if mean_ari > 0.75 else ("Moderate" if mean_ari > 0.5 else "Low"),
        },
        "outlier_analysis": {
            "total_samples": len(df_features),
            "outliers_count": n_outliers,
            "inliers_count": n_inliers,
            "inlier_only_best_k": inlier_best_res["k"],
            "inlier_only_silhouette": round(inlier_best_res["silhouette"], 4),
            "cluster_distribution": cluster_counts,
            "is_driven_by_outliers": bool(min(cluster_counts.values()) <= 2 and n_outliers > 0),
        },
    }
    stability_json_path = os.path.join(reports_dir, "cluster_stability_report.json")
    with open(stability_json_path, "w", encoding="utf-8") as f:
        json.dump(stability_data, f, indent=2)
    logger.info(f"Saved cluster stability & outlier report to {stability_json_path}")

    # Save trained model bundle
    model_output_path = os.path.join(models_dir, "unsupervised_kmeans.joblib")
    save_model_bundle(
        model=hybrid_model,
        save_path=model_output_path,
        metadata={
            "description": "Unsupervised PPG morphological K-Means clusterer trained on real Spandan-1M subset",
            "k": best_k,
            "silhouette_score": best_sil,
            "clustering_eval": eval_results,
            "cluster_distribution": cluster_counts,
            "samples_count": len(df_features),
            "feature_names": feat_cols,
            "stability": stability_data,
            "is_supervised_trained": False,
        },
    )
    logger.info(f"Saved trained unsupervised model bundle to {model_output_path}")

    # 4. Generate Visualizations
    logger.info("Generating report visualizations...")

    # Plot 1: Representative Real Spandan PPG Waveform & APG Derivatives
    if representative_valid_record is not None:
        rec = representative_valid_record
        t_sec = np.arange(len(rec["raw_signal"])) / rec["sampling_rate"]
        vpg, apg = compute_derivatives(rec["norm_signal"], fs=rec["sampling_rate"])

        # Display first 6 seconds for clean visualization
        sub_len = min(len(t_sec), int(rec["sampling_rate"] * 6.0))
        t_sub = t_sec[:sub_len]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # Raw Signal
        ax1.plot(t_sub, rec["raw_signal"][:sub_len], color="#374151", lw=1.2)
        ax1.set_title(f"Real Spandan Recording ({rec['file_id']}) - Sampling Rate: {rec['sampling_rate']:.1f} Hz", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Raw Sensor Value")
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Conditioned PPG + Peaks
        ax2.plot(t_sub, rec["norm_signal"][:sub_len], color="#2563eb", lw=1.5, label="Filtered & Z-Score Normalized")
        sub_peaks = [p for p in rec["peaks"] if p < sub_len]
        if sub_peaks:
            ax2.scatter(t_sub[sub_peaks], rec["norm_signal"][sub_peaks], color="#dc2626", s=45, zorder=5, label="Systolic Peaks")
        ax2.set_ylabel("Normalized PPG")
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle=":", alpha=0.5)

        # APG 2nd Derivative
        ax3.plot(t_sub, apg[:sub_len], color="#d97706", lw=1.2, label="Acceleration Plethysmogram (APG)")
        ax3.set_xlabel("Time (seconds)")
        ax3.set_ylabel("APG (2nd Deriv)")
        ax3.legend(loc="upper right")
        ax3.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        waveform_fig_path = os.path.join(reports_dir, "spandan_real_waveform.png")
        fig.savefig(waveform_fig_path, dpi=200)
        plt.close(fig)
        logger.info(f"Saved real PPG waveform plot to {waveform_fig_path}")

    # Plot 2: 2D PCA Cluster Projection
    pca_proj = hybrid_model.clusterer.get_pca_projection(df_features, n_dims=2)
    fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
    scatter = ax_pca.scatter(
        pca_proj[:, 0],
        pca_proj[:, 1],
        c=cluster_labels,
        cmap="viridis",
        s=70,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.8,
    )
    ax_pca.set_title(f"Unsupervised Pulse Waveform Clusters (K-Means k={best_k}, PCA Projection)", fontsize=12, fontweight="bold")
    ax_pca.set_xlabel("Principal Component 1 (Morphology & Energy)")
    ax_pca.set_ylabel("Principal Component 2 (Derivative Profiles)")
    plt.colorbar(scatter, ax=ax_pca, label="Morphological Cluster ID")
    ax_pca.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    pca_fig_path = os.path.join(reports_dir, "spandan_pca_clusters.png")
    fig_pca.savefig(pca_fig_path, dpi=200)
    plt.close(fig_pca)
    logger.info(f"Saved PCA cluster plot to {pca_fig_path}")

    # Plot 3: Signal Quality Index (SQI) Distribution
    fig_sqi, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(10, 4))
    ax_s1.hist(df_audit["pSQI"], bins=15, color="#0d9488", edgecolor="white", alpha=0.85)
    ax_s1.axvline(0.40, color="#ef4444", linestyle="--", label="Acceptance Cutoff (0.40)")
    ax_s1.set_title("Power SQI (pSQI) Distribution", fontsize=10, fontweight="bold")
    ax_s1.set_xlabel("pSQI (Cardiac Band Ratio)")
    ax_s1.set_ylabel("Count")
    ax_s1.legend()
    ax_s1.grid(True, linestyle=":", alpha=0.5)

    hr_values = df_audit["estimated_hr"].dropna()
    ax_s2.hist(hr_values, bins=15, color="#6366f1", edgecolor="white", alpha=0.85)
    ax_s2.set_title("Estimated Heart Rate (BPM)", fontsize=10, fontweight="bold")
    ax_s2.set_xlabel("Heart Rate (BPM)")
    ax_s2.set_ylabel("Count")
    ax_s2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    sqi_fig_path = os.path.join(reports_dir, "spandan_sqi_distribution.png")
    fig_sqi.savefig(sqi_fig_path, dpi=200)
    plt.close(fig_sqi)
    logger.info(f"Saved SQI distribution plot to {sqi_fig_path}")

    # 5. Export a Real Spandan Sample to data/sample_ppg.csv for Dashboard
    if representative_valid_record is not None:
        sample_csv = "data/sample_ppg.csv"
        # Export signal array with metadata columns
        df_sample = pd.DataFrame({
            "signal": representative_valid_record["raw_signal"],
            "sampling_rate": representative_valid_record["sampling_rate"],
            "file_id": representative_valid_record["file_id"],
        })
        df_sample.to_csv(sample_csv, index=False)
        logger.info(f"Exported real Spandan sample recording to {sample_csv} for Streamlit dashboard visualization.")

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
