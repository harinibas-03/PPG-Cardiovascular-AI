"""
Synthetic PPG Dataset Generator
===============================
Generates 1,000 physiologically plausible synthetic PPG recordings across 100 subjects
belonging to two distinct morphological classes:
- Class 0: Synthetic Pattern A (sharp systolic upstroke, distinct dicrotic notch, normocardic rhythm)
- Class 1: Synthetic Pattern B (prolonged systolic rise, blunted/attenuated dicrotic wave, altered morphology)

Includes subject IDs to enforce strictly leakage-free subject-level splitting during supervised evaluation.

DISCLAIMER:
These classes represent simulated pulse-contour patterns for demonstrating hybrid machine
learning pipelines. They DO NOT represent clinically confirmed Healthy or CAD diagnoses.
"""

import os
import sys
import json
import logging
import argparse
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_synthetic")


def generate_single_pulse_wave(
    t: np.ndarray,
    bpm: float,
    pattern: str = "A",
    amplitude: float = 1.0,
    noise_level: float = 0.02,
    wander_amp: float = 0.15,
    wander_freq: float = 0.25,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Generate a realistic synthetic PPG time-series with physiological pulse contour.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    beat_freq = bpm / 60.0
    sig = np.zeros_like(t)

    if pattern == "A":
        # Pattern A: Sharp systolic rise, distinct dicrotic notch and diastolic rebound
        systolic_center = 0.18
        systolic_width = 0.007
        diastolic_center = 0.42
        diastolic_width = 0.016
        diastolic_ratio = 0.38
    else:
        # Pattern B: Slower systolic rise (prolonged crest time), blunted/attenuated dicrotic notch
        systolic_center = 0.28
        systolic_width = 0.014
        diastolic_center = 0.50
        diastolic_width = 0.022
        diastolic_ratio = 0.18

    # Synthesize cardiac cycles
    for i, ti in enumerate(t):
        phase = (ti * beat_freq) % 1.0
        # Systolic component
        systolic = np.exp(-((phase - systolic_center) ** 2) / systolic_width)
        # Diastolic reflection
        diastolic = diastolic_ratio * np.exp(-((phase - diastolic_center) ** 2) / diastolic_width)
        sig[i] = amplitude * (systolic + diastolic)

    # Respiratory baseline wander
    respiration = wander_amp * np.sin(2 * np.pi * wander_freq * t + rng.uniform(0, 2 * np.pi))
    # Gaussian measurement noise
    noise = rng.normal(0, noise_level, size=len(t))

    return (sig + respiration + noise).astype(np.float32)


def generate_dataset(
    n_subjects: int = 100,
    records_per_subject: int = 10,
    fs: float = 30.0,
    duration_sec: float = 6.0,
    output_dir: str = "data/synthetic",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate 1,000 synthetic recordings across 100 subjects (50 Pattern A, 50 Pattern B).
    """
    rng = np.random.RandomState(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    records = []
    metadata = []
    t = np.linspace(0, duration_sec, int(fs * duration_sec))

    logger.info(f"Generating {n_subjects * records_per_subject} synthetic PPG recordings...")

    # Assign 50 subjects to Class 0 (Pattern A) and 50 to Class 1 (Pattern B)
    for sub_idx in range(n_subjects):
        sub_id = f"sub_{sub_idx:03d}"
        class_label = 0 if sub_idx < (n_subjects // 2) else 1
        pattern_name = "Pattern A" if class_label == 0 else "Pattern B"

        # Subject-specific baseline characteristics
        if class_label == 0:
            base_bpm = rng.normal(70.0, 5.0)
            base_amp = rng.normal(1.15, 0.08)
        else:
            base_bpm = rng.normal(84.0, 6.0)
            base_amp = rng.normal(0.95, 0.08)

        base_bpm = np.clip(base_bpm, 55.0, 110.0)
        base_amp = np.clip(base_amp, 0.6, 1.8)

        for rec_idx in range(records_per_subject):
            rec_id = f"{sub_id}_rec_{rec_idx:02d}"
            # Recording-specific variation
            rec_bpm = base_bpm + rng.normal(0, 2.0)
            rec_amp = base_amp + rng.normal(0, 0.04)
            noise_lvl = rng.uniform(0.015, 0.035)
            wander_amp = rng.uniform(0.08, 0.20)
            wander_freq = rng.uniform(0.20, 0.35)

            sig = generate_single_pulse_wave(
                t=t,
                bpm=rec_bpm,
                pattern="A" if class_label == 0 else "B",
                amplitude=rec_amp,
                noise_level=noise_lvl,
                wander_amp=wander_amp,
                wander_freq=wander_freq,
                rng=rng,
            )

            records.append({
                "record_id": rec_id,
                "subject_id": sub_id,
                "label": int(class_label),
                "pattern": pattern_name,
                "sampling_rate": float(fs),
                "duration_sec": float(duration_sec),
                "bpm_simulated": round(float(rec_bpm), 2),
                "signal": sig.tolist(),
            })

            metadata.append({
                "record_id": rec_id,
                "subject_id": sub_id,
                "label": int(class_label),
                "pattern": pattern_name,
                "sampling_rate": float(fs),
                "duration_sec": float(duration_sec),
                "bpm_simulated": round(float(rec_bpm), 2),
                "amplitude_simulated": round(float(rec_amp), 3),
            })

    # Save JSON raw signals
    json_path = os.path.join(output_dir, "synthetic_ppg_records.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    logger.info(f"Saved {len(records)} raw synthetic records to {json_path}")

    # Save metadata CSV
    df_meta = pd.DataFrame(metadata)
    meta_path = os.path.join(output_dir, "synthetic_metadata.csv")
    df_meta.to_csv(meta_path, index=False)
    logger.info(f"Saved metadata to {meta_path}")

    # Extract 75-dimensional physiological feature vectors using existing extractor
    logger.info("Extracting physiological feature vectors using PPGFeatureExtractor...")
    extractor = PPGFeatureExtractor(default_fs=fs)
    feat_rows = []

    for r in records:
        sig_arr = np.array(r["signal"], dtype=np.float32)
        f_dict = extractor.extract_from_signal(
            raw_signal=sig_arr,
            fs=r["sampling_rate"],
            file_id=r["record_id"],
            bpm_reference=r["bpm_simulated"],
        )
        if f_dict is not None:
            f_dict["subject_id"] = r["subject_id"]
            f_dict["label"] = r["label"]
            f_dict["pattern"] = r["pattern"]
            feat_rows.append(f_dict)

    df_feats = pd.DataFrame(feat_rows)
    feat_path = os.path.join(output_dir, "synthetic_features.csv")
    df_feats.to_csv(feat_path, index=False)
    logger.info(f"Extracted and saved {len(df_feats)} feature vectors to {feat_path} (shape: {df_feats.shape})")

    return df_meta, df_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic PPG benchmark dataset.")
    parser.add_argument("--subjects", type=int, default=100, help="Number of synthetic subjects")
    parser.add_argument("--records-per-subject", type=int, default=10, help="Records per subject")
    parser.add_argument("--fs", type=float, default=30.0, help="Sampling frequency (Hz)")
    parser.add_argument("--duration", type=float, default=6.0, help="Signal duration in seconds")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_dataset(
        n_subjects=args.subjects,
        records_per_subject=args.records_per_subject,
        fs=args.fs,
        duration_sec=args.duration,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )
