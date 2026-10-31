"""
Arohan_PPG_CVD - Interactive Clinical AI Research Dashboard
============================================================
Streamlit application for optical photoplethysmogram (PPG) inspection,
signal conditioning, morphological feature extraction, exploratory clustering,
and supervised CVD / CAD screening inference.

Includes two dedicated operational modes:
1. 🫀 Clinical PPG Research (Real Spandan recordings & user uploads with exploratory K-Means clustering)
2. 🧪 Synthetic Hybrid AI Demo (Proof-of-concept pipeline combining K-Means representation with Random Forest)
"""

import os
import sys
import io
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arohan_ppg_cvd.preprocessing.filters import clean_ppg_signal
from arohan_ppg_cvd.preprocessing.normalization import normalize_signal
from arohan_ppg_cvd.preprocessing.segmentation import detect_systolic_peaks, segment_pulse_waves
from arohan_ppg_cvd.preprocessing.sqi import evaluate_signal_quality
from arohan_ppg_cvd.features.derivatives import compute_derivatives
from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor
from arohan_ppg_cvd.inference.predictor import ArohanPredictor
from arohan_ppg_cvd.models.persistence import load_model_bundle
from arohan_ppg_cvd.models.hybrid_pipeline import HybridPPGClassifier
from arohan_ppg_cvd.adapters.spandan_adapter import SpandanAdapter
from arohan_ppg_cvd.adapters.upload_parser import parse_and_validate_upload, UploadValidationError
from scripts.generate_synthetic_dataset import generate_single_pulse_wave

# Page Configuration
st.set_page_config(
    page_title="Arohan PPG-CVD Screening AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Professional Clinical Demonstration
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 5px solid #d97706;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 22px;
        color: #92400e;
        font-size: 0.92rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .synthetic-disclaimer {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 22px;
        color: #1e40af;
        font-size: 0.92rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .status-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 20px;
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        align-items: center;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #166534;
    }
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
    }
    .badge-purple {
        background-color: #f3e8ff;
        color: #6b21a8;
    }
    .sqi-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def generate_synthetic_ppg(fs: float = 30.0, duration_sec: float = 10.0, bpm: float = 72.0) -> np.ndarray:
    """Generate realistic synthetic PPG pulse waveform with systolic peak and dicrotic notch."""
    t = np.linspace(0, duration_sec, int(fs * duration_sec))
    beat_freq = bpm / 60.0
    sig = np.zeros_like(t)

    # Sum harmonics to emulate physiological pulse wave
    for i, ti in enumerate(t):
        phase = (ti * beat_freq) % 1.0
        # Systolic component (steep rise)
        systolic = np.exp(-((phase - 0.20) ** 2) / 0.008)
        # Dicrotic notch and diastolic wave
        diastolic = 0.35 * np.exp(-((phase - 0.42) ** 2) / 0.015)
        sig[i] = systolic + diastolic

    # Add low-frequency respiration baseline wander and minor sensor noise
    respiration = 0.15 * np.sin(2 * np.pi * 0.25 * t)
    noise = np.random.normal(0, 0.02, size=len(t))
    return (sig + respiration + noise).astype(np.float32)


# ==========================================
# Sidebar: Mode Selection
# ==========================================
st.sidebar.title("🫀 Arohan AI Control Panel")

workflow_mode = st.sidebar.radio(
    "Select Workflow Mode:",
    [
        "🫀 Clinical PPG Research & Real Data",
        "🧪 Synthetic Hybrid AI Demo (Proof of Concept)",
    ],
    index=0,
    help="Switch between clinical Spandan research mode and the synthetic hybrid ML demonstration.",
)

st.sidebar.markdown("---")

# ==============================================================================
# WORKFLOW MODE 1: CLINICAL PPG RESEARCH & REAL DATA
# ==============================================================================
if workflow_mode == "🫀 Clinical PPG Research & Real Data":

    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Load Real Spandan Sample", "Upload CSV / JSON", "Synthetic Pulse (Demo)"],
    )

    # Sampling Frequency Input
    manual_fs_default = 30.0
    if "auto_detected_fs" in st.session_state and st.session_state["auto_detected_fs"] is not None:
        manual_fs_default = float(st.session_state["auto_detected_fs"])

    fs = st.sidebar.number_input(
        "Sampling Frequency (Hz):",
        min_value=5.0,
        max_value=500.0,
        value=manual_fs_default,
        step=1.0,
        help="Sampling rate of the recording. Auto-detected from file metadata when present.",
    )
    st.sidebar.caption("• 25–30 Hz: Smartphone optical camera (e.g. Spandan)\n• 50–100+ Hz: Clinical / wearable pulse oximeter")

    available_model_opts = []
    if os.path.exists("models/clinical_cvd_model.joblib"):
        available_model_opts.append("models/clinical_cvd_model.joblib (CVD Benchmark: Deep 1D CNN + Structured Fusion)")
    if os.path.exists("models/unsupervised_kmeans.joblib"):
        available_model_opts.append("models/unsupervised_kmeans.joblib (Real Spandan K-Means Exploration)")
    available_model_opts.append("Custom Path...")

    selected_opt = st.sidebar.selectbox(
        "AI Model Architecture:",
        available_model_opts,
        index=0,
        help="Select trained clinical model or exploratory unsupervised K-Means clusterer.",
    )
    if "Custom Path" in selected_opt:
        model_path_input = st.sidebar.text_input("Trained Model Path:", value="models/clinical_cvd_model.joblib")
    else:
        model_path_input = selected_opt.split(" ")[0]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Preprocessing Settings")
    lowcut = st.sidebar.slider("Bandpass Low Cut (Hz)", 0.1, 1.5, 0.5, 0.1, help="Removes respiratory baseline wander")
    highcut = st.sidebar.slider("Bandpass High Cut (Hz)", 4.0, 15.0, 8.0, 0.5, help="Attenuates high-frequency sensor noise")
    norm_method = st.sidebar.selectbox("Signal Normalization", ["z_score", "min_max", "robust"], index=0)

    # Data Ingestion
    raw_signal = None
    signal_label = "Signal"
    source_info = {}
    upload_warnings = []
    device_bpm_ref = None

    if data_source == "Load Real Spandan Sample":
        sample_path = "data/sample_ppg.csv"
        if os.path.exists(sample_path):
            try:
                with open(sample_path, "rb") as f:
                    parsed = parse_and_validate_upload(f.read(), "sample_ppg.csv", manual_fs=fs)
                raw_signal = parsed["signal"]
                fs = parsed["sampling_rate"]
                st.session_state["auto_detected_fs"] = fs
                fid_val = parsed["file_id"]
                device_bpm_ref = parsed["device_bpm"]
                signal_label = f"Real Spandan Recording ({fid_val})"
                source_info = {
                    "source": "Local Spandan Sample",
                    "filename": "sample_ppg.csv",
                    "file_id": fid_val,
                    "n_samples": parsed["n_samples"],
                    "duration_sec": parsed["duration_sec"],
                    "fs": fs,
                    "fs_detected": parsed["fs_auto_detected"],
                }
            except Exception as e:
                st.sidebar.error(f"Error loading sample file: {e}")
        else:
            st.info("Creating default sample recording in data/sample_ppg.csv...")
            raw_signal = generate_synthetic_ppg(fs=30.0, duration_sec=10.0, bpm=72.0)
            os.makedirs("data", exist_ok=True)
            pd.DataFrame({"signal": raw_signal, "sampling_rate": 30.0, "file_id": "synthetic_sample"}).to_csv(sample_path, index=False)
            signal_label = "Generated Sample (72 BPM)"
            source_info = {
                "source": "Generated Sample",
                "filename": "sample_ppg.csv",
                "file_id": "synthetic_sample",
                "n_samples": len(raw_signal),
                "duration_sec": 10.0,
                "fs": 30.0,
                "fs_detected": False,
            }

    elif data_source == "Upload CSV / JSON":
        st.sidebar.markdown("### 📁 Upload File")
        uploaded_file = st.sidebar.file_uploader(
            "Upload PPG Recording (.csv, .json)",
            type=["csv", "json", "txt"],
            help="Upload CSV with PPG amplitudes or JSON in Spandan/raw format.",
        )

        if uploaded_file is not None:
            try:
                parsed = parse_and_validate_upload(
                    file_bytes_or_buffer=uploaded_file,
                    filename=uploaded_file.name,
                    manual_fs=fs,
                )
                raw_signal = parsed["signal"]
                fs = parsed["sampling_rate"]
                st.session_state["auto_detected_fs"] = fs if parsed["fs_auto_detected"] else None
                device_bpm_ref = parsed["device_bpm"]
                upload_warnings = parsed["warnings"]
                signal_label = f"Uploaded: {uploaded_file.name}"
                source_info = {
                    "source": "User Upload",
                    "filename": uploaded_file.name,
                    "file_id": parsed["file_id"],
                    "n_samples": parsed["n_samples"],
                    "duration_sec": parsed["duration_sec"],
                    "fs": fs,
                    "fs_detected": parsed["fs_auto_detected"],
                    "channel": parsed["channel_used"],
                }
            except UploadValidationError as e:
                st.error(f"❌ **File Validation Error:** {e}")
                st.stop()
            except Exception as e:
                st.error(f"❌ **Unexpected Error Parsing File:** {e}")
                st.stop()
        else:
            st.markdown("<div class='main-header'>Arohan_PPG_CVD Clinical AI Dashboard</div>", unsafe_allow_html=True)
            st.markdown("<div class='sub-header'>Non-invasive pulse wave analysis, morphology extraction & CVD screening research</div>", unsafe_allow_html=True)
            st.info("👈 **Please select or drag-and-drop a CSV or JSON file in the sidebar to begin analysis.**")

            col_u1, col_u2 = st.columns(2)
            with col_u1:
                st.markdown("### 📋 Supported CSV Formats")
                st.markdown(
                    """
                    - **Single or Multi-column CSV**:
                      - Amplitude columns: `signal`, `ppg`, `green`, `g`, `pleth`, or raw numbers.
                      - Metadata columns: `sampling_rate`, `fs`, `file_id`, `bpm`.
                      - Timestamps: `t` or `timestamp` (sampling rate is auto-calculated).
                    """
                )
            with col_u2:
                st.markdown("### 📋 Supported JSON Formats")
                st.markdown(
                    """
                    - **Spandan-1M Hardware Format**: `{'file_id', 'bpm', 'ppg': [{'r','g','b','t'}]}`.
                    - **Processed / Subset Format**: `{'file_id', 'sampling_rate', 'signal': [...]}`.
                    - **Raw Array JSON**: `[125.4, 126.1, ...]`.
                    """
                )
            st.stop()

    elif data_source == "Synthetic Pulse (Demo)":
        demo_bpm = st.sidebar.slider("Synthetic Heart Rate (BPM):", 50, 140, 75)
        demo_duration = st.sidebar.slider("Duration (seconds):", 5, 30, 10)
        raw_signal = generate_synthetic_ppg(fs=fs, duration_sec=float(demo_duration), bpm=float(demo_bpm))
        signal_label = f"Synthetic PPG ({demo_bpm} BPM)"
        source_info = {
            "source": "Synthetic Generator",
            "filename": "synthetic_signal",
            "file_id": f"syn_{demo_bpm}bpm",
            "n_samples": len(raw_signal),
            "duration_sec": float(demo_duration),
            "fs": fs,
            "fs_detected": False,
        }

    # Main Header & Clinical Notice
    st.markdown("<div class='main-header'>Arohan_PPG_CVD Clinical AI Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Non-invasive pulse wave analysis, morphology extraction & CVD screening research</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='disclaimer-box'>
            <b>⚠️ MEDICAL ML RESEARCH & SAFETY NOTICE:</b><br>
            The Indian <b>Spandan-1M-V1.0-raw</b> dataset does NOT contain clinical diagnosis labels (CAD / CVD).
            This platform distinguishes exploratory mathematical pulse clustering from medically validated diagnostic models.
            Clinical disease predictions remain disabled until a model trained on verified medical diagnoses is loaded.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for w in upload_warnings:
        st.warning(f"⚠️ **Notice:** {w}")

    if raw_signal is None or len(raw_signal) < int(fs * 2.0):
        st.warning("⚠️ Insufficient signal length: minimum 2.0 seconds of valid data required.")
        st.stop()

    # Status Indicator
    fs_badge = "✅ Auto-Detected" if source_info.get("fs_detected") else "⚙️ User Defined"
    st.markdown(
        f"""
        <div class='status-card'>
            <div><b>📁 Source:</b> <code>{source_info.get('filename', 'Unknown')}</code></div>
            <div><b>🆔 ID:</b> <code>{source_info.get('file_id', 'Unknown')}</code></div>
            <div><b>⏱️ Duration:</b> <code>{source_info.get('duration_sec', 0.0):.1f}s</code> ({source_info.get('n_samples', 0)} samples)</div>
            <div><b>📡 Sampling Rate:</b> <code>{source_info.get('fs', fs):.1f} Hz</code> <span class='badge-pill badge-info'>{fs_badge}</span></div>
            <div><b>⚡ Status:</b> <span class='badge-pill badge-success'>Pipeline Active</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Signal Processing
    t_sec = np.arange(len(raw_signal)) / fs
    filtered_sig = clean_ppg_signal(raw_signal, fs=fs, lowcut=lowcut, highcut=highcut)
    norm_sig = normalize_signal(filtered_sig, method=norm_method)
    is_acceptable, sqi_metrics = evaluate_signal_quality(norm_sig, fs=fs)
    peaks, _ = detect_systolic_peaks(norm_sig, fs=fs)
    beats = segment_pulse_waves(norm_sig, fs=fs)

    # Features
    extractor = PPGFeatureExtractor(
        default_fs=fs,
        filter_lowcut=lowcut,
        filter_highcut=highcut,
        normalization_method=norm_method,
    )
    features = extractor.extract_from_signal(
        raw_signal=raw_signal,
        fs=fs,
        file_id=source_info.get("file_id"),
        bpm_reference=device_bpm_ref,
    )

    # Scorecard
    st.markdown("### 📊 Pulse & Quality Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        hr_val = features.get("hr_bpm", 0.0) if features else 0.0
        st.metric("Estimated Heart Rate", f"{hr_val:.1f} BPM" if hr_val > 0 else "N/A")
    with col2:
        st.metric("Detected Pulse Beats", f"{len(beats)}")
    with col3:
        sqi_badge = "✅ Acceptable" if is_acceptable else "⚠️ Review Signal"
        st.metric("Signal Quality (SQI)", sqi_badge)
    with col4:
        pi_val = sqi_metrics.get("perfusion_index", 0.0)
        st.metric("Perfusion Index", f"{pi_val:.2f}%")
    with col5:
        aging_idx = features.get("mean_apg_aging_index", 0.0) if features else 0.0
        st.metric("APG Aging Index", f"{aging_idx:.2f}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 1. Signal & Filtering",
        "🩺 2. Morphology & Derivatives (APG)",
        "📑 3. Feature Vector",
        "🤖 4. AI Model Inference",
    ])

    with tab1:
        st.subheader("Signal Conditioning & Systolic Peak Detection")
        total_dur = len(raw_signal) / fs
        if total_dur > 15.0:
            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                time_window = st.slider(
                    "Display Time Window (seconds):",
                    min_value=0.0,
                    max_value=float(np.floor(total_dur)),
                    value=(0.0, min(15.0, float(total_dur))),
                    step=1.0,
                )
            with col_t2:
                show_full = st.checkbox("Show Entire Signal", value=False)
                if show_full:
                    idx_slice = slice(0, len(t_sec))
                else:
                    idx_slice = (t_sec >= time_window[0]) & (t_sec <= time_window[1])
        else:
            idx_slice = slice(0, len(t_sec))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        ax1.plot(t_sec[idx_slice], raw_signal[idx_slice], color="#334155", lw=1.3, label="Raw Optical Signal")
        ax1.set_ylabel("Raw Amplitude", fontsize=10)
        ax1.set_title(f"Raw Input: {signal_label}", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", frameon=True)

        ax2.plot(t_sec[idx_slice], norm_sig[idx_slice], color="#2563eb", lw=1.5, label=f"Conditioned ({lowcut}–{highcut} Hz, {norm_method})")
        if len(peaks) > 0:
            visible_peaks = peaks if isinstance(idx_slice, slice) else [p for p in peaks if idx_slice[p]]
            if len(visible_peaks) > 0:
                ax2.scatter(t_sec[visible_peaks], norm_sig[visible_peaks], color="#dc2626", s=48, zorder=5, label=f"Systolic Peaks (n={len(peaks)})")
        ax2.set_xlabel("Time (seconds)", fontsize=10)
        ax2.set_ylabel("Normalized Amplitude", fontsize=10)
        ax2.set_title("Conditioned PPG with Systolic Peak Detection", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Signal Quality Indices (SQI)")
        col_sqi1, col_sqi2, col_sqi3, col_sqi4 = st.columns(4)
        with col_sqi1:
            st.markdown(f"<div class='sqi-box'><div style='color: #64748b; font-size: 0.85rem;'>Skewness (sSQI)</div><div style='font-size: 1.4rem; font-weight: 700;'>{sqi_metrics.get('sSQI', 0.0):.3f}</div><div style='font-size: 0.8rem; color: #475569;'>Normal: > 0.0</div></div>", unsafe_allow_html=True)
        with col_sqi2:
            st.markdown(f"<div class='sqi-box'><div style='color: #64748b; font-size: 0.85rem;'>Kurtosis (kSQI)</div><div style='font-size: 1.4rem; font-weight: 700;'>{sqi_metrics.get('kSQI', 0.0):.3f}</div><div style='font-size: 0.8rem; color: #475569;'>Normal: > 2.0</div></div>", unsafe_allow_html=True)
        with col_sqi3:
            st.markdown(f"<div class='sqi-box'><div style='color: #64748b; font-size: 0.85rem;'>Power Ratio (pSQI)</div><div style='font-size: 1.4rem; font-weight: 700;'>{sqi_metrics.get('pSQI', 0.0):.3f}</div><div style='font-size: 0.8rem; color: #475569;'>Normal: 0.5 – 0.99</div></div>", unsafe_allow_html=True)
        with col_sqi4:
            st.markdown(f"<div class='sqi-box'><div style='color: #64748b; font-size: 0.85rem;'>Perfusion Index (PI)</div><div style='font-size: 1.4rem; font-weight: 700;'>{sqi_metrics.get('perfusion_index', 0.0):.2f}%</div><div style='font-size: 0.8rem; color: #475569;'>Normal: > 0.1%</div></div>", unsafe_allow_html=True)

    with tab2:
        st.subheader("Vascular Waveform Morphology & Acceleration Plethysmogram (APG)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Ensemble Pulse Waveform Profile")
            if len(beats) > 0:
                fig_beat, ax_b = plt.subplots(figsize=(6.5, 4.2))
                for b in beats[:15]:
                    t_b = np.linspace(0, 100, len(b["beat"]))
                    b_zero = b["beat"] - np.min(b["beat"])
                    ax_b.plot(t_b, b_zero, color="#93c5fd", lw=0.9, alpha=0.55)
                aligned = [np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(b["beat"])), b["beat"] - np.min(b["beat"])) for b in beats]
                mean_b = np.mean(aligned, axis=0)
                ax_b.plot(np.linspace(0, 100, 100), mean_b, color="#1e40af", lw=2.5, label=f"Mean Beat Profile (n={len(beats)})")
                ax_b.set_xlabel("Normalized Beat Phase (%)", fontsize=10)
                ax_b.set_ylabel("Pulse Amplitude", fontsize=10)
                ax_b.set_title("Single-Beat Cardiac Segmentation", fontsize=11, fontweight="bold")
                ax_b.legend(loc="upper right", frameon=True)
                ax_b.grid(True, linestyle=":", alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig_beat)
                plt.close(fig_beat)
            else:
                st.info("ℹ️ Insufficient clean beats segmented to plot ensemble beat profile.")

        with col_b:
            st.markdown("##### Velocity & Acceleration Derivatives (VPG & APG)")
            vpg, apg = compute_derivatives(norm_sig, fs)
            fig_der, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)
            sample_slice = slice(0, min(len(t_sec), int(fs * 4.0)))
            ax_v.plot(t_sec[sample_slice], vpg[sample_slice], color="#0d9488", lw=1.3)
            ax_v.set_ylabel("VPG (1st Deriv)", fontsize=9)
            ax_v.set_title("Vascular Derivatives: VPG & APG Waves", fontsize=11, fontweight="bold")
            ax_v.grid(True, linestyle=":", alpha=0.5)
            ax_a.plot(t_sec[sample_slice], apg[sample_slice], color="#d97706", lw=1.3)
            ax_a.set_ylabel("APG (2nd Deriv)", fontsize=9)
            ax_a.set_xlabel("Time (seconds)", fontsize=10)
            ax_a.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig_der)
            plt.close(fig_der)

        st.markdown("##### Vascular Compliance & Arterial Stiffness Indices")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Augmentation Index (AI)", f"{features.get('mean_augmentation_index', 0.0):.2f}" if features else "0.0")
        with col_m2:
            st.metric("Reflection Index (RI)", f"{features.get('mean_reflection_index', 0.0):.2f}" if features else "0.0")
        with col_m3:
            st.metric("Crest Time (Systolic Rise)", f"{features.get('mean_crest_time', 0.0):.3f} s" if features else "0.0 s")
        with col_m4:
            st.metric("Mean Pulse Duration", f"{features.get('mean_pulse_duration', 0.0):.3f} s" if features else "0.0 s")

    with tab3:
        st.subheader("Extracted Physiological Feature Vector")
        st.markdown("Comprehensive 75-dimensional physiological, morphological, derivative, and spectral feature set.")
        if features:
            df_feat = pd.DataFrame([features])
            meta_keys = {"file_id", "sampling_rate", "duration_sec", "detected_peaks_count", "segmented_beats_count", "device_bpm_ref", "is_acceptable"}
            feat_cols = [c for c in df_feat.columns if c not in meta_keys]
            df_display = df_feat[feat_cols].T.reset_index()
            df_display.columns = ["Feature Name", "Extracted Value"]
            df_display["Extracted Value"] = df_display["Extracted Value"].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else str(v))
            st.dataframe(df_display, width="stretch", height=420)
            csv_bytes = df_feat.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Extracted Features (CSV)",
                data=csv_bytes,
                file_name=f"{source_info.get('file_id', 'ppg')}_features.csv",
                mime="text/csv",
            )

    with tab4:
        st.subheader("AI Screening & Clinical Risk Analysis")
        predictor = ArohanPredictor()
        model_error = None
        if os.path.exists(model_path_input):
            try:
                predictor.load_model(model_path_input)
            except Exception as e:
                model_error = str(e)

        col_inf1, col_inf2 = st.columns(2)

        # Check if the loaded model is a clinical supervised/multimodal bundle
        if predictor.is_ready_for_clinical_inference:
            meta_dict = predictor.bundle_dict or {}
            held_metrics = meta_dict.get("held_out_metrics", {})
            pop_note = meta_dict.get("population", "External Clinical Benchmark")
            diag_target = meta_dict.get("diagnoses", "CVD vs Healthy Control")
            dataset_name = meta_dict.get("dataset", "PPG-BP Database (Guilin People's Hospital)")

            with col_inf1:
                st.markdown("#### 1. Clinical Model Architecture & Benchmark")
                st.markdown(f"**Target Diagnoses:** `{diag_target}`")
                st.markdown(f"**Training Benchmark:** `{dataset_name}`")
                st.markdown(
                    f"""
                    <div style='background-color: #fff7ed; border-left: 4px solid #ea580c; padding: 10px 14px; border-radius: 6px; margin: 10px 0; color: #9a3412; font-size: 0.88rem;'>
                        <b>⚠️ POPULATION PROVENANCE NOTICE:</b><br>
                        Trained on <b>{pop_note}</b>.<br>
                        This model is an algorithmic research benchmark demonstrating multi-branch deep 1D CNN + structured feature fusion.
                        It has <b>NOT</b> been validated on Indian hospital cohorts and is NOT a diagnostic medical device.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if held_metrics:
                    st.markdown("##### Held-Out Test Evaluation (Zero Patient Leakage)")
                    col_met1, col_met2, col_met3 = st.columns(3)
                    with col_met1:
                        st.metric("Test Accuracy", f"{held_metrics.get('accuracy', 0.0):.1%}")
                        st.metric("Sensitivity", f"{held_metrics.get('recall_sensitivity', 0.0):.1%}")
                    with col_met2:
                        st.metric("F1-Score", f"{held_metrics.get('f1_score', 0.0):.3f}")
                        st.metric("Specificity", f"{held_metrics.get('specificity', 0.0):.1%}")
                    with col_met3:
                        st.metric("ROC-AUC", f"{held_metrics.get('roc_auc', 0.0):.3f}")
                        st.metric("PR-AUC", f"{held_metrics.get('pr_auc', 0.0):.3f}")

            with col_inf2:
                st.markdown("#### 2. Live Patient Recording Prediction")
                if features and raw_signal is not None:
                    res = predictor.analyze_signal(raw_signal, fs=fs)
                    pred_info = res.get("clinical_prediction", {})
                    p_cvd = float(pred_info.get("probability_cvd", 0.0))
                    p_healthy = float(pred_info.get("probability_healthy", 1.0 - p_cvd))
                    tier = pred_info.get("risk_tier", "Unknown")

                    badge_class = "badge-warning" if p_cvd >= 0.50 else "badge-success"
                    st.markdown(f"**Predicted Category:** <span class='badge-pill {badge_class}'>{tier}</span>", unsafe_allow_html=True)
                    st.progress(p_cvd)
                    st.write(f"P(Cardiovascular Disease): **{p_cvd * 100:.1f}%** | P(Healthy Normotensive): **{p_healthy * 100:.1f}%**")

                    st.markdown("##### 🧬 Multimodal Fusion Readiness")
                    st.markdown(
                        f"""
                        - 🫀 **Branch A (Structured 75-D Physiology):** Extracted ({len(features)} metrics)
                        - 🌊 **Branch B (Unstructured Deep 1D CNN):** Generated 32-D latent embedding
                        - 🩺 **Branch C (Clinical Labs):** Schema ready (`MultimodalPatientRecord`)
                        - 📄 **Branch D (Clinical Text Notes):** BioClinicalBERT embedding ready
                        """
                    )
                else:
                    st.info("Upload or load a recording to view live multi-branch CVD prediction.")

        else:
            with col_inf1:
                st.markdown("#### 1. Exploratory Pulse Cluster (Unsupervised K-Means)")
                if predictor.model and getattr(predictor.model, "is_unsupervised_fitted", False) and features:
                    df_feat = pd.DataFrame([features])
                    cluster_id = int(predictor.model.clusterer.predict_clusters(df_feat)[0])
                    n_clusters = predictor.model.clusterer.n_clusters
                    st.success(f"Assigned Morphological Cluster: **Cluster #{cluster_id}** (of {n_clusters})")
                    dists = predictor.model.clusterer.transform_to_cluster_distances(df_feat)[0]
                    st.markdown(f"**Distance to Centroid:** `{dists[cluster_id]:.3f}`")
                    st.markdown(
                        """
                        > [!NOTE]
                        > **Waveform Subtype Interpretation:**
                        > Unsupervised K-Means groups pulse contours with similar geometric shapes.
                        > This is purely mathematical clustering and NOT a medical disease diagnosis.
                        """
                    )
                else:
                    if model_error:
                        st.warning(f"Could not load model from `{model_path_input}`: {model_error}")
                    else:
                        st.info("Unsupervised model not loaded. Check model path in sidebar.")

            with col_inf2:
                st.markdown("#### 2. Supervised CVD / CAD Screening Prediction")
                st.markdown(
                    """
                    <div style='background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; margin-bottom: 12px;'>
                        <div style='color: #475569; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px;'>
                            ℹ️ Exploratory Unsupervised Mode Active
                        </div>
                        <div style='color: #64748b; font-size: 0.9rem; line-height: 1.5;'>
                            The Indian <b>Spandan-1M-V1.0-raw</b> dataset does not contain confirmed clinical disease labels.
                            <br><br>
                            <b>To inspect real disease screening:</b><br>
                            Select <code>models/clinical_cvd_model.joblib</code> from the sidebar dropdown.
                            This loads our multi-branch deep 1D CNN + structured feature fusion model trained on confirmed cardiovascular disease patients.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div class='disclaimer-box'>
                <b>🩺 MEDICAL ML SAFETY DISCLAIMER:</b>
                Arohan PPG Cardiovascular AI is an experimental assistive screening research framework.
                Predictions are intended exclusively for algorithmic benchmarking and clinician-in-the-loop triage support.
                All cardiovascular diagnoses require standard-of-care clinical diagnostics (12-lead ECG, echocardiography, coronary angiography, serum troponin).
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# WORKFLOW MODE 2: SYNTHETIC HYBRID AI DEMO (PROOF OF CONCEPT)
# ==============================================================================
else:
    st.markdown("<div class='main-header'>🧪 Synthetic Hybrid AI Demonstration</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Proof-of-concept pipeline combining unsupervised K-Means representations with supervised Random Forest</div>",
        unsafe_allow_html=True,
    )

    # Mandatory Safety Notice for Synthetic Demonstration Mode
    st.markdown(
        """
        <div class='synthetic-disclaimer'>
            <b>⚠️ RESEARCH DEMONSTRATION ONLY:</b><br>
            This model distinguishes computer-generated PPG patterns and cannot diagnose cardiovascular disease.<br>
            Simulated <b>Pattern A</b> and <b>Pattern B</b> represent synthetic morphological waveforms created to demonstrate
            end-to-end feature extraction, unsupervised K-Means representation augmentation, and Random Forest classification.
            Never interpret these simulated patterns as clinical Healthy, CAD, or CVD diagnoses.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls for Synthetic Mode
    st.sidebar.markdown("### 🧪 Synthetic Signal Generation")
    syn_source = st.sidebar.radio(
        "Simulation Mode:",
        [
            "Generate Pattern A (Sharp Upstroke, Distinct Dicrotic Wave)",
            "Generate Pattern B (Prolonged Rise, Blunted Dicrotic Wave)",
            "Upload Synthetic CSV / JSON",
        ],
    )

    syn_fs = 30.0
    syn_bpm = st.sidebar.slider("Simulated Heart Rate (BPM):", 55, 110, 72)
    syn_amp = st.sidebar.slider("Pulse Amplitude:", 0.6, 1.8, 1.1, 0.05)
    syn_noise = st.sidebar.slider("Noise Level (Gaussian):", 0.005, 0.08, 0.02, 0.005)
    syn_wander = st.sidebar.slider("Baseline Respiration Wander:", 0.02, 0.30, 0.12, 0.02)
    syn_duration = st.sidebar.slider("Signal Duration (seconds):", 5.0, 15.0, 6.0, 1.0)
    syn_seed = st.sidebar.number_input("Random Seed:", value=42, min_value=0, max_value=999999, step=1)

    syn_model_path = st.sidebar.text_input(
        "Trained Synthetic Hybrid Model:",
        value="models/synthetic_hybrid_model.joblib",
        help="Model bundle combining K-Means representation and Random Forest trained on 1,000 synthetic records.",
    )

    # Generate or Upload Synthetic Signal
    syn_raw_signal = None
    syn_label = ""
    ground_truth_pattern = None

    if syn_source.startswith("Generate Pattern A"):
        t_arr = np.linspace(0, syn_duration, int(syn_fs * syn_duration))
        rng = np.random.RandomState(int(syn_seed))
        syn_raw_signal = generate_single_pulse_wave(
            t=t_arr,
            bpm=float(syn_bpm),
            pattern="A",
            amplitude=float(syn_amp),
            noise_level=float(syn_noise),
            wander_amp=float(syn_wander),
            rng=rng,
        )
        syn_label = "Simulated Pattern A"
        ground_truth_pattern = "Synthetic Pattern A (Class 0)"

    elif syn_source.startswith("Generate Pattern B"):
        t_arr = np.linspace(0, syn_duration, int(syn_fs * syn_duration))
        rng = np.random.RandomState(int(syn_seed))
        syn_raw_signal = generate_single_pulse_wave(
            t=t_arr,
            bpm=float(syn_bpm),
            pattern="B",
            amplitude=float(syn_amp),
            noise_level=float(syn_noise),
            wander_amp=float(syn_wander),
            rng=rng,
        )
        syn_label = "Simulated Pattern B"
        ground_truth_pattern = "Synthetic Pattern B (Class 1)"

    else:
        syn_uploaded = st.sidebar.file_uploader("Upload Synthetic PPG (.csv, .json)", type=["csv", "json", "txt"])
        if syn_uploaded is not None:
            try:
                parsed_syn = parse_and_validate_upload(syn_uploaded, syn_uploaded.name, manual_fs=syn_fs)
                syn_raw_signal = parsed_syn["signal"]
                syn_fs = parsed_syn["sampling_rate"]
                syn_label = f"Uploaded: {syn_uploaded.name}"
                ground_truth_pattern = "Uploaded File"
            except Exception as e:
                st.error(f"Error parsing upload: {e}")
                st.stop()
        else:
            st.info("👈 Please select a simulation pattern or upload a synthetic test file in the sidebar.")
            st.stop()

    # Preprocessing of Synthetic Signal
    t_syn = np.arange(len(syn_raw_signal)) / syn_fs
    syn_clean = clean_ppg_signal(syn_raw_signal, fs=syn_fs, lowcut=0.5, highcut=8.0)
    syn_norm = normalize_signal(syn_clean, method="z_score")
    syn_peaks, _ = detect_systolic_peaks(syn_norm, fs=syn_fs)
    syn_beats = segment_pulse_waves(syn_norm, fs=syn_fs)

    # Feature Extraction
    syn_extractor = PPGFeatureExtractor(default_fs=syn_fs)
    syn_feats = syn_extractor.extract_from_signal(syn_raw_signal, fs=syn_fs, bpm_reference=float(syn_bpm))

    # Metric Banner
    st.markdown(
        f"""
        <div class='status-card'>
            <div><b>🧪 Signal Type:</b> <code>{syn_label}</code></div>
            <div><b>🎯 Ground Truth:</b> <span class='badge-pill badge-purple'>{ground_truth_pattern}</span></div>
            <div><b>⏱️ Duration:</b> <code>{len(syn_raw_signal)/syn_fs:.1f}s</code> ({len(syn_raw_signal)} samples)</div>
            <div><b>💓 Heart Rate:</b> <code>{syn_feats.get('hr_bpm', 0.0):.1f} BPM</code></div>
            <div><b>⚡ Beats Detected:</b> <span class='badge-pill badge-success'>{len(syn_beats)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Tabs for Synthetic Demo
    s_tab1, s_tab2, s_tab3, s_tab4 = st.tabs([
        "📈 1. Waveform Conditioning & Peaks",
        "🩺 2. Morphology & Vascular Derivatives",
        "📑 3. Extracted Feature Vector",
        "🤖 4. Hybrid AI Classification & Benchmark",
    ])

    with s_tab1:
        st.subheader("Waveform Conditioning & Systolic Peaks")
        fig_s1, (ax_s1, ax_s2) = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True)

        ax_s1.plot(t_syn, syn_raw_signal, color="#475569", lw=1.3, label="Raw Simulated Waveform")
        ax_s1.set_ylabel("Amplitude")
        ax_s1.set_title(f"Input: {syn_label}", fontsize=11, fontweight="bold")
        ax_s1.grid(True, linestyle=":", alpha=0.5)
        ax_s1.legend(loc="upper right")

        ax_s2.plot(t_syn, syn_norm, color="#0284c7", lw=1.5, label="Filtered & Normalized (0.5–8 Hz)")
        if len(syn_peaks) > 0:
            ax_s2.scatter(t_syn[syn_peaks], syn_norm[syn_peaks], color="#dc2626", s=50, zorder=5, label=f"Systolic Peaks (n={len(syn_peaks)})")
        ax_s2.set_xlabel("Time (seconds)")
        ax_s2.set_ylabel("Standardized (Z-Score)")
        ax_s2.set_title("Conditioned Signal with Systolic Peak Detections", fontsize=11, fontweight="bold")
        ax_s2.grid(True, linestyle=":", alpha=0.5)
        ax_s2.legend(loc="upper right")

        plt.tight_layout()
        st.pyplot(fig_s1)
        plt.close(fig_s1)

    with s_tab2:
        st.subheader("Pulse Wave Morphology & Derivatives")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("##### Average Pulse Beat Profile")
            if len(syn_beats) > 0:
                fig_sb, ax_sb = plt.subplots(figsize=(6, 4))
                for b in syn_beats[:10]:
                    t_b = np.linspace(0, 100, len(b["beat"]))
                    ax_sb.plot(t_b, b["beat"] - np.min(b["beat"]), color="#93c5fd", lw=0.9, alpha=0.5)
                aligned = [np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(b["beat"])), b["beat"] - np.min(b["beat"])) for b in syn_beats]
                mean_beat = np.mean(aligned, axis=0)
                ax_sb.plot(np.linspace(0, 100, 100), mean_beat, color="#1e40af", lw=2.5, label="Mean Beat Profile")
                ax_sb.set_xlabel("Beat Phase (%)")
                ax_sb.set_ylabel("Normalized Amplitude")
                ax_sb.set_title("Segmented Beat Contour", fontsize=11, fontweight="bold")
                ax_sb.grid(True, linestyle=":", alpha=0.5)
                ax_sb.legend()
                plt.tight_layout()
                st.pyplot(fig_sb)
                plt.close(fig_sb)

        with col_m2:
            st.markdown("##### 1st & 2nd Derivatives (VPG & APG)")
            vpg_s, apg_s = compute_derivatives(syn_norm, syn_fs)
            fig_sd, (ax_sv, ax_sa) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
            slice_s = slice(0, min(len(t_syn), int(syn_fs * 4.0)))

            ax_sv.plot(t_syn[slice_s], vpg_s[slice_s], color="#0d9488", lw=1.3)
            ax_sv.set_ylabel("VPG (1st Deriv)")
            ax_sv.grid(True, linestyle=":", alpha=0.5)

            ax_sa.plot(t_syn[slice_s], apg_s[slice_s], color="#d97706", lw=1.3)
            ax_sa.set_ylabel("APG (2nd Deriv)")
            ax_sa.set_xlabel("Time (seconds)")
            ax_sa.grid(True, linestyle=":", alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig_sd)
            plt.close(fig_sd)

        st.markdown("##### Distinguishing Morphological Metrics")
        col_sm1, col_sm2, col_sm3, col_sm4 = st.columns(4)
        with col_sm1:
            st.metric("Crest Time (Systolic Rise)", f"{syn_feats.get('mean_crest_time', 0.0):.3f} s")
        with col_sm2:
            st.metric("Augmentation Index", f"{syn_feats.get('mean_augmentation_index', 0.0):.2f}")
        with col_sm3:
            st.metric("APG Aging Index", f"{syn_feats.get('mean_apg_aging_index', 0.0):.2f}")
        with col_sm4:
            st.metric("Mean Pulse Duration", f"{syn_feats.get('mean_pulse_duration', 0.0):.3f} s")

    with s_tab3:
        st.subheader("Extracted Physiological Feature Vector")
        if syn_feats:
            df_syn_feat = pd.DataFrame([syn_feats])
            meta_cols = {"file_id", "sampling_rate", "duration_sec", "detected_peaks_count", "segmented_beats_count", "device_bpm_ref", "is_acceptable"}
            feat_cols = [c for c in df_syn_feat.columns if c not in meta_cols]
            df_disp = df_syn_feat[feat_cols].T.reset_index()
            df_disp.columns = ["Feature Name", "Extracted Value"]
            df_disp["Extracted Value"] = df_disp["Extracted Value"].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else str(v))
            st.dataframe(df_disp, width="stretch", height=400)
            csv_syn = df_syn_feat.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Extracted Features (CSV)", csv_syn, "synthetic_features.csv", "text/csv")

    with s_tab4:
        st.subheader("Hybrid AI Classification & Benchmark Performance")

        # Load Trained Synthetic Hybrid Model
        hybrid_loaded = False
        syn_model = None
        if os.path.exists(syn_model_path):
            try:
                syn_model = load_model_bundle(syn_model_path)
                hybrid_loaded = syn_model.is_supervised_fitted
            except Exception as e:
                st.warning(f"Could not load synthetic model from `{syn_model_path}`: {e}")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.markdown("#### 1. Real-Time Pattern Classification")
            if hybrid_loaded and syn_feats:
                df_inp = pd.DataFrame([syn_feats])
                probs = syn_model.predict_proba(df_inp)[0]
                pred_cls = int(syn_model.predict(df_inp)[0])

                pred_name = "Synthetic Pattern A" if pred_cls == 0 else "Synthetic Pattern B"
                badge_style = "badge-info" if pred_cls == 0 else "badge-warning"

                st.markdown(
                    f"""
                    <div style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 12px;'>
                        <div style='font-size: 0.9rem; color: #64748b; font-weight: 600;'>PREDICTED SIMULATED CLASS</div>
                        <div style='font-size: 1.5rem; font-weight: 700; color: #0f172a; margin: 4px 0 10px 0;'>
                            {pred_name}
                        </div>
                        <div style='font-size: 0.88rem; color: #334155;'>
                            P(Pattern A): <b>{probs[0]*100:.1f}%</b> | P(Pattern B): <b>{probs[1]*100:.1f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(float(probs[1]))

                # Unsupervised Cluster Affinity
                cluster_id = int(syn_model.clusterer.predict_clusters(df_inp)[0])
                cluster_dists = syn_model.clusterer.transform_to_cluster_distances(df_inp)[0]
                st.markdown(f"**Unsupervised Morphological Subtype:** Cluster #{cluster_id} (Centroid Distance: `{cluster_dists[cluster_id]:.3f}`)")
            else:
                st.info("Synthetic model not loaded. Run `scripts/train_synthetic_hybrid.py` to create the model bundle.")

        with col_p2:
            st.markdown("#### 2. Baseline vs Hybrid Model Benchmark")
            eval_path = "reports/synthetic_hybrid_evaluation.json"
            if os.path.exists(eval_path):
                with open(eval_path, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)

                b_m = eval_data.get("baseline_model", {})
                h_m = eval_data.get("hybrid_model", {})

                df_comp = pd.DataFrame([
                    {
                        "Model Architecture": "Baseline Random Forest",
                        "Features": f"{b_m.get('feature_count')} base features",
                        "Accuracy": f"{b_m.get('accuracy', 0)*100:.1f}%",
                        "F1-Score": f"{b_m.get('f1_score', 0):.4f}",
                        "ROC-AUC": f"{b_m.get('roc_auc', 0):.4f}",
                    },
                    {
                        "Model Architecture": "Hybrid (K-Means + RF)",
                        "Features": f"{h_m.get('augmented_feature_count')} features (Base + K-Means)",
                        "Accuracy": f"{h_m.get('accuracy', 0)*100:.1f}%",
                        "F1-Score": f"{h_m.get('f1_score', 0):.4f}",
                        "ROC-AUC": f"{h_m.get('roc_auc', 0):.4f}",
                    },
                ])
                st.dataframe(df_comp, use_container_width=True, hide_index=True)
                st.caption("ℹ️ Evaluated on 200 held-out test recordings from 20 unseen synthetic subjects (zero subject leakage).")
            else:
                st.info("Evaluation report not found. Run `scripts/train_synthetic_hybrid.py`.")

        st.markdown("---")
        st.markdown("#### 3. Confusion Matrix & ROC Curve Comparison")
        plot_eval_img = "reports/synthetic_evaluation_metrics.png"
        if os.path.exists(plot_eval_img):
            st.image(plot_eval_img, caption="Baseline vs Hybrid Model Evaluation on 200 Held-Out Synthetic Test Recordings")

        st.markdown("#### 4. Clinical Translation Roadmap")
        st.markdown(
            """
            This demonstration validates the complete hybrid machine-learning workflow on controlled synthetic data.
            When gold-standard clinically labeled datasets become available (e.g., MIMIC-III, VitalDB, or clinically annotated Spandan cohorts):
            1. Replace `data/synthetic/synthetic_features.csv` with the clinically annotated CSV.
            2. Run `scripts/train_supervised.py --data <clinical_dataset.csv> --target-col diagnosis`.
            3. Point the clinical dashboard mode to the resulting `.joblib` model bundle to activate real-world CVD/CAD screening predictions.
            """
        )

st.markdown("---")
st.caption("Arohan PPG-CVD AI Research Platform | Dual-Mode Clinical & Synthetic Demonstration Interface")
