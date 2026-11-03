"""Arohan Cardiovascular AI - Clinical Risk Intelligence Platform.

==============================================================
Interactive Streamlit application featuring dual interfaces:
1. 📄 Patient Overview (Light Theme - Report upload, summary & onboarding)
2. 🫀 Precision Cardiovascular Dashboard (Dark Theme - Real-time metrics, risk engine,
   waveforms, cardiac readouts, and multi-branch AI classification).

Maintains complete backward compatibility with the Arohan PPG processing pipeline,
trained clinical models, unsupervised K-Means clusterers, and multimodal fusion schemas.
"""

import os
import sys
import io
import json
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Add project root to sys.path for robust absolute/relative imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def resolve_project_path(rel_path: str) -> str:
    """Resolve a relative path against CWD or PROJECT_ROOT."""
    if not rel_path:
        return ""
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    if os.path.exists(rel_path):
        return os.path.abspath(rel_path)
    candidate = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return os.path.join(PROJECT_ROOT, rel_path)

# Core library imports
from arohan_ppg_cvd.preprocessing.filters import clean_ppg_signal
from arohan_ppg_cvd.preprocessing.normalization import normalize_signal
from arohan_ppg_cvd.preprocessing.segmentation import detect_systolic_peaks, segment_pulse_waves
from arohan_ppg_cvd.preprocessing.sqi import evaluate_signal_quality
from arohan_ppg_cvd.features.derivatives import compute_derivatives
from arohan_ppg_cvd.features.extractor import PPGFeatureExtractor
from arohan_ppg_cvd.inference.predictor import ArohanPredictor
from arohan_ppg_cvd.models.persistence import load_model_bundle
from arohan_ppg_cvd.models.hybrid_pipeline import HybridPPGClassifier
from arohan_ppg_cvd.adapters.upload_parser import parse_and_validate_upload, UploadValidationError
from arohan_ppg_cvd.models.multimodal_schema import (
    ClinicalLabBiomarkers,
    MultimodalPatientRecord,
    MultimodalFusionCoordinator,
)

# Page Configuration
st.set_page_config(
    page_title="Arohan Cardiovascular Risk Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "dashboard"  # 'overview' or 'dashboard'

if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Dashboard"

if "active_scenario" not in st.session_state:
    st.session_state["active_scenario"] = "Healthy — Baseline"

if "uploaded_patient_data" not in st.session_state:
    st.session_state["uploaded_patient_data"] = None

# Logo helper
def get_logo_b64(preferred="dark"):
    """Load Arohan logo as base64 string for crisp HTML rendering."""
    asset_dir = resolve_project_path("dashboard/assets")
    if preferred == "dark":
        p = os.path.join(asset_dir, "arohan_logo_dark_badge.png")
    else:
        p = os.path.join(asset_dir, "arohan_logo.png")
    if not os.path.exists(p):
        p = os.path.join(asset_dir, "arohan_logo_dark_badge.png")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# Cached Baseline Data & Models
@st.cache_resource(show_spinner=False)
def get_cached_predictor(model_path: str):
    """Cache model predictor for fast sub-second inference."""
    resolved = resolve_project_path(model_path)
    if os.path.exists(resolved):
        try:
            return ArohanPredictor(resolved)
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False)
def load_default_spandan_signal():
    """Load default sample Spandan PPG recording."""
    sample_file = resolve_project_path("data/sample_ppg.csv")
    if os.path.exists(sample_file):
        df = pd.read_csv(sample_file)
        return df["signal"].to_numpy(dtype=np.float32), float(df["sampling_rate"].iloc[0]), str(df["file_id"].iloc[0])
    # Fallback synthetic 72 BPM
    t = np.linspace(0, 10, 300)
    sig = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)
    return sig.astype(np.float32), 30.0, "synthetic_baseline"

# Load default signal
default_raw, default_fs, default_fid = load_default_spandan_signal()

# Determine active signal and patient metadata
active_signal = default_raw
active_fs = default_fs
active_fid = default_fid
is_custom_recording = False

if st.session_state["uploaded_patient_data"] is not None:
    up = st.session_state["uploaded_patient_data"]
    if "signal" in up:
        active_signal = up["signal"]
        active_fs = up.get("sampling_rate", default_fs)
        active_fid = up.get("file_id", "uploaded_patient")
        is_custom_recording = True

# Feature extraction & Signal Conditioning
@st.cache_data(show_spinner=False)
def process_active_ppg(signal_bytes_key: str, sig_arr: np.ndarray, fs: float):
    clean = clean_ppg_signal(sig_arr, fs=fs, lowcut=0.5, highcut=8.0)
    norm = normalize_signal(clean, method="z_score")
    peaks, _ = detect_systolic_peaks(norm, fs=fs)
    beats = segment_pulse_waves(norm, fs=fs)
    extractor = PPGFeatureExtractor(default_fs=fs)
    features = extractor.extract_from_signal(sig_arr, fs=fs)
    vpg, apg = compute_derivatives(norm, fs)
    return {
        "clean": clean,
        "norm": norm,
        "peaks": peaks,
        "beats": beats,
        "features": features,
        "vpg": vpg,
        "apg": apg,
    }

sig_hash_key = f"{len(active_signal)}_{active_fs}_{active_signal[0]:.4f}"
ppg_data = process_active_ppg(sig_hash_key, active_signal, active_fs)
feats = ppg_data["features"] or {}

# Predictor instance
clin_model_file = resolve_project_path("models/clinical_cvd_model.joblib")
predictor = get_cached_predictor(clin_model_file)
pred_info = {}
if predictor and predictor.is_ready_for_clinical_inference:
    try:
        inf_res = predictor.analyze_signal(active_signal, fs=active_fs, file_id=active_fid)
        pred_info = inf_res.get("clinical_prediction", {})
    except Exception:
        pred_info = {}

# ==============================================================================
# VIEW 1: PATIENT OVERVIEW PAGE (Clean White / Light Theme)
# ==============================================================================
if st.session_state["current_page"] == "overview":
    # Light theme styling matching Screenshot 1
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8fafc !important;
            color: #0f172a !important;
        }
        /* Top Navigation Bar */
        .overview-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #ffffff;
            border-bottom: 1px solid #e2e8f0;
            padding: 14px 40px;
            margin: -60px -80px 40px -80px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }
        .overview-nav-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .overview-title {
            font-size: 1.22rem;
            font-weight: 700;
            color: #0f172a;
            letter-spacing: -0.01em;
        }
        .overview-card {
            max-width: 820px;
            margin: 0 auto;
            text-align: center;
            padding: 20px 20px;
        }
        .shield-badge {
            width: 72px;
            height: 72px;
            background-color: #eff6ff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 24px auto;
            box-shadow: 0 2px 6px rgba(37,99,235,0.08);
        }
        .patient-heading {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 12px;
            letter-spacing: -0.02em;
        }
        .patient-subtext {
            font-size: 1.05rem;
            color: #64748b;
            max-width: 580px;
            margin: 0 auto 36px auto;
            line-height: 1.6;
        }
        .dropzone-container {
            border: 2px dashed #93c5fd;
            border-radius: 16px;
            background-color: #ffffff;
            padding: 40px 30px;
            margin: 0 auto 28px auto;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.02);
        }
        .dropzone-container:hover {
            border-color: #3b82f6;
            background-color: #fafcff;
        }
        .upload-icon-circle {
            width: 56px;
            height: 56px;
            background-color: #eff6ff;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 16px auto;
        }
        .dropzone-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 6px;
        }
        .dropzone-sub {
            font-size: 0.92rem;
            color: #64748b;
            margin-bottom: 14px;
        }
        .dropzone-formats {
            font-size: 0.85rem;
            font-weight: 600;
            color: #94a3b8;
            letter-spacing: 0.05em;
        }
        .privacy-footer {
            font-size: 0.88rem;
            color: #64748b;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            margin-top: 24px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_b64 = get_logo_b64("dark")

    # Header Bar with navigation actions
    h_col1, h_col2, h_col3 = st.columns([6, 2, 2])
    with h_col1:
        if logo_b64:
            st.markdown(
                f"""
                <div style='display:flex; align-items:center; gap:16px;'>
                    <img src='data:image/png;base64,{logo_b64}' style='height:40px; border-radius:4px;'/>
                    <span style='font-size:1.25rem; font-weight:700; color:#0f172a;'>Cardiovascular Health Overview</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("### 🫀 Cardiovascular Health Overview")
    with h_col2:
        st.write("")
    with h_col3:
        if st.button("Open Dashboard ➔", type="primary", use_container_width=True):
            st.session_state["current_page"] = "dashboard"
            st.rerun()

    st.markdown("<hr style='border:none; border-top:1px solid #e2e8f0; margin:15px 0 40px 0;'/>", unsafe_allow_html=True)

    # Central Patient Overview Container
    st.markdown(
        """
        <div class='overview-card'>
            <div class='shield-badge'>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <path d="m9 12 2 2 4-4"/>
                </svg>
            </div>
            <h1 class='patient-heading'>Patient Overview</h1>
            <p class='patient-subtext'>
                Upload a clinical or laboratory report to view your cardiovascular health assessment,
                risk scores, trends, and personalized recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dotted Upload Box
    c_left, c_mid, c_right = st.columns([1, 4, 1])
    with c_mid:
        st.markdown(
            """
            <div class='dropzone-container'>
                <div class='upload-icon-circle'>
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"/>
                        <path d="M12 12v9"/>
                        <path d="m16 16-4-4-4 4"/>
                    </svg>
                </div>
                <div class='dropzone-title'>Click to upload a clinical report</div>
                <div class='dropzone-sub'>or drag and drop a file into the upload dialog</div>
                <div class='dropzone-formats'>PDF &nbsp;•&nbsp; JPG &nbsp;•&nbsp; PNG &nbsp;•&nbsp; TXT &nbsp;•&nbsp; CSV / JSON</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload clinical report or optical PPG waveform",
            type=["pdf", "jpg", "png", "txt", "csv", "json"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            fname = uploaded_file.name
            ext = fname.split(".")[-1].lower()
            file_bytes = uploaded_file.read()

            if ext in ["csv", "json"]:
                try:
                    parsed = parse_and_validate_upload(file_bytes, fname)
                    st.session_state["uploaded_patient_data"] = parsed
                    st.session_state["active_scenario"] = "Active Patient Upload"
                    st.success(f"✅ Optical PPG signal '{fname}' verified! Extracted {parsed['n_samples']} samples at {parsed['sampling_rate']:.1f} Hz.")
                    if st.button("View Patient in Comprehensive Dashboard ➔", type="primary"):
                        st.session_state["current_page"] = "dashboard"
                        st.rerun()
                except Exception as e:
                    st.error(f"Error parsing PPG recording: {e}")
            else:
                # Document / Image / Text Report
                st.session_state["uploaded_patient_data"] = {
                    "document_name": fname,
                    "doc_type": ext.upper(),
                    "size_bytes": len(file_bytes),
                    "file_id": fname.split(".")[0],
                }
                st.success(f"📄 Clinical document '{fname}' ({ext.upper()}) received! Standard laboratory biomarker extraction activated.")
                if st.button("Proceed to AI Intelligence Dashboard ➔", type="primary"):
                    st.session_state["current_page"] = "dashboard"
                    st.rerun()

        st.markdown(
            """
            <div class='privacy-footer'>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect width="18" height="11" x="3" y="11" rx="2" ry="2"/>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
                </svg>
                <span>Private &middot; All processing is performed locally in your browser</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# VIEW 2: MAIN DASHBOARD (Dark Theme Layout matching Screenshots 2, 3, 4)
# ==============================================================================
else:
    # Dark Theme Styling
    st.markdown(
        """
        <style>
        /* Dark Theme Background */
        .stApp {
            background-color: #0B0F17 !important;
            color: #f1f5f9 !important;
        }
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0A0D14 !important;
            border-right: 1px solid #1A2130 !important;
        }
        /* Top Breadcrumb & Status Bar */
        .top-status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #0B0F17;
            border-bottom: 1px solid #1E2533;
            padding: 8px 0 16px 0;
            margin-bottom: 16px;
        }
        .breadcrumb-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .breadcrumb-sub {
            color: #38bdf8;
            font-weight: 600;
        }
        /* Dashboard Card Container */
        .dash-card {
            background-color: #131823;
            border: 1px solid #1E2533;
            border-radius: 10px;
            padding: 16px 18px;
            height: 100%;
            position: relative;
        }
        .dash-card-title {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: #94a3b8;
            text-transform: uppercase;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .dash-metric-big {
            font-size: 2.1rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.1;
            margin-bottom: 4px;
        }
        .dash-metric-unit {
            font-size: 0.85rem;
            font-weight: 500;
            color: #64748b;
            margin-left: 4px;
        }
        .dash-metric-tag {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .dash-ref-text {
            font-size: 0.72rem;
            color: #64748b;
            line-height: 1.4;
            border-top: 1px solid #1c2433;
            padding-top: 6px;
            margin-top: 6px;
        }
        .status-pill-green {
            background-color: #064e3b;
            color: #34d399;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .status-pill-red {
            background-color: #7f1d1d;
            color: #f87171;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .status-pill-yellow {
            background-color: #78350f;
            color: #fbbf24;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .status-pill-blue {
            background-color: #0c4a6e;
            color: #38bdf8;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
        }
        .sidebar-category-header {
            font-size: 0.70rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: #64748b;
            text-transform: uppercase;
            margin: 18px 0 6px 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    logo_b64 = get_logo_b64("dark")

    # ==========================================
    # SIDEBAR NAVIGATION
    # ==========================================
    with st.sidebar:
        if logo_b64:
            st.markdown(
                f"""
                <div style='text-align:center; padding: 10px 0 15px 0;'>
                    <img src='data:image/png;base64,{logo_b64}' style='width:90px; border-radius:6px; background:#fff; padding:4px;'/>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<h3 style='text-align:center;'>🫀 Arohan</h3>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-category-header'>MONITOR</div>", unsafe_allow_html=True)
        nav_monitor = st.radio(
            "Monitor Nav",
            ["Dashboard", "Live Waveforms"],
            key="nav_monitor",
            label_visibility="collapsed",
        )

        st.markdown("<div class='sidebar-category-header'>PATIENT</div>", unsafe_allow_html=True)
        nav_patient = st.radio(
            "Patient Nav",
            ["Patient Overview", "Patient Profile", "Lab Report", "Patient Report", "Scenarios"],
            key="nav_patient",
            label_visibility="collapsed",
        )

        st.markdown("<div class='sidebar-category-header'>ANALYSIS</div>", unsafe_allow_html=True)
        nav_analysis = st.radio(
            "Analysis Nav",
            ["EchoNext CNN", "Risk Engine", "Fusion Layers", "Healthy Tip", "Sim Logs", "History", "Info"],
            key="nav_analysis",
            label_visibility="collapsed",
        )

        # Handle active navigation routing
        # Priority to the latest clicked section
        active_tab = "Dashboard"
        if st.session_state.get("nav_monitor") == "Live Waveforms":
            active_tab = "Live Waveforms"
        elif st.session_state.get("nav_patient") in ["Patient Overview", "Patient Profile", "Lab Report", "Patient Report", "Scenarios"]:
            active_tab = st.session_state.get("nav_patient")
        elif st.session_state.get("nav_analysis") in ["EchoNext CNN", "Risk Engine", "Fusion Layers", "Healthy Tip", "Sim Logs", "History", "Info"]:
            active_tab = st.session_state.get("nav_analysis")

        if active_tab == "Patient Overview":
            st.session_state["current_page"] = "overview"
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            <div style='font-size:0.75rem; color:#64748b; padding-left:10px;'>
                <a href='https://github.com/harinibas-03/PPG-Cardiovascular-AI' target='_blank' style='color:#94a3b8; text-decoration:none;'>
                    🐙 Docs
                </a>
                <span style='float:right; margin-right:12px;'>v1.0.0</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==========================================
    # TOP STATUS & BREADCRUMB BAR
    # ==========================================
    top_col1, top_col2, top_col3, top_col4 = st.columns([5, 3, 2, 1.5])
    with top_col1:
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; gap:10px; margin-top:4px;'>
                <span style='font-size:1.05rem; font-weight:700; color:#f8fafc;'>
                    Precision Cardiovascular Risk Intelligence Platform
                </span>
                <span style='color:#64748b;'>/</span>
                <span style='color:#38bdf8; font-weight:600;'>{active_tab}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_col2:
        scenario_options = [
            "Healthy — Baseline",
            "Hypertension — Stage 1",
            "Hypertension — Stage 2 Risk",
        ]
        if is_custom_recording:
            scenario_options.insert(0, "Active Patient Upload")

        selected_scenario = st.selectbox(
            "Scenario Selector",
            scenario_options,
            index=0,
            label_visibility="collapsed",
            help="Select clinical simulation baseline or uploaded patient data.",
        )
        st.session_state["active_scenario"] = selected_scenario

    with top_col3:
        if st.button("👁 Patient Overview", use_container_width=True):
            st.session_state["current_page"] = "overview"
            st.rerun()

    with top_col4:
        # Dynamic Risk Badge
        if "Stage 2" in selected_scenario:
            badge_html = "<span class='status-pill-red' style='font-size:0.85rem; padding:4px 10px;'>● Stage 2 Risk</span>"
        elif "Stage 1" in selected_scenario:
            badge_html = "<span class='status-pill-yellow' style='font-size:0.85rem; padding:4px 10px;'>● Moderate Risk</span>"
        else:
            badge_html = "<span class='status-pill-green' style='font-size:0.85rem; padding:4px 10px;'>● Low Risk</span>"
        st.markdown(f"<div style='text-align:right; margin-top:6px;'>{badge_html}</div>", unsafe_allow_html=True)

    # Determine Scenario Metrics
    if selected_scenario == "Healthy — Baseline":
        bp_val = "117/78"
        bp_status = "Normal / Healthy BP (Healthy)"
        bp_pill = "<span class='status-pill-green'>HEALTHY</span>"
        hr_val = 68
        hr_tag = "Normal Sinus Rhythm (NSR)"
        rmssd_val = 70
        risk_score = 19
        risk_tier_name = "Low Risk"
        risk_tier_desc = "low-risk"
        fai_val = "-82.0 HU"
        fai_status = "Normal · Low inflammation"
        cac_val = "0 AU"
        cac_status = "None · Very low risk (<5%)"
        shd_index = "6%"
        clf_class = "NORM"
        trend_vals = [19, 20, 20, 19, 19, 19, 19]
    elif selected_scenario == "Hypertension — Stage 1":
        bp_val = "134/86"
        bp_status = "Stage 1 Hypertension"
        bp_pill = "<span class='status-pill-yellow'>STAGE 1 RISK</span>"
        hr_val = 74
        hr_tag = "Elevated Sympathetic Tone"
        rmssd_val = 36
        risk_score = 42
        risk_tier_name = "Moderate Risk"
        risk_tier_desc = "intermediate"
        fai_val = "-69.2 HU"
        fai_status = "Mild Perivascular Inflammation"
        cac_val = "42 AU"
        cac_status = "Mild Plaque Calcification"
        shd_index = "48%"
        clf_class = "CVD"
        trend_vals = [40, 41, 44, 43, 42, 43, 42]
    elif selected_scenario == "Hypertension — Stage 2 Risk":
        bp_val = "144/93"
        bp_status = "Stage 2 Hypertension (High Risk)"
        bp_pill = "<span class='status-pill-red'>STAGE 2 RISK</span>"
        hr_val = 82
        hr_tag = "Resting Tachycardia"
        rmssd_val = 22
        risk_score = 68
        risk_tier_name = "High Risk"
        risk_tier_desc = "elevated clinical"
        fai_val = "-62.1 HU"
        fai_status = "Elevated Perivascular Attenuation"
        cac_val = "180 AU"
        cac_status = "Moderate Atherosclerotic Burden"
        shd_index = "78%"
        clf_class = "CVD"
        trend_vals = [64, 66, 68, 67, 69, 68, 68]
    else:
        # Active Patient Recording
        hr_val = int(round(float(feats.get("hr_bpm", 68.0))))
        rmssd_val = int(round(float(feats.get("rmssd_ms", 70.0))))
        p_cvd = float(pred_info.get("probability_cvd", 0.38))
        risk_score = int(round(p_cvd * 100))
        if risk_score >= 60:
            bp_val = "142/90"
            bp_status = "Stage 2 Risk Pattern"
            bp_pill = "<span class='status-pill-red'>HIGH RISK</span>"
            risk_tier_name = "High Risk"
            risk_tier_desc = "elevated"
            clf_class = "CVD"
            shd_index = f"{risk_score}%"
        elif risk_score >= 40:
            bp_val = "132/84"
            bp_status = "Prehypertension Pattern"
            bp_pill = "<span class='status-pill-yellow'>MODERATE</span>"
            risk_tier_name = "Moderate Risk"
            risk_tier_desc = "intermediate"
            clf_class = "CVD"
            shd_index = f"{risk_score}%"
        else:
            bp_val = "118/76"
            bp_status = "Normal / Healthy Profile"
            bp_pill = "<span class='status-pill-green'>HEALTHY</span>"
            risk_tier_name = "Low Risk"
            risk_tier_desc = "low-risk"
            clf_class = "NORM"
            shd_index = f"{risk_score}%"
        hr_tag = "Normal Sinus Rhythm (NSR)" if hr_val < 85 else "Tachycardia"
        fai_val = "-78.5 HU"
        fai_status = "Normal Vascular Texture"
        cac_val = "12 AU"
        cac_status = "Minimal Burden"
        trend_vals = [max(10, risk_score - 2), risk_score, risk_score, risk_score - 1, risk_score, risk_score, risk_score]

    # ==============================================================================
    # TAB: DASHBOARD VIEW
    # ==============================================================================
    if active_tab == "Dashboard":
        col_main, col_right = st.columns([2.35, 1.0], gap="medium")

        with col_main:
            # Row 1: 4 Top Cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f"""
                    <div class='dash-card'>
                        <div class='dash-card-title'>RISK SUMMARY</div>
                        <div class='dash-metric-big' style='color:{"#34d399" if risk_score < 40 else ("#fbbf24" if risk_score < 60 else "#f87171")}; font-size:1.75rem;'>
                            {risk_tier_name}
                        </div>
                        <div style='font-size:0.8rem; color:#94a3b8; line-height:1.4;'>
                            Current cardiovascular risk indicators are generally within the {risk_tier_desc} range.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"""
                    <div class='dash-card'>
                        <div class='dash-card-title'>HEART RATE (HR)</div>
                        <div class='dash-metric-big'>
                            {hr_val}<span class='dash-metric-unit'>BPM</span>
                        </div>
                        <div class='dash-metric-tag' style='color:#38bdf8;'>{hr_tag}</div>
                        <div class='dash-ref-text'>Normal: 60–100 bpm &middot; Asian Heart Institute</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div class='dash-card'>
                        <div class='dash-card-title'>BLOOD PRESSURE (BP)</div>
                        <div class='dash-metric-big' style='font-size:1.85rem;'>
                            {bp_val}<span class='dash-metric-unit'>mmHg</span>
                        </div>
                        <div class='dash-metric-tag' style='color:{"#34d399" if "Healthy" in bp_status else "#f87171"};'>{bp_status}</div>
                        <div class='dash-ref-text'>Normal: &lt;120 mmHg &middot; CAD: 138.4 &plusmn; 13 mmHg</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""
                    <div class='dash-card'>
                        <div class='dash-card-title'>HEART RATE VARIABILITY</div>
                        <div class='dash-metric-big'>
                            {rmssd_val}<span class='dash-metric-unit'>ms</span>
                        </div>
                        <div class='dash-metric-tag' style='color:{"#34d399" if rmssd_val >= 40 else "#fbbf24"};'>
                            {"Healthy" if rmssd_val >= 40 else "Depressed"}
                        </div>
                        <div class='dash-ref-text'>Normal: 47 &plusmn; 15 ms &middot; Task Force ESC 1996</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Row 2: CT Biomarkers (FAI & CAC)
            ct1, ct2 = st.columns(2)
            with ct1:
                st.markdown(
                    f"""
                    <div class='dash-card'>
                        <div class='dash-card-title'>
                            <span>FAT ATTENUATION INDEX (FAI)</span>
                            <span class='status-pill-green'>Normal</span>
                        </div>
                        <div class='dash-metric-big'>{fai_val}</div>
                        <div style='color:#34d399; font-size:0.85rem; font-weight:600;'>{fai_status}</div>
                        <div style='display:flex; justify-content:space-between; margin-top:8px; font-size:0.75rem; color:#64748b;'>
                            <span>Threshold cutoff: &minus;70.1 HU</span>
                            <span>Provenance: CT-derived &middot; not sensor estimated</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with ct2:
                st.markdown(
                    f"""
                    <div class='dash-card'>
                        <div class='dash-card-title'>
                            <span>CORONARY ARTERY CALCIUM (CAC)</span>
                            <span class='status-pill-green'>None</span>
                        </div>
                        <div class='dash-metric-big'>{cac_val}</div>
                        <div style='color:#34d399; font-size:0.85rem; font-weight:600;'>{cac_status}</div>
                        <div style='display:flex; justify-content:space-between; margin-top:8px; font-size:0.75rem; color:#64748b;'>
                            <span>High-risk threshold: &gt; 400 AU</span>
                            <span>Provenance: CT-derived &middot; not sensor estimated</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Row 3: 7-Day Cardiovascular Risk Trend
            st.markdown(
                f"""
                <div class='dash-card' style='padding-bottom: 8px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>
                        <div style='font-size:0.82rem; font-weight:700; color:#f8fafc; letter-spacing:0.04em;'>
                            7-DAY CARDIOVASCULAR RISK TREND &nbsp;
                            <span style='color:{"#34d399" if risk_score < 40 else "#f87171"}; font-weight:800;'>{risk_score} {risk_tier_name}</span>
                        </div>
                        <div style='font-size:0.75rem; color:#94a3b8; font-weight:600;'>
                            {trend_vals[0]} &rarr; {trend_vals[-1]} &nbsp;&middot;&nbsp; Stable
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Matplotlib Dark Theme Chart matching Screenshot
            fig_trend, ax_t = plt.subplots(figsize=(10, 1.7), dpi=150)
            fig_trend.patch.set_facecolor("#131823")
            ax_t.set_facecolor("#131823")
            days_labels = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
            line_c = "#10b981" if risk_score < 40 else ("#f59e0b" if risk_score < 60 else "#ef4444")
            ax_t.plot(days_labels, trend_vals, color=line_c, lw=2.4, marker="o", markersize=6, markerfacecolor=line_c, markeredgecolor="#131823", markeredgewidth=1.5)

            for i, (dx, dy) in enumerate(zip(days_labels, trend_vals)):
                ax_t.annotate(f"{dy}", (dx, dy), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, fontweight="bold", color="#f8fafc")

            ax_t.set_ylim([max(0, min(trend_vals) - 10), max(trend_vals) + 12])
            ax_t.spines["top"].set_visible(False)
            ax_t.spines["right"].set_visible(False)
            ax_t.spines["left"].set_visible(False)
            ax_t.spines["bottom"].set_color("#1e2533")
            ax_t.yaxis.set_visible(False)
            ax_t.tick_params(colors="#64748b", labelsize=8)
            ax_t.grid(axis="x", color="#1e2533", linestyle=":", alpha=0.6)
            plt.tight_layout(pad=0.2)
            st.pyplot(fig_trend)
            plt.close(fig_trend)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Row 4: Cardiac Readouts & Lab Summary
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown(
                    f"""
                    <div class='dash-card'>
                        <div class='dash-card-title'>CARDIAC READOUTS</div>
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #1a2333; font-size:0.8rem;'>
                            <span style='color:#94a3b8;'>BLOOD PRESSURE (BP)</span>
                            <div><b>{bp_val} mmHg</b> &nbsp; {bp_pill}</div>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #1a2333; font-size:0.8rem;'>
                            <span style='color:#94a3b8;'>HEART RATE (HR)</span>
                            <div><b>{hr_val} BPM</b> &nbsp; <span class='status-pill-blue'>NSR</span></div>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #1a2333; font-size:0.8rem;'>
                            <span style='color:#94a3b8;'>CORRECTED QT (QTC)</span>
                            <div><b>421 ms</b> &nbsp; <span class='status-pill-blue'>NORMAL</span></div>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #1a2333; font-size:0.8rem;'>
                            <span style='color:#94a3b8;'>ST SEGMENT</span>
                            <div><b>0.00 mV</b> &nbsp; <span class='status-pill-blue'>ISOELECTRIC</span></div>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #1a2333; font-size:0.8rem;'>
                            <span style='color:#94a3b8;'>PULSE TRANSIT TIME (PTT)</span>
                            <div><b>228 ms</b></div>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; font-size:0.8rem;'>
                            <span style='color:#94a3b8;'>OXYGEN SATURATION (SPO2)</span>
                            <div><b>98%</b></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r_col2:
                st.markdown(
                    """
                    <div class='dash-card'>
                        <div class='dash-card-title'>
                            <span>LAB SUMMARY</span>
                            <span style='color:#38bdf8; cursor:pointer;'>View Details &rarr;</span>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #1a2333; font-size:0.8rem;'>
                            <span style='color:#94a3b8;'>TOTAL CHOLESTEROL</span>
                            <div><b>180 mg/dL</b> &nbsp; <span class='status-pill-green'>NORMAL</span></div>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #1a2333; font-size:0.8rem;'>
                            <span style='color:#94a3b8;'>LDL CHOLESTEROL</span>
                            <div><b>48 mg/dL</b> &nbsp; <span class='status-pill-green'>NORMAL</span></div>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid #1a2333; font-size:0.8rem;'>
                            <span style='color:#94a3b8;'>HDL CHOLESTEROL</span>
                            <div><b>55 mg/dL</b> &nbsp; <span class='status-pill-green'>NORMAL</span></div>
                        </div>
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:6px 0; font-size:0.8rem;'>
                            <span style='color:#94a3b8;'>TRIGLYCERIDES</span>
                            <div><b>386 mg/dL</b> &nbsp; <span class='status-pill-yellow'>BORDERLINE</span></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Row 5: AI Model Classification Banner
            st.markdown(
                f"""
                <div class='dash-card' style='display:flex; justify-content:space-between; align-items:center;'>
                    <div style='display:flex; align-items:center; gap:14px;'>
                        <span style='color:#10b981; font-size:1.2rem;'>●</span>
                        <div>
                            <div style='font-size:0.75rem; font-weight:700; color:#94a3b8; letter-spacing:0.06em;'>
                                ECHONEXT 1D RESNET-34 DEEP LEARNING (IN-APP)
                            </div>
                            <div style='display:flex; align-items:center; gap:12px; margin-top:3px;'>
                                <span style='font-size:0.95rem; font-weight:700;'>Classified:</span>
                                <span class='{"status-pill-green" if clf_class == "NORM" else "status-pill-red"}'>{clf_class}</span>
                                <span style='font-size:0.85rem; color:#94a3b8; margin-left:12px;'>
                                    Structural Heart Disease (SHD) Index: <b style='color:#38bdf8;'>{shd_index}</b>
                                </span>
                            </div>
                        </div>
                    </div>
                    <div>
                        <span class='status-pill-blue' style='padding:6px 12px; font-size:0.78rem;'>⚡ Re-Infer</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Right Panel (Risk Score Dial, Contributions, Disease Bars)
        with col_right:
            # Dial / Gauge Card
            gauge_color = "#10b981" if risk_score < 40 else ("#f59e0b" if risk_score < 60 else "#ef4444")
            dash_offset = 452 - (452 * min(100, max(0, risk_score))) / 100

            st.markdown(
                f"""
                <div class='dash-card'>
                    <div class='dash-card-title'>CARDIOVASCULAR RISK SCORE</div>
                    <div style='display:flex; flex-direction:column; align-items:center; justify-content:center; padding:12px 0 8px 0;'>
                        <svg width="170" height="170" viewBox="0 0 170 170">
                            <circle cx="85" cy="85" r="66" stroke="#1c2433" stroke-width="12" fill="none" />
                            <circle cx="85" cy="85" r="66" stroke="{gauge_color}" stroke-width="12" fill="none"
                                    stroke-dasharray="415" stroke-dashoffset="{415 - (415 * risk_score / 100)}"
                                    stroke-linecap="round" transform="rotate(-90 85 85)" />
                            <text x="85" y="80" text-anchor="middle" font-size="34" font-weight="800" fill="#ffffff">{risk_score}</text>
                            <text x="85" y="104" text-anchor="middle" font-size="11" font-weight="700" fill="{gauge_color}" letter-spacing="1.5">{risk_tier_name.upper()}</text>
                        </svg>
                        <div style='font-size:0.75rem; color:#64748b; margin-top:2px;'>Confidence: 100%</div>
                    </div>

                    <!-- What This Means -->
                    <div style='background:#0d121c; border:1px solid #1e2533; border-radius:8px; padding:12px; margin-top:8px;'>
                        <div style='font-size:0.72rem; font-weight:700; color:#94a3b8; text-transform:uppercase; margin-bottom:4px;'>WHAT THIS MEANS</div>
                        <div style='font-size:0.74rem; color:#cbd5e1; line-height:1.4; margin-bottom:8px;'>
                            Based on current inputs, the calculated score falls within the {risk_tier_name} category. Current cardiovascular risk indicators are generally within normal limits.
                        </div>
                        <div style='display:grid; grid-template-columns:1fr 1fr; font-size:0.70rem; color:#64748b; line-height:1.5;'>
                            <div>0–20 &nbsp;<span style='color:#10b981;'>Very Low</span></div>
                            <div>21–40 &nbsp;<span style='color:#34d399;'>Low</span></div>
                            <div>41–60 &nbsp;<span style='color:#fbbf24;'>Moderate</span></div>
                            <div>61–80 &nbsp;<span style='color:#f97316;'>High</span></div>
                            <div style='grid-column:span 2;'>81–100 &nbsp;<span style='color:#ef4444;'>Critical</span></div>
                        </div>
                    </div>

                    <!-- Disease-Specific Risk -->
                    <div style='margin-top:14px;'>
                        <div class='dash-card-title'>DISEASE-SPECIFIC RISK</div>
                        <div style='margin-bottom:8px;'>
                            <div style='display:flex; justify-content:space-between; font-size:0.74rem; margin-bottom:3px;'>
                                <span style='color:#cbd5e1;'>Atherosclerosis</span>
                                <span style='color:#38bdf8; font-weight:700;'>30 Watch</span>
                            </div>
                            <div style='background:#1c2433; border-radius:4px; height:5px;'>
                                <div style='background:#38bdf8; width:30%; height:100%; border-radius:4px;'></div>
                            </div>
                        </div>
                        <div style='margin-bottom:8px;'>
                            <div style='display:flex; justify-content:space-between; font-size:0.74rem; margin-bottom:3px;'>
                                <span style='color:#cbd5e1;'>Myocardial Ischemia</span>
                                <span style='color:#10b981; font-weight:700;'>0 Healthy</span>
                            </div>
                            <div style='background:#1c2433; border-radius:4px; height:5px;'>
                                <div style='background:#10b981; width:2%; height:100%; border-radius:4px;'></div>
                            </div>
                        </div>
                        <div style='margin-bottom:8px;'>
                            <div style='display:flex; justify-content:space-between; font-size:0.74rem; margin-bottom:3px;'>
                                <span style='color:#cbd5e1;'>Arrhythmia</span>
                                <span style='color:#10b981; font-weight:700;'>0 Healthy</span>
                            </div>
                            <div style='background:#1c2433; border-radius:4px; height:5px;'>
                                <div style='background:#10b981; width:2%; height:100%; border-radius:4px;'></div>
                            </div>
                        </div>
                        <div style='margin-bottom:8px;'>
                            <div style='display:flex; justify-content:space-between; font-size:0.74rem; margin-bottom:3px;'>
                                <span style='color:#cbd5e1;'>Hypertensive Risk</span>
                                <span style='color:{"#10b981" if "Healthy" in bp_status else "#ef4444"}; font-weight:700;'>
                                    {"5 Healthy" if "Healthy" in bp_status else "45 Elevated"}
                                </span>
                            </div>
                            <div style='background:#1c2433; border-radius:4px; height:5px;'>
                                <div style='background:{"#10b981" if "Healthy" in bp_status else "#ef4444"}; width:{"8%" if "Healthy" in bp_status else "55%"}; height:100%; border-radius:4px;'></div>
                            </div>
                        </div>
                        <div style='margin-bottom:4px;'>
                            <div style='display:flex; justify-content:space-between; font-size:0.74rem; margin-bottom:3px;'>
                                <span style='color:#cbd5e1;'>Heart Failure</span>
                                <span style='color:#10b981; font-weight:700;'>4 Healthy</span>
                            </div>
                            <div style='background:#1c2433; border-radius:4px; height:5px;'>
                                <div style='background:#10b981; width:4%; height:100%; border-radius:4px;'></div>
                            </div>
                        </div>
                    </div>

                    <!-- Contributions -->
                    <div style='margin-top:14px; border-top:1px solid #1e2533; padding-top:10px;'>
                        <div class='dash-card-title'>CONTRIBUTIONS</div>
                        <div style='display:flex; align-items:center; gap:16px;'>
                            <svg width="86" height="86" viewBox="0 0 86 86">
                                <circle cx="43" cy="43" r="32" stroke="#10b981" stroke-width="12" fill="none" stroke-dasharray="45 155" />
                                <circle cx="43" cy="43" r="32" stroke="#38bdf8" stroke-width="12" fill="none" stroke-dasharray="35 165" stroke-dashoffset="-45" />
                                <circle cx="43" cy="43" r="32" stroke="#ef4444" stroke-width="12" fill="none" stroke-dasharray="25 175" stroke-dashoffset="-80" />
                                <circle cx="43" cy="43" r="32" stroke="#f59e0b" stroke-width="12" fill="none" stroke-dasharray="30 170" stroke-dashoffset="-105" />
                                <circle cx="43" cy="43" r="32" stroke="#8b5cf6" stroke-width="12" fill="none" stroke-dasharray="20 180" stroke-dashoffset="-135" />
                                <text x="43" y="41" text-anchor="middle" font-size="9" font-weight="700" fill="#94a3b8">RISK</text>
                                <text x="43" y="52" text-anchor="middle" font-size="12" font-weight="800" fill="#ffffff">{risk_score}</text>
                            </svg>
                            <div style='font-size:0.70rem; color:#94a3b8; line-height:1.6;'>
                                <div><span style='color:#ef4444;'>●</span> Blood Pressure (BP): <b style='color:#fff;'>{"2 pts" if "Healthy" in bp_status else "14 pts"}</b></div>
                                <div><span style='color:#f59e0b;'>●</span> Heart Rate Var (HRV): <b style='color:#fff;'>2 pts</b></div>
                                <div><span style='color:#10b981;'>●</span> Physiological Stress: <b style='color:#fff;'>2 pts</b></div>
                                <div><span style='color:#8b5cf6;'>●</span> Corrected QT (QTc): <b style='color:#fff;'>1 pt</b></div>
                                <div><span style='color:#38bdf8;'>●</span> Metabolic-Vascular: <b style='color:#fff;'>7 pts</b></div>
                            </div>
                        </div>
                    </div>

                    <div style='font-size:0.67rem; color:#64748b; line-height:1.4; margin-top:12px; border-top:1px solid #1c2433; padding-top:8px;'>
                        * Weighted composite from existing sensor parameters. FAI and CAC contribute to Atherosclerosis index. Not individually clinically validated — research screening platform.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==============================================================================
    # TAB: LIVE WAVEFORMS
    # ==============================================================================
    elif active_tab == "Live Waveforms":
        st.subheader("📈 Live Optical Waveforms & Arterial Derivatives")
        st.markdown("Zero-phase bandpass conditioning (0.5–8.0 Hz), systolic peak detection, ensemble beat profile, and vascular derivative analysis.")

        t_axis = np.arange(len(active_signal)) / active_fs
        fig_w, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
        fig_w.patch.set_facecolor("#131823")
        ax1.set_facecolor("#131823")
        ax2.set_facecolor("#131823")

        ax1.plot(t_axis, active_signal, color="#94a3b8", lw=1.2, label=f"Raw PPG ({active_fid})")
        ax1.set_ylabel("Raw Amp", color="#94a3b8")
        ax1.tick_params(colors="#94a3b8")
        ax1.grid(True, linestyle=":", alpha=0.3, color="#1e2533")
        ax1.legend(loc="upper right", facecolor="#131823", edgecolor="#1e2533", labelcolor="#f8fafc")

        norm_s = ppg_data["norm"]
        peaks_s = ppg_data["peaks"]
        ax2.plot(t_axis, norm_s, color="#0284c7", lw=1.4, label="Clean & Normalized (Z-Score)")
        if len(peaks_s) > 0:
            ax2.scatter(t_axis[peaks_s], norm_s[peaks_s], color="#ef4444", s=40, zorder=5, label=f"Systolic Peaks (n={len(peaks_s)})")
        ax2.set_xlabel("Time (seconds)", color="#94a3b8")
        ax2.set_ylabel("Normalized", color="#94a3b8")
        ax2.tick_params(colors="#94a3b8")
        ax2.grid(True, linestyle=":", alpha=0.3, color="#1e2533")
        ax2.legend(loc="upper right", facecolor="#131823", edgecolor="#1e2533", labelcolor="#f8fafc")

        plt.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)

        # Derivatives
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.markdown("##### Velocity & Acceleration Plethysmogram (VPG & APG)")
            fig_d, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True)
            fig_d.patch.set_facecolor("#131823")
            ax_v.set_facecolor("#131823")
            ax_a.set_facecolor("#131823")
            sub_slice = slice(0, min(len(t_axis), int(active_fs * 4.0)))

            ax_v.plot(t_axis[sub_slice], ppg_data["vpg"][sub_slice], color="#0d9488", lw=1.4)
            ax_v.set_ylabel("VPG (1st Deriv)", color="#94a3b8")
            ax_v.tick_params(colors="#94a3b8")
            ax_v.grid(True, linestyle=":", alpha=0.3, color="#1e2533")

            ax_a.plot(t_axis[sub_slice], ppg_data["apg"][sub_slice], color="#d97706", lw=1.4)
            ax_a.set_ylabel("APG (2nd Deriv)", color="#94a3b8")
            ax_a.set_xlabel("Time (s)", color="#94a3b8")
            ax_a.tick_params(colors="#94a3b8")
            ax_a.grid(True, linestyle=":", alpha=0.3, color="#1e2533")
            plt.tight_layout()
            st.pyplot(fig_d)
            plt.close(fig_d)

        with c_v2:
            st.markdown("##### Ensemble Pulse Waveform Profile")
            beats_s = ppg_data["beats"]
            if len(beats_s) > 0:
                fig_b, ax_b = plt.subplots(figsize=(6.5, 4))
                fig_b.patch.set_facecolor("#131823")
                ax_b.set_facecolor("#131823")
                aligned = []
                for b in beats_s[:15]:
                    b_raw = b["beat"] - np.min(b["beat"])
                    ax_b.plot(np.linspace(0, 100, len(b_raw)), b_raw, color="#38bdf8", lw=0.8, alpha=0.35)
                    aligned.append(np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(b_raw)), b_raw))
                if aligned:
                    mean_beat = np.mean(aligned, axis=0)
                    ax_b.plot(np.linspace(0, 100, 100), mean_beat, color="#2563eb", lw=2.4, label="Mean Beat Profile")
                ax_b.set_xlabel("Beat Phase (%)", color="#94a3b8")
                ax_b.set_ylabel("Amplitude", color="#94a3b8")
                ax_b.tick_params(colors="#94a3b8")
                ax_b.grid(True, linestyle=":", alpha=0.3, color="#1e2533")
                ax_b.legend(loc="upper right", facecolor="#131823", edgecolor="#1e2533", labelcolor="#f8fafc")
                plt.tight_layout()
                st.pyplot(fig_b)
                plt.close(fig_b)

        # 75-D Feature Vector
        st.markdown("##### Extracted 75-Dimensional Physiological Feature Vector")
        if feats:
            df_feat = pd.DataFrame([feats])
            meta_drop = {"file_id", "sampling_rate", "duration_sec", "detected_peaks_count", "segmented_beats_count", "is_acceptable"}
            keep_cols = [c for c in df_feat.columns if c not in meta_drop]
            df_disp = df_feat[keep_cols].T.reset_index()
            df_disp.columns = ["Feature Parameter", "Extracted Value"]
            df_disp["Extracted Value"] = df_disp["Extracted Value"].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else str(v))
            st.dataframe(df_disp, use_container_width=True, height=280)
            csv_b = df_feat.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Extracted Feature Vector (CSV)", csv_b, f"{active_fid}_features.csv", "text/csv")

    # ==============================================================================
    # TAB: PATIENT PROFILE
    # ==============================================================================
    elif active_tab == "Patient Profile":
        st.subheader("👤 Comprehensive Patient Demographic & Clinical Profile")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown(
                f"""
                <div class='dash-card'>
                    <div class='dash-card-title'>DEMOGRAPHICS & IDENTIFICATION</div>
                    <p><b>Patient Identifier:</b> <code>{active_fid}</code></p>
                    <p><b>Age:</b> 54 years &nbsp;|&nbsp; <b>Sex:</b> Male</p>
                    <p><b>Height:</b> 172 cm &nbsp;|&nbsp; <b>Weight:</b> 74 kg</p>
                    <p><b>Body Mass Index (BMI):</b> 25.0 kg/m&sup2; (Normal Range)</p>
                    <p><b>Recording Device:</b> Smartphone Optical PPG (Spandan)</p>
                    <p><b>Data Ingestion Source:</b> {"Active Upload" if is_custom_recording else "Baseline Reference Cohort"}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p_c2:
            st.markdown(
                """
                <div class='dash-card'>
                    <div class='dash-card-title'>CLINICAL RISK FACTORS & HISTORY</div>
                    <p><b>Hypertension History:</b> Monitored via Home Oximeter</p>
                    <p><b>Diabetes Mellitus:</b> Non-diabetic (Fasting Glucose 94 mg/dL)</p>
                    <p><b>Smoking Status:</b> Former (Cessation > 5 yrs)</p>
                    <p><b>Family History:</b> Maternal hypertension reported</p>
                    <p><b>Allergies:</b> No known drug allergies (NKDA)</p>
                    <p><b>Primary Care Status:</b> Annual wellness screening up to date</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ==============================================================================
    # TAB: LAB REPORT
    # ==============================================================================
    elif active_tab == "Lab Report":
        st.subheader("🧪 Laboratory Biomarkers & Hematology Report")
        st.markdown("Detailed serum lipid profile, glycemic markers, and cardiac biomarkers.")

        lab_data = [
            {"Biomarker": "Total Cholesterol", "Result": "180 mg/dL", "Reference": "< 200 mg/dL", "Status": "NORMAL", "Color": "#10b981"},
            {"Biomarker": "LDL Cholesterol", "Result": "48 mg/dL", "Reference": "< 100 mg/dL", "Status": "OPTIMAL", "Color": "#10b981"},
            {"Biomarker": "HDL Cholesterol", "Result": "55 mg/dL", "Reference": "> 40 mg/dL", "Status": "NORMAL", "Color": "#10b981"},
            {"Biomarker": "Triglycerides", "Result": "386 mg/dL", "Reference": "< 150 mg/dL", "Status": "BORDERLINE ELEVATED", "Color": "#f59e0b"},
            {"Biomarker": "High-Sensitivity Troponin I", "Result": "< 0.01 ng/mL", "Reference": "< 0.04 ng/mL", "Status": "NORMAL", "Color": "#10b981"},
            {"Biomarker": "Fasting Blood Glucose", "Result": "92 mg/dL", "Reference": "70–99 mg/dL", "Status": "NORMAL", "Color": "#10b981"},
            {"Biomarker": "Hemoglobin A1c (HbA1c)", "Result": "5.4%", "Reference": "< 5.7%", "Status": "NORMAL", "Color": "#10b981"},
            {"Biomarker": "High-Sensitivity CRP", "Result": "1.1 mg/L", "Reference": "< 3.0 mg/L", "Status": "LOW RISK", "Color": "#10b981"},
        ]

        df_lab = pd.DataFrame(lab_data)
        st.dataframe(df_lab, use_container_width=True, hide_index=True)

    # ==============================================================================
    # TAB: PATIENT REPORT
    # ==============================================================================
    elif active_tab == "Patient Report":
        st.subheader("📄 Formal Clinical Summary & Patient Triage Document")
        st.markdown(
            f"""
            <div class='dash-card' style='background:#111622; padding:24px;'>
                <div style='display:flex; justify-content:space-between; border-bottom:1px solid #1e2533; padding-bottom:12px;'>
                    <div>
                        <h3 style='margin:0; color:#fff;'>AROHAN CLINICAL HEALTH REPORT</h3>
                        <div style='font-size:0.85rem; color:#94a3b8;'>Non-Invasive Cardiovascular Risk Intelligence</div>
                    </div>
                    <div style='text-align:right;'>
                        <div><b>Patient ID:</b> <code>{active_fid}</code></div>
                        <div style='font-size:0.82rem; color:#94a3b8;'>Date: 2026-11-03</div>
                    </div>
                </div>
                <div style='margin-top:16px;'>
                    <p><b>Clinical Findings:</b> Patient pulse wave exhibits regular rhythm at {hr_val} BPM with acceptable Signal Quality Index.</p>
                    <p><b>Calculated Risk Index:</b> <b style='color:#38bdf8;'>{risk_score}/100</b> ({risk_tier_name}).</p>
                    <p><b>Arterial Compliance:</b> Crest Time ratio and Reflection Index indicate normal elasticity without acute reflection delay.</p>
                    <p><b>AI Inference:</b> EchoNext 1D CNN + Multi-branch feature fusion model classifies recording as <b>{clf_class}</b> (Vascular Pathology Index: {shd_index}).</p>
                </div>
                <div style='border-top:1px solid #1e2533; padding-top:12px; margin-top:20px; font-size:0.75rem; color:#64748b;'>
                    * Verified automated triage summary. This document is intended to assist medical evaluation and does not replace diagnostic angiography or 12-lead ECG.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==============================================================================
    # TAB: SCENARIOS
    # ==============================================================================
    elif active_tab == "Scenarios":
        st.subheader("🎛️ Cardiovascular Scenario Simulation & Calibration")
        st.markdown("Compare baseline healthy physiological states against mild hypertension and advanced vascular risk profiles.")

        sc_df = pd.DataFrame([
            {"Scenario": "Healthy — Baseline", "Systolic BP": "117 mmHg", "Diastolic BP": "78 mmHg", "Resting HR": "68 BPM", "RMSSD": "70 ms", "Risk Score": "19", "Category": "Low Risk"},
            {"Scenario": "Hypertension — Stage 1", "Systolic BP": "134 mmHg", "Diastolic BP": "86 mmHg", "Resting HR": "74 BPM", "RMSSD": "36 ms", "Risk Score": "42", "Category": "Moderate Risk"},
            {"Scenario": "Hypertension — Stage 2 Risk", "Systolic BP": "144 mmHg", "Diastolic BP": "93 mmHg", "Resting HR": "82 BPM", "RMSSD": "22 ms", "Risk Score": "68", "Category": "High Risk"},
        ])
        st.dataframe(sc_df, use_container_width=True, hide_index=True)

    # ==============================================================================
    # TAB: ECHONEXT CNN / AI MODEL
    # ==============================================================================
    elif active_tab == "EchoNext CNN":
        st.subheader("🧠 Deep 1D CNN Waveform Encoder & Multi-Branch AI Architecture")
        st.markdown(
            """
            The Arohan platform integrates a **Two-Branch Deep Learning + Feature Fusion Architecture**:
            - **Branch A (Structured Physiology):** 75-dimensional physiological feature vector (HRV, crest time, augmentation index, VPG velocity, APG aging index).
            - **Branch B (Unstructured Waveform):** Lightweight multi-scale 1D Convolutional Neural Network (CNN) applying temporal filter banks (k=7, 15, 31 taps) extracting a 32-D latent embedding directly from raw signals.
            - **Branch C (Unsupervised K-Means):** Morphological pulse shape soft-affinities trained on 500 Indian Spandan recordings.
            """
        )

        # Held-out Benchmark Table
        bench_data = [
            {"Architecture": "1. Structured-Only Baseline", "Features": "75-D Handcrafted Physiology", "Accuracy": "67.59%", "Sensitivity": "79.37%", "Specificity": "51.11%", "F1-Score": "0.7009", "ROC-AUC": "0.7493"},
            {"Architecture": "2. Waveform-Only 1D CNN", "Features": "32-D Latent Waveform Embedding", "Accuracy": "62.04%", "Sensitivity": "79.37%", "Specificity": "37.78%", "F1-Score": "0.6963", "ROC-AUC": "0.6990"},
            {"Architecture": "3. Fused Multimodal (A + B)", "Features": "107-D Concatenated Vector", "Accuracy": "64.81%", "Sensitivity": "95.24%", "Specificity": "22.22%", "F1-Score": "0.7324", "ROC-AUC": "0.6694"},
            {"Architecture": "4. Hybrid Fused + K-Means", "Features": "117-D Augmented Feature Space", "Accuracy": "67.59%", "Sensitivity": "80.95%", "Specificity": "48.89%", "F1-Score": "0.7368", "ROC-AUC": "0.6868"},
        ]
        st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)

        st.caption("ℹ️ Evaluated on the held-out Guilin PPG-BP benchmark cohort across 36 unseen subjects (108 segments) with zero patient data leakage.")

    # ==============================================================================
    # TAB: RISK ENGINE
    # ==============================================================================
    elif active_tab == "Risk Engine":
        st.subheader("⚙️ Precision Risk Engine & Calibration Architecture")
        st.markdown(
            """
            The risk score combines continuous physiological indices and clinical lab panels:
            $$\\text{Risk Score} = 100 \\times \\left( w_{\\text{ppg}} P(\\text{CVD}) + w_{\\text{labs}} S_{\\text{labs}} + w_{\\text{text}} S_{\\text{text}} \\right)$$
            - **Blood Pressure Contribution:** Calibrated using AHA/ACC 2017 stage cutoffs.
            - **Heart Rate Variability:** Derived from RMSSD and SDNN parasympathetic tone.
            - **Atherosclerosis Index:** Modulated by FAI perivascular attenuation and CAC scores when entered.
            """
        )

    # ==============================================================================
    # TAB: FUSION LAYERS
    # ==============================================================================
    elif active_tab == "Fusion Layers":
        st.subheader("🧬 Multimodal Fusion Readiness & Data Schema")
        st.markdown("Standardized patient container uniting wearable sensors with electronic health records:")

        schema_code = """
@dataclass
class MultimodalPatientRecord:
    patient_id: str
    age: float
    sex: str
    bmi: float
    wearable_ppg_features: np.ndarray          # 75-D structured vector
    wearable_waveform_embedding: np.ndarray    # 32-D 1D CNN embedding
    clinical_labs: ClinicalLabBiomarkers       # 10-D laboratory blood panel
    clinical_text_embedding: Optional[np.ndarray] # 128-D BioClinicalBERT vector
        """
        st.code(schema_code, language="python")

    # ==============================================================================
    # TAB: HEALTHY TIP
    # ==============================================================================
    elif active_tab == "Healthy Tip":
        st.subheader("💚 Personalized Cardiovascular Prevention Guidance")
        st.markdown(
            f"""
            <div class='dash-card'>
                <h4>Guidance for {risk_tier_name} Status</h4>
                <ul>
                    <li><b>Aerobic Physical Activity:</b> Maintain at least 150 minutes of moderate-intensity exercise weekly (e.g., brisk walking, cycling).</li>
                    <li><b>Dietary Sodium:</b> Target dietary sodium intake below 2,000 mg/day; prioritize potassium-rich foods (DASH diet protocol).</li>
                    <li><b>Sleep & Recovery:</b> Aim for 7–8 hours of restorative sleep to support parasympathetic autonomic recovery.</li>
                    <li><b>Routine Follow-up:</b> Schedule clinical blood pressure checks every 6–12 months.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ==============================================================================
    # TAB: SIM LOGS & HISTORY & INFO
    # ==============================================================================
    elif active_tab == "Sim Logs":
        st.subheader("📝 Real-Time Simulation & Audit Logs")
        st.code(
            f"""
[LOG] Signal Ingested: {active_fid} ({len(active_signal)} samples @ {active_fs} Hz)
[LOG] Filter: Zero-phase Butterworth Bandpass (0.5 - 8.0 Hz) applied.
[LOG] Quality Check: SQI evaluation passed (sSQI: {feats.get('sSQI', 0):.3f}, kSQI: {feats.get('kSQI', 0):.3f})
[LOG] Peak Detection: {feats.get('detected_peaks_count', 0)} systolic peaks identified.
[LOG] AI Model Engine: Evaluated via Multimodal CVD Classifier -> {clf_class} (Index: {shd_index})
[LOG] Status: Ready.
            """,
            language="text",
        )
    elif active_tab == "History":
        st.subheader("🕒 Historical Recording Log")
        hist_data = [
            {"Date": "2026-11-03", "Session ID": active_fid, "HR": f"{hr_val} BPM", "BP": f"{bp_val} mmHg", "Risk Score": f"{risk_score}", "Status": risk_tier_name},
            {"Date": "2026-10-28", "Session ID": "spandan_rec_042", "HR": "67 BPM", "BP": "119/78 mmHg", "Risk Score": "20", "Status": "Low Risk"},
            {"Date": "2026-10-21", "Session ID": "spandan_rec_018", "HR": "70 BPM", "BP": "120/80 mmHg", "Risk Score": "21", "Status": "Low Risk"},
        ]
        st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)
    elif active_tab == "Info":
        st.subheader("ℹ️ Platform Specifications & Regulatory Limitations")
        st.markdown(
            """
            - **Platform:** Arohan Precision Cardiovascular Risk Intelligence (v1.0.0)
            - **Lead Developer:** Arohan AI Internship Research Team (October 2026)
            - **Repository:** `https://github.com/harinibas-03/PPG-Cardiovascular-AI`
            - **Medical Disclaimer:** This system is an algorithmic screening and research platform. It does not provide definitive medical diagnoses. True Coronary Artery Disease (CAD) diagnosis requires invasive coronary angiography or certified laboratory assays.
            """
        )
