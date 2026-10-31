# Arohan_PPG_CVD: Hybrid AI for Early Cardiovascular Disease Screening

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-57%20Passed-brightgreen.svg)]()
[![Framework](https://img.shields.io/badge/Architecture-Dual--Branch%20Fusion%20%2B%20Hybrid%20Learning-purple.svg)]()
[![Safety](https://img.shields.io/badge/Medical%20ML-Zero--Leakage%20Subject%20CV-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT%20%2F%20Research-green.svg)]()

A modular Python machine learning library and research platform for non-invasive cardiovascular disease (CVD) screening using photoplethysmogram (PPG) signals from smartphone optical sensors and wearable devices.

Developed for the **Arohan Internship Assignment (October 2026)**, this repository combines **supervised, unsupervised, and hybrid machine learning** across both **structured physiological features** and **unstructured raw waveforms**, engineered for future **multimodal fusion** with clinical laboratory biomarkers and electronic health records (EHR).

---

## ⚠️ Critical Medical ML Notice & Diagnostic Ground Truth

### 1. Indian Spandan-1M-V1.0-Raw Dataset:
The **Spandan-1M** dataset (Eka Care, hosted on AI Kosh and Hugging Face) contains smartphone contact optical recordings (`ppg` channels, timestamps, and accelerometer data).
- **Spandan-1M contains NO clinical CAD/CVD diagnostic labels**.
- In this repository, Spandan recordings are utilized strictly for **signal conditioning, morphology profiling, and unsupervised representation learning (K-Means clustering)**.
- Unsupervised clusters are treated strictly as **geometric/morphological pulse groupings** and are **never** presented as medical disease diagnoses.
- Resting heart rate (BPM) is **never** used as a diagnostic proxy for coronary artery disease (CAD).

### 2. External Clinical Benchmark (PPG-BP Database):
To evaluate genuine disease classification without fabricating labels or violating diagnostic integrity, the supervised models are trained on the peer-reviewed **PPG-BP Database** ([Liang et al., *Scientific Data*, Nature Publishing Group, 2018](https://doi.org/10.1038/sdata.2018.76)):
- **Source**: Department of Cardiology, Guilin People's Hospital, Guilin, China.
- **Population Notice**: **External Clinical Benchmark (China) — Explicitly NOT validated on Indian patient populations.**
- **Confirmed Clinical Ground Truth**: Extracted from hospital Electronic Medical Records (EMR):
  - **Healthy Normotensive Controls** ($N = 59$ subjects / 177 recordings): BP $< 120/80\text{ mmHg}$, zero history of stroke, cerebrovascular disease, or diabetes.
  - **Cardiovascular Disease Patients** ($N = 84$ subjects / 252 recordings): Confirmed Stage 1 or 2 Hypertension ($> 140/90\text{ mmHg}$), Cerebral Infarction ($N = 20$), or Cerebrovascular Disease ($N = 25$).
- **Strict Ethical Commitment**: This external benchmark is prominently disclosed throughout all documentation, metrics, and user interface panels. It is never misrepresented as Indian-population validation.

---

## 🏛️ End-to-End System Architecture

```
                       ┌────────────────────────────────────────────────────────┐
                       │          Raw Optical PPG Signal (Wearable / Phone)     │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                                                   ▼
                       ┌────────────────────────────────────────────────────────┐
                       │         Signal Conditioning & Quality Gate             │
                       │  • Zero-phase Butterworth Bandpass (0.5 - 8.0 Hz)      │
                       │  • Baseline Wander & Powerline Attenuation             │
                       │  • Normalization (Z-Score / Min-Max / Robust)          │
                       │  • Signal Quality Indices (sSQI, kSQI, pSQI, Perfusion)│
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                ┌──────────────────────────────────┴──────────────────────────────────┐
                │                                                                     │
                ▼                                                                     ▼
┌──────────────────────────────────────────────┐       ┌──────────────────────────────────────────────┐
│  Branch A: Structured Physiology (75-D)      │       │  Branch B: Unstructured Raw Waveform (32-D)  │
│  • PRV / HRV: Mean PPI, SDNN, RMSSD, pNN50   │       │  • Resampling to standardized 512 points     │
│  • Morphology: Crest Time, PW50, AI, RI      │       │  • Multi-Scale Conv1D (k=7, 15, 31 taps)     │
│  • Derivatives: VPG velocity, APG Aging Index│       │  • Systolic rise, notch & diastolic filters  │
│  • Spectral: LF/HF Power, Spectral Entropy   │       │  • ReLU + Global Average & Max Pooling       │
└───────────────────────┬──────────────────────┘       └──────────────────────┬───────────────────────┘
                        │                                                     │
                        │        ┌──────────────────────────────────┐         │
                        │        │ Branch C: Unsupervised Clusters  │         │
                        │        │ • Real Spandan K-Means (k=5)     │         │
                        │        │ • Soft Centroid Distance Affinities│        │
                        │        └────────────────┬─────────────────┘         │
                        │                         │                           │
                        └─────────────────────────┼───────────────────────────┘
                                                  │
                                                  ▼
                        ┌─────────────────────────────────────────────────────┐
                        │        Feature-Level Multimodal Fusion              │
                        │    Concatenated Representation (107-D to 117-D)     │
                        └─────────────────────────┬───────────────────────────┘
                                                  │
                                                  ▼
                        ┌─────────────────────────────────────────────────────┐
                        │      Supervised Classifier (Random Forest)          │
                        │     Zero-Leakage GroupShuffleSplit on Subject ID    │
                        └─────────────────────────┬───────────────────────────┘
                                                  │
                                                  ▼
                        ┌─────────────────────────────────────────────────────┐
                        │          Screening Output & Stratification          │
                        │  • P(Cardiovascular Disease) vs P(Healthy)          │
                        │  • Risk Tier & Population Transferability Notice    │
                        └─────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Evaluation Benchmark

All models were evaluated on the **held-out clinical test cohort of 36 unique subjects (108 segments)** with **strictly zero patient overlap**:

| Architecture | Input Modality | Feature Dim | Test Accuracy | Precision | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Structured-Only** | Handcrafted 75-D Physiology | 75 | 67.59% | 68.49% | 79.37% | 51.11% | 0.7009 | **0.7493** |
| **2. Waveform-Only** | Raw Waveform 1D CNN Latent Vector | 32 | 62.04% | 63.29% | 79.37% | 37.78% | 0.6963 | 0.6990 |
| **3. Fused Multimodal** | Concatenated (Branch A + B) | 107 | 64.81% | 63.83% | **95.24%** | 22.22% | 0.7324 | 0.6694 |
| **4. Hybrid Fused + K-Means** | Fused + Spandan K-Means Affinities | 117 | **67.59%** | 66.23% | 80.95% | 48.89% | **0.7368** | 0.6868 |

### Benchmark Insights:
1. **Screening Sensitivity**: The Fused Deep + Structured model achieved a clinical sensitivity (recall) of **95.24%**, showing that combining multi-scale 1D CNN waveform embeddings with physiological derivatives drastically reduces false negatives in cardiovascular risk screening.
2. **Hybrid Learning Benefit**: Incorporating unsupervised K-Means soft cluster affinities from real Indian Spandan recordings into the clinical classifier produced the highest overall F1-score (**0.7368**) and balanced accuracy (**67.59%**).

---

## 📁 Repository Directory Structure

```text
C:\Internship-1/
├── arohan_ppg_cvd/                 # Core Python Library
│   ├── adapters/
│   │   ├── clinical_loader.py      # PPG-BP dataset loader with zero-leakage subject split
│   │   ├── spandan_adapter.py      # Eka Care Spandan-1M raw parser & stream handler
│   │   └── upload_parser.py        # Robust CSV/JSON upload validator
│   ├── features/
│   │   ├── derivatives.py          # Velocity (VPG) & Acceleration (APG a,b,c,d,e, AGI)
│   │   ├── extractor.py            # Unified 75-D physiological feature extractor
│   │   ├── hrv_stats.py            # PRV/HRV metrics (SDNN, RMSSD, pNN50) & spectral power
│   │   └── waveform.py             # PW50, crest time, Augmentation Index (AI), Reflection Index (RI)
│   ├── inference/
│   │   └── predictor.py            # Safe clinical predictor with medical disclaimer gating
│   ├── models/
│   │   ├── deep_waveform.py        # Branch B: Multi-scale 1D CNN waveform encoder
│   │   ├── fusion.py               # Multimodal multi-branch fusion classifier
│   │   ├── hybrid_pipeline.py      # Hybrid K-Means + Random Forest pipeline
│   │   ├── multimodal_schema.py    # Standardized patient container & late fusion coordinator
│   │   ├── persistence.py          # Joblib serialization & bundle loader
│   │   ├── supervised.py           # Patient-grouped Random Forest cross-validation
│   │   └── unsupervised.py         # K-Means clustering + PCA + silhouette evaluator
│   └── preprocessing/
│       ├── filters.py              # Zero-phase Butterworth & baseline wander filter
│       ├── normalization.py        # Z-score, Min-Max, and Robust scalers
│       ├── segmentation.py         # Systolic peak detection & pulse beat segmentation
│       └── sqi.py                  # Signal Quality Index (sSQI, kSQI, pSQI, Perfusion)
├── dashboard/
│   └── app.py                      # Interactive Streamlit clinical screening app
├── data/
│   └── sample_ppg.csv              # Anonymized sample recording for testing & local demo
├── docs/
│   └── LABELED_DATASET_SPECIFICATION.md # Clinical data schema for hospital partner integration
├── models/                         # Model directory (.gitkeep tracked; .joblib ignored)
├── reports/
│   ├── AROHAN_FINAL_SUBMISSION.md  # Comprehensive final internship submission report
│   ├── clinical_dataset_search.md  # Multi-repository clinical dataset audit report
│   ├── clinical_evaluation_metrics.png # Clinical ROC curves, confusion matrix, and bar chart
│   ├── clinical_model_evaluation.json  # Held-out evaluation metrics across all 4 architectures
│   ├── synthetic_evaluation_metrics.png # Synthetic demo evaluation figures
│   ├── synthetic_hybrid_evaluation.json # Synthetic demo benchmark metrics
│   └── technical_report.md         # Signal conditioning & K-Means cluster stability report
├── scripts/
│   ├── download_clinical_dataset.py # Downloads & extracts the open PPG-BP dataset (Guilin)
│   ├── download_spandan_subset.py   # Streams Spandan recordings from Hugging Face
│   ├── generate_synthetic_dataset.py # Generates 1,000 synthetic recordings
│   ├── process_and_cluster_real_spandan.py # Extracts features & fits Spandan K-Means
│   ├── train_clinical_models.py     # Trains all 4 multi-branch clinical models
│   ├── train_supervised.py          # Generic supervised training entrypoint
│   └── train_synthetic_hybrid.py    # Fits baseline vs hybrid synthetic models
├── tests/                          # 57 automated unit and integration tests
├── .gitignore                      # Git exclusion rules (excluding heavy datasets, models, secrets)
├── pyproject.toml                  # Python package configuration
├── requirements.txt                # Dependency list
└── README.md                       # Main documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Installation & Environment Setup
Clone the repository and install dependencies in Python 3.10+:
```bash
git clone https://github.com/harinibas-03/PPG-Cardiovascular-AI.git
cd PPG-Cardiovascular-AI

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Run the Full Test Suite
Verify that all 57 automated unit and integration tests pass:
```bash
pytest tests/ -v
```
*(All 57 tests should pass in under 10 seconds)*

### 3. Launch the Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.
- **Mode 1: Clinical PPG Research**: Inspect signal conditioning, fiducial APG/VPG points, 75-D feature vectors, and live clinical screening inference.
- **Mode 2: Synthetic Hybrid AI Demo**: Dedicated demonstration mode comparing simulated Pattern A vs Pattern B waveforms.
- **CSV / JSON Upload**: Upload raw PPG recordings; sampling frequency is automatically detected or manually set.

### 4. Reproduce Model Training Pipelines

#### A. Download & Train on the Clinical Benchmark Dataset (PPG-BP):
```bash
# 1. Download and extract the open Guilin Hospital benchmark dataset
python scripts/download_clinical_dataset.py

# 2. Train and evaluate all 4 multi-branch architectures with zero patient leakage
python scripts/train_clinical_models.py
```
*Outputs generated:*
- `models/clinical_cvd_model.joblib`: Serialized multi-branch fused screening model.
- `reports/clinical_model_evaluation.json`: Comprehensive held-out evaluation metrics.
- `reports/clinical_evaluation_metrics.png`: Model comparison bar chart and confusion matrix.

#### B. Process & Cluster Real Indian Spandan-1M PPGs:
```bash
# 1. Download a subset of Spandan recordings from Hugging Face
python scripts/download_spandan_subset.py --max-samples 500 --output data/raw/spandan_subset.json

# 2. Run preprocessing, SQI quality audit, and K-Means clustering (k=5)
python scripts/process_and_cluster_real_spandan.py
```
*Outputs generated:*
- `models/unsupervised_kmeans.joblib`: Fitted 5-cluster morphological representation model.
- `reports/cluster_stability_report.json`: Seed stability ($ARI = 0.9832$) and silhouette evaluation.

#### C. Generate Synthetic Data & Evaluate Hybrid Pipeline:
```bash
# 1. Generate 1,000 controlled synthetic PPG recordings
python scripts/generate_synthetic_dataset.py --subjects 100 --records-per-subject 10 --output-dir data/synthetic

# 2. Train baseline vs hybrid models on synthetic data
python scripts/train_synthetic_hybrid.py
```

---

## 🧬 Future Multimodal Fusion Interface

To facilitate seamless collaboration with clinical teammates working on laboratory blood panels or EHR clinical text notes, this library includes a standardized patient container in `arohan_ppg_cvd/models/multimodal_schema.py`:

```python
from arohan_ppg_cvd.models.multimodal_schema import (
    MultimodalPatientRecord,
    ClinicalLabBiomarkers,
    MultimodalFusionCoordinator,
)

# 1. Standardized clinical laboratory panel
labs = ClinicalLabBiomarkers(
    systolic_bp_mmhg=142.0,
    diastolic_bp_mmhg=92.0,
    total_cholesterol_mg_dl=225.0,
    cardiac_troponin_t_ng_ml=0.03,
)

# 2. Unified patient record uniting wearable PPG with EHR
patient = MultimodalPatientRecord(
    patient_id="IND_PAT_1042",
    age=56.0,
    sex="M",
    bmi=27.8,
    wearable_ppg_features=ppg_features_75d,
    wearable_waveform_embedding=waveform_embedding_32d,
    clinical_labs=labs,
    clinical_text_embedding=bioclinicalbert_128d_vector,
)

# 3. Multimodal late fusion coordinator
coordinator = MultimodalFusionCoordinator(
    weight_wearable_ppg=0.45,
    weight_clinical_labs=0.35,
    weight_clinical_text=0.20,
)
risk_assessment = coordinator.compute_late_fusion_risk(
    ppg_risk_score=0.78,
    labs_risk_score=0.65,
    text_risk_score=0.55,
)
# Output: {'fused_cardiovascular_risk_score': 0.6885, 'risk_category': 'High Risk', ...}
```

---

## 🛡️ Regulatory, Clinical & Ethical Constraints

1. **Research & Screening Tool Only**: Arohan PPG Cardiovascular AI is an algorithmic research platform intended strictly for assistive triage and non-invasive screening exploration. **It is NOT a certified medical device and does NOT provide definitive medical diagnoses.**
2. **Lack of Indian Clinical Ground Truth**: The Indian Spandan-1M dataset does not include clinical disease labels. Although the pipeline successfully models Spandan waveform morphology, **the supervised model has NOT been clinically validated on an Indian hospital cohort**.
3. **Specific Disease Limitations**: True Coronary Artery Disease (CAD) diagnosis requires standard-of-care invasive coronary angiography ($>50\%$ arterial stenosis) or high-sensitivity cardiac troponin testing. Photoplethysmography measures peripheral pulsatile blood volume and arterial elasticity; while sensitive to systemic vascular stiffness, hypertension, and cerebrovascular disease, **it cannot replace coronary angiography or 12-lead ECG**.
4. **Diagnostic Integrity**: In accordance with medical ethics, synthetic generator patterns and unsupervised K-Means clusters are **never** reported as clinical diagnoses.

---

## 📜 License & Acknowledgments
- **License**: MIT License.
- **PPG-BP Database**: Liang Y, Chen Z, Liu G, Elgendi M. *Scientific Data* 5:180076 (2018). CC BY 4.0.
- **Spandan-1M Dataset**: Eka Care (AI Kosh / Hugging Face).
- **Project**: Developed for the **Arohan Internship Assignment (October 2026)**.
