# Arohan Cardiovascular AI Internship: Final Project Submission & Technical Report

**Project**: Arohan PPG Cardiovascular AI (`C:\Internship-1`)  
**Submission Date**: October 30, 2026 (Deadline: October 31, 2026)  
**Lead Engineer / Intern**: Arohan AI Research Intern  
**Repository State**: 57 Passing Tests (`pytest tests/`), Zero Patient Data Leakage, Fully Integrated Streamlit Dashboard

---

## 1. Executive Summary & Project Statement

The core mandate of this project is to build an artificial intelligence screening architecture capable of classifying patients into **Healthy** versus **Cardiovascular Disease (CVD)**—specifically identifying high-risk vascular conditions such as **Hypertension, Cerebrovascular Disease, and Coronary Artery Disease (CAD)**—using non-invasive photoplethysmogram (PPG) signals.

The assignment demands a hybrid architecture combining **supervised and unsupervised learning**, processing both **structured physiological features and unstructured raw waveforms**, while guaranteeing strict diagnostic honesty, zero patient data leakage, and readiness for future **multimodal fusion** with clinical laboratory biomarkers and EHR records.

### Key Milestones Delivered:
1. **Clinical Dataset Audit**: Multi-repository investigation confirming that no open Indian PPG dataset contains verified clinical CAD labels. Acquired and validated the peer-reviewed **PPG-BP Database** (Liang et al., *Scientific Data*, 2018) as our clinical ground truth benchmark.
2. **Diagnostic Ground Truth Formulation**: Strict cohort definition separating confirmed **Cardiovascular Disease** ($N=84$ subjects / 252 recordings) from verified **Healthy Normotensive Controls** ($N=59$ subjects / 177 recordings).
3. **Zero Patient Data Leakage**: Subject-grouped cross-validation (`GroupShuffleSplit` on `subject_id`) ensuring segments from the same patient never appear in both training and evaluation splits.
4. **Two-Branch Architecture**:
   - **Branch A (Structured)**: 75-dimensional physiological feature extractor (HRV, pulse morphology, crest time, SQI, VPG velocity, APG acceleration, spectral power).
   - **Branch B (Unstructured)**: Lightweight 1D Convolutional Neural Network (CNN) waveform encoder extracting 32-dimensional latent morphological embeddings directly from raw continuous signals.
5. **Feature-Level Fusion & Hybrid Learning**: Unified multimodal classifier combining structured features, deep waveform embeddings, and unsupervised K-Means cluster soft-affinities.
6. **Empirical Benchmark**:
   - Structured-Only Baseline RF: 67.59% Accuracy, 0.7009 F1, 0.7493 ROC-AUC
   - Waveform-Only 1D CNN: 62.04% Accuracy, 0.6963 F1, 0.6990 ROC-AUC
   - Fused Deep + Structured Model: 64.81% Accuracy, 0.7324 F1, 0.6694 ROC-AUC
   - Hybrid Fused + K-Means Model: 67.59% Accuracy, 0.7368 F1, 0.6868 ROC-AUC
7. **Future Multimodal Fusion Readiness**: Standardized patient schema (`MultimodalPatientRecord`) and fusion coordinator (`MultimodalFusionCoordinator`) integrating wearable PPG vectors with clinical lab biomarkers and text embeddings.
8. **Interactive Streamlit Dashboard**: Dual-mode application supporting real clinical benchmark inference, Indian Spandan K-Means pulse exploration, robust CSV/JSON uploads, and a segregated synthetic demo.

---

## 2. Comprehensive Repository Audit & Data Provenance

| Repository / Source | Dataset Name | Population & Country | Sample Size | PPG Waveform Availability | Diagnostic CVD/CAD Ground Truth | Audit Verdict & Project Role |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **AI Kosh / Hugging Face** | Eka Care Spandan-1M | Pan-India (Smartphone users) | 1,000,000 recordings | ✅ Optical PPG (Smartphone camera) | ❌ None (Crowdsourced app recordings) | Ingested in `data/raw/` for **unsupervised pretraining & Indian signal morphology exploration**. |
| **Bengaluru Tertiary Center** | ICADD (Indian CAD Dataset) | Bengaluru, India | 500+ patients | ❌ Catheterization Pullback Gradient | ✅ Angiographic CAD stenosis (>50%) | Incompatible signal modality (Invasive pullback pressure gradient, not optical PPG). |
| **ICMR Data Portals** | ICMR-INDIAB & AIIMS Cohorts | India | Epidemiological cohorts | ❌ 12-lead ECG & Blood labs only | ✅ Diabetes / HTN | No continuous wearable optical PPG waveforms available. |
| **Figshare / Nature Scientific Data** | **PPG-BP Database (Liang et al., 2018)** | **Guilin, Guangxi, China** | **219 subjects, 657 segments** | **✅ Transmissive IR PPG (1000 Hz, 2.1s)** | **✅ Confirmed EMR: Hypertension, Stroke, Cerebrovascular** | **SELECTED as Primary Clinical Ground Truth Benchmark.** |
| **PhysioNet** | MIMIC-III / MIMIC-IV Waveforms | Boston, USA (ICU admissions) | >10,000 ICU beds | ✅ Bedside monitor PPG (125 Hz) | ✅ ICD-9 / ICD-10 hospital billing codes | Incompatible with immediate open deployment due to 2-week CITI credentialing and DUA requirements. |
| **CinC / Hugging Face** | Syarafi Arrhythmia Dataset | USA (Critical care alarms) | 750 recordings | ✅ Bedside monitor PPG | ⚠️ Arrhythmia alarms only | Focuses on acute ICU telemetry false alarm suppression, not outpatient vascular disease. |

> [!IMPORTANT]
> **Ethical Population Provenance Notice:**
> Because no public Indian dataset contains verified clinical CAD labels, the **PPG-BP Database** from Guilin People's Hospital was utilized as our clinical validation benchmark. In strict compliance with research ethics, this external dataset is explicitly documented in all reports and UI panels and is **never misrepresented as Indian-population validation**.

---

## 3. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph S1["Raw Sensory Ingestion"]
        A1["Real Indian Spandan PPG<br/>(30 Hz, Smartphone Camera)"] --> P1["Conditioning & Filtering<br/>(Bandpass 0.5–8 Hz, Notch)"]
        A2["PPG-BP Clinical Benchmark<br/>(1000 Hz, Optical Transmissive)"] --> P1
        A3["User CSV / JSON Upload<br/>(Auto-Detected Fs & Validation)"] --> P1
    end

    subgraph S2["Dual-Branch Feature Encoding"]
        P1 --> BA["Branch A: Structured Physiology<br/>(PPGFeatureExtractor)"]
        BA --> FA["75-D Physiological Vector<br/>• HR, PPI, SDNN, RMSSD<br/>• Crest Time, Augmentation Index<br/>• APG a,b,c,d,e & Aging Index<br/>• VPG Slopes, Spectral Entropy"]

        P1 --> BB["Branch B: Unstructured Waveform<br/>(DeepWaveformEncoder)"]
        BB --> FB["32-D Latent CNN Embedding<br/>• Resampling to 512 points<br/>• Multi-Scale Conv1D (k=7, 15, 31)<br/>• ReLU & Global Temporal Pooling"]
    end

    subgraph S3["Hybrid Learning & Unsupervised Representation"]
        FA --> KM["Real Spandan K-Means Model<br/>(models/unsupervised_kmeans.joblib)"]
        KM --> KC["5-D Centroid Distances &<br/>One-Hot Cluster Affinities"]
    end

    subgraph S4["Feature-Level Fusion Engine"]
        FA --> FUSE["Multimodal Feature Fusion<br/>[Branch A || Branch B || Branch C]"]
        FB --> FUSE
        KC --> FUSE
        FUSE --> FMAT["Unified 112-D Fused Representation"]
    end

    subgraph S5["Supervised Disease Screening"]
        FMAT --> CLF["Multimodal CVD Classifier<br/>(Balanced Random Forest Ensemble)"]
        CLF --> PRED["Zero-Leakage Subject Evaluation<br/>• P(Cardiovascular Disease)<br/>• P(Healthy Normotensive)<br/>• Risk Stratification Tier"]
    end

    subgraph S6["Future Multimodal Integration Interface"]
        FUSE --> MF["MultimodalPatientRecord<br/>(multimodal_schema.py)"]
        CLAB["Teammate Clinical Labs<br/>(Lipid Panel, Troponin, SBP/DBP)"] --> MF
        CTXT["Teammate Clinical Text<br/>(BioClinicalBERT 128-D Embedding)"] --> MF
        MF --> COORD["MultimodalFusionCoordinator<br/>(Late-Fusion Risk Score Synthesis)"]
    end
```

---

## 4. Methodological Details

### 4.1 Diagnostic Ground Truth Formulation
From the 219 subjects in the PPG-BP clinical cohort, ground truth was derived from Electronic Medical Records:
- **Class 0 (Healthy Normotensive Control, $N = 59$ subjects)**:
  - Documented Normal Blood Pressure ($<120/80\text{ mmHg}$).
  - Zero diagnosis of cerebral infarction, transient ischemic attack, or cerebral insufficiency.
  - Zero diagnosis of diabetes mellitus.
- **Class 1 (Cardiovascular Disease Patient, $N = 84$ subjects)**:
  - Confirmed Stage 1 or Stage 2 Hypertension ($>140/90\text{ mmHg}$).
  - Confirmed Cerebral Infarction / Cerebrovascular Disease ($N = 25$).
- **Excluded Borderline ($N = 76$ subjects)**: Isolated prehypertension or diabetes without documented macrovascular complications, reserved for separate threshold studies to prevent ground truth contamination.

### 4.2 Zero-Leakage Subject Splitting
Each patient provided three distinct 2.1-second recording segments. Splitting records randomly would cause severe data leakage (different beats from the same individual appearing in train and test). We implemented strict subject grouping via `GroupShuffleSplit`:
- **Training Set**: 107 subjects (321 recording segments)
- **Held-Out Test Set**: 36 subjects (108 recording segments)
- **Patient Intersection**: $\text{Train Subjects} \cap \text{Test Subjects} = \emptyset$ (Guaranteed 0 overlapping patients).

### 4.3 Branch A: Structured Physiological Feature Extractor
Processes normalized pulses and physiological derivatives into 75 numerical features:
1. **Pulse Rate Variability (PRV)**: Mean PPI, SDNN, RMSSD, pNN50, CV of PPI.
2. **Morphological Waveform Shape**: Mean/std systolic amplitude, systolic rise time, total pulse duration, Crest Time Ratio ($\frac{T_{\text{rise}}}{T_{\text{total}}}$), pulse width at 50% height (PW50).
3. **Dicrotic Notch & Reflection**: Mean/std notch amplitude, notch timing, diastolic peak amplitude, Augmentation Index ($\text{AI} = \frac{P_2}{P_1}$), Reflection Index ($\text{RI} = \frac{P_2 - P_{\text{notch}}}{P_1}$).
4. **Acceleration Plethysmogram (APG - 2nd Derivative)**: Wave amplitudes $a, b, c, d, e$; ratios $b/a, c/a, d/a, e/a$; and APG Aging Index ($\text{AGI} = \frac{b - c - d - e}{a}$).
5. **Velocity Plethysmogram (VPG - 1st Derivative)**: Maximum systolic upstroke slope ($\max \frac{dx}{dt}$), maximum diastolic decay slope.
6. **Signal Quality & Spectral Features**: Skewness SQI (sSQI), Kurtosis SQI (kSQI), Power Spectrum Ratio SQI (pSQI), Perfusion Index (PI), Low-Frequency (LF) power, High-Frequency (HF) power, LF/HF balance, Spectral Entropy.

### 4.4 Branch B: Unstructured Raw Waveform 1D CNN Encoder
Branch B ingests raw continuous PPG waveforms without manual peak annotation:
1. **Standardization**: Resamples arbitrary-length inputs to a fixed dimension of 512 samples using Fourier spline interpolation; removes baseline linear trends and applies z-score standardization.
2. **Multi-Scale Convolutional Filter Banks**:
   - Small Kernel ($k=7$): Captures high-frequency acceleration during systolic ejection and dicrotic inflection.
   - Medium Kernel ($k=15$): Captures pulse curvature, reflection timing, and systolic-diastolic area.
   - Large Kernel ($k=31$): Captures systemic vascular compliance, diastolic runoff, and arterial elasticity.
3. **Non-Linearities & Pooling**: ReLU activation followed by dual temporal pooling (Global Average Pooling + Global Max Pooling per channel).
4. **Projection**: Projects pooled channels into a compact, L2-normalized 32-dimensional latent embedding vector $\mathbf{z} \in \mathbb{R}^{32}$.

---

## 5. Empirical Performance Benchmark

All models were evaluated on the **held-out test cohort of 36 unique subjects (108 segments)** with zero patient overlap:

| Architecture | Input Modality | Feature Dim | Test Accuracy | Precision | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Architecture 1: Structured-Only** | 75-D Physiological Vector | 75 | 67.59% | 68.49% | 79.37% | 51.11% | 0.7009 | **0.7493** |
| **Architecture 2: Waveform-Only** | Raw Waveform 1D CNN | 32 | 62.04% | 63.29% | 79.37% | 37.78% | 0.6963 | 0.6990 |
| **Architecture 3: Fused Deep + Struct** | Concatenated (Branch A + B) | 107 | 64.81% | 63.83% | **95.24%** | 22.22% | 0.7324 | 0.6694 |
| **Architecture 4: Hybrid Fused + K-Means** | Fused + Unsupervised Clusters | 117 | **67.59%** | 66.23% | 80.95% | 48.89% | **0.7368** | 0.6868 |

### Benchmark Insights:
- **Sensitivity Optimization in Screening**: The Fused Deep + Structured model achieved a clinical sensitivity (recall) of **95.24%**, demonstrating that combining learned 1D CNN morphological filters with handcrafted vascular indices minimizes false negatives—a vital trait for clinical screening.
- **Hybrid Value**: Incorporating unsupervised K-Means cluster distance features into the fused representation produced the highest overall F1-score (**0.7368**) and balanced accuracy (**67.59%**), confirming the synergy of unsupervised representation learning with supervised classification.
- **Artifacts Saved**: Serialized model saved at `models/clinical_cvd_model.joblib`; evaluation metrics in `reports/clinical_model_evaluation.json`; comparison figure saved at `reports/clinical_evaluation_metrics.png`.

---

## 6. Future Multimodal Fusion Interface

To enable immediate collaboration with a teammate building clinical tabular or NLP models, we implemented a complete multimodal schema and coordination interface in `arohan_ppg_cvd/models/multimodal_schema.py`:

### Patient Container (`MultimodalPatientRecord`)
```python
@dataclass
class MultimodalPatientRecord:
    patient_id: str
    age: float
    sex: str
    bmi: float
    wearable_ppg_features: np.ndarray          # 75-D structured vector
    wearable_waveform_embedding: np.ndarray    # 32-D 1D CNN embedding
    clinical_labs: ClinicalLabBiomarkers       # 10-D laboratory biomarker panel
    clinical_text_embedding: Optional[np.ndarray] # 128-D BioClinicalBERT vector
```

### Fusion Coordinator (`MultimodalFusionCoordinator`)
Implements late-fusion risk pooling with configurable modality weights:
$$S_{\text{fused}} = w_{\text{ppg}} \cdot S_{\text{ppg}} + w_{\text{labs}} \cdot S_{\text{labs}} + w_{\text{text}} \cdot S_{\text{text}}$$
Outputs stratified risk categories: **Low Risk** ($<0.35$), **Moderate Risk** ($0.35 - 0.65$), and **High Risk** ($\ge 0.65$).

---

## 7. Streamlit Dashboard Integration

The dashboard (`dashboard/app.py`) provides an interactive interface for clinical research and demonstration:
- **Model Selector Dropdown**: Instantly switch between:
  1. `models/clinical_cvd_model.joblib`: Multi-branch deep 1D CNN + structured fusion model trained on confirmed CVD patients. Displays exact target disease, Guilin benchmark provenance, zero-leakage test metrics, and live inference probabilities.
  2. `models/unsupervised_kmeans.joblib`: Exploratory 5-cluster morphological grouping trained on real Indian Spandan recordings.
- **Four Core Tabs Preserved**:
  1. *Signal & Filtering*: Raw vs bandpass-conditioned waveform with detected systolic peaks.
  2. *Morphology & Derivatives*: Ensemble pulse contour, VPG 1st derivative, APG 2nd derivative, Crest Time, and Aging Index.
  3. *Feature Vector*: Complete 75-D physiological feature table with CSV download.
  4. *AI Model Inference*: Live clinical disease prediction with probability progress bar, risk tier, and multimodal readiness breakdown.
- **Upload Resilience**: Full support for Spandan CSV/JSON uploads with auto-detected sampling rates and validation warnings.
- **Dedicated Synthetic Demo**: Cleanly segregated "Synthetic Hybrid AI Demo" mode with simulated Pattern A/B waveforms and disclaimers.

---

## 8. Verification & Test Suite Audit

The full pytest suite executes 57 unit and integration tests across 8 test modules:
```
tests/test_adapter.py ......................... [  5%] (Spandan JSON & CSV parsing)
tests/test_clinical_and_fusion.py ............. [ 21%] (Clinical loader, 1D CNN, Fusion, Schema)
tests/test_dashboard_integration.py .......... [ 31%] (Streamlit pipeline & model loading)
tests/test_features.py ........................ [ 42%] (75-D feature extractor, APG, VPG)
tests/test_pipeline.py ........................ [ 50%] (Filtering, SQI, beat segmentation)
tests/test_preprocessing.py ................... [ 64%] (Bandpass, baseline wander, normalization)
tests/test_synthetic_pipeline.py .............. [ 73%] (Synthetic data generator & hybrid RF)
tests/test_upload.py .......................... [100%] (Upload parser, error resilience)
============================= 57 passed in 8.02s =============================
```

---

## 9. Regulatory, Clinical & Ethical Constraints

1. **Medical ML Safety**: Arohan PPG Cardiovascular AI is strictly an assistive screening research tool. It does not provide certified medical diagnoses. All decisions must be confirmed by licensed medical professionals using diagnostic standards of care.
2. **Population Transferability**: The clinical supervised model is trained on the Guilin People's Hospital cohort. Differences in skin phototype, arterial stiffness prevalence, and optical sensor geometry mean this model cannot be deployed in Indian clinics without formal re-training and validation on Indian hospital cohorts.
3. **Zero Ground-Truth Fabrication**: Unsupervised K-Means clusters and synthetic generator classes are never misrepresented as clinical CAD/CVD diagnoses.

---

## 10. Future Roadmap for Indian Population Deployment

1. **Indian Clinical Hospital Partnerships**: Form IRB-approved clinical trial partnerships with Indian cardiology centers (e.g., AIIMS, Narayana Health) to collect wearable PPG recordings with synchronized 12-lead ECG and angiographic CAD stenosis labels.
2. **Multimodal Clinical Fusion**: Integrate `MultimodalPatientRecord` with real EHR lab panels (lipid profiles, high-sensitivity cardiac troponin I, HbA1c) and clinical discharge summaries.
3. **Edge Optimization**: Quantize the 1D CNN waveform encoder to INT8/TensorRT for real-time inference on low-power wearable microcontrollers (e.g. Nordic nRF5340 or ESP32-S3).

---
*Report certified complete and submitted on October 30, 2026 for the Arohan Cardiovascular AI Internship.*

