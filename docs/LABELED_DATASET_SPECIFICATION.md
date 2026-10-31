# Medically Labeled PPG Dataset Specification & Clinical Ingestion Guide
**Project:** Arohan PPG Cardiovascular AI  
**Document Version:** 1.0 (Clinical Integration Specification)  
**Date:** October 28, 2026  
**Status:** Approved Schema for Supervised CAD/CVD Model Training  

---

## 1. Executive Clinical & ML Rationale

The Indian **Spandan-1M-V1.0-raw** dataset is an exploratory optical PPG collection lacking clinical diagnostic outcomes. While it enables unsupervised morphological representation learning (e.g., K-Means pulse profiling), supervised disease prediction requires a **strictly validated, medically confirmed ground-truth dataset**.

> [!IMPORTANT]
> **Zero-Fabrication Policy:**
> Under no circumstances may clinical diagnostic labels (e.g., CAD, MI, CVD) be inferred, derived, or synthesized from heart rate, K-Means cluster IDs, or morphological waveform features. Models claiming clinical efficacy without patient-consented, diagnostically verified labels violate medical ML safety standards.

---

## 2. Medically Verified Target Labels & Gold Standards

To train a clinically defensible supervised classifier, target labels must be anchored to recognized cardiology diagnostic standards:

| Label Code | Clinical Category | Mandatory Diagnostic Gold Standard Verification |
| :---: | :---: | :--- |
| **`0`** | **Healthy / Control** | Confirmed absence of known cardiovascular disease: normal resting 12-lead ECG, no history of myocardial infarction or revascularization, asymptomatic, and normal cardiac biomarkers where available. |
| **`1`** | **Coronary Artery Disease (CAD) / CVD** | Objectively documented coronary artery disease: catheter coronary angiography ($\ge 50\%$ luminal stenosis in $\ge 1$ major epicardial coronary artery), CT coronary angiography, documented prior acute coronary syndrome (ACS/STEMI/NSTEMI), or cardiologist-confirmed ischemic heart disease. |

*Note: Multi-class sub-stratification (e.g., 0 = Healthy, 1 = Stable Angina, 2 = Acute Myocardial Infarction, 3 = Arrhythmia) may be used if sufficient sub-cohort numbers are collected ($N \ge 100$ per class).*

---

## 3. Mandatory Dataset Schema

Ingested datasets may be provided as **CSV** or **Apache Parquet** (`.parquet`). Each row represents one recording session or segment.

| Column Name | Data Type | Requirement | Description & Clinical Guidance |
| :--- | :---: | :---: | :--- |
| **`patient_id`** | `string` / `int` | **MANDATORY** | Unique de-identified subject/patient ID. Used as the grouping key for `GroupKFold` cross-validation to prevent intra-patient data leakage. |
| **`label`** | `int` (`0` or `1`) | **MANDATORY** | Ground truth clinical diagnosis (`0`: Healthy, `1`: Confirmed CAD/CVD). |
| **`signal`** | `list[float]` or `string` | **MANDATORY*** | 1D raw optical photoplethysmogram readings (minimum 10 seconds duration). If string, must be valid JSON array. |
| **`sampling_rate`** | `float` | **MANDATORY** | Device sampling frequency in Hertz (typically $25.0 - 100.0\text{ Hz}$). |
| **`file_id`** | `string` | Recommended | Unique recording/session identifier. |
| **`age`** | `float` / `int` | Optional (Covariate) | Patient age in years. |
| **`gender`** | `string` | Optional (Covariate) | Biological sex (`M`, `F`, `O`). |
| **`clinical_site`** | `string` | Optional (Covariate) | Hospital or clinic site ID for multi-center batch effect auditing. |
| **`acquisition_device`**| `string` | Optional (Covariate) | Hardware model (e.g., "Smartphone Camera Green", "Reflectance Pulse Oximeter"). |

*\*If the dataset is already pre-extracted into tabular features, the file must contain the engineered morphological columns (e.g., `mean_pw50`, `mean_apg_aging_index`, `pSQI`) alongside `patient_id` and `label`.*

---

## 4. Strict Protocol for Preventing Data Leakage

Data leakage is the most frequent flaw in published biomedical ML studies. Pulse signals from the same individual share idiosyncratic vascular signatures (vessel compliance, skin tone, micro-vascular anatomy). 

```
[ CLINICAL COHORT: N Patients ]
             │
             ▼
   GroupKFold Partitioning (by patient_id)
   ┌────────────────────────────────────────┐
   │ Fold 1: Patients P001..P020 (Train)    │ ───► Evaluated ONLY on
   │         Patients P021..P025 (Test/Val) │      Unseen Patients!
   └────────────────────────────────────────┘
```

1. **Patient-Level Separation**: All beats and recordings belonging to a given `patient_id` MUST be partitioned exclusively into either the training fold or the validation fold. They must never be split across both.
2. **Standardization & Scaling Isolation**: Feature scalers (`StandardScaler`) and imputation statistics must be fit strictly on the training fold and applied out-of-fold to validation data.
3. **No Target Leakage in Unsupervised Stage**: In the hybrid pipeline, if unsupervised cluster distances are used as augmented features, the clusterer centroids must either be pre-trained on an independent unlabeled cohort (e.g., Spandan-1M exploratory subset) or fitted strictly on the cross-validation training fold.

---

## 5. Supervised Training Command Line Ingestion

Once a medically labeled dataset complying with this schema is acquired, execute the existing production training pipeline:

```powershell
# Ingest labeled dataset and train supervised hybrid model
.venv\Scripts\python.exe scripts\train_supervised.py `
    --data "data/labeled_clinical_ppg.csv" `
    --target-col "label" `
    --patient-col "patient_id" `
    --n-clusters 4 `
    --n-splits 5 `
    --output "models/arohan_hybrid_cvd_model.joblib" `
    --eval-plot "reports/supervised_roc_pr_curve.png"
```

### Generated Artifacts
- **Model Bundle**: `models/arohan_hybrid_cvd_model.joblib` (contains both unsupervised feature representation learner and calibrated Random Forest classifier).
- **Audit Metrics**: 5-fold cross-validated ROC-AUC, PR-AUC, Sensitivity, Specificity, and F1-score across independent patients.
- **Top Morphological Biomarkers**: Feature importance ranking identifying which pulse wave components (e.g., APG aging index, reflection index) drive the clinical prediction.
