# Arohan Internship: Clinical PPG Dataset Search & Audit Report

**Date**: October 30, 2026  
**Project**: Arohan PPG Cardiovascular AI (`C:\Internship-1`)  
**Status**: Comprehensive Multi-Repository Clinical Audit Completed

---

## Executive Summary

The primary mandate of Task 1 is to locate real, clinically labeled photoplethysmogram (PPG) data to transition our cardiovascular AI architecture from synthetic demonstration to genuine clinical disease classification. We conducted a systematic, verified investigation across Indian medical research repositories (AI Kosh, ICMR, Indian institutional databases), international biomedical archives (PhysioNet, Figshare, Zenodo), and open machine learning hubs (Hugging Face, Kaggle).

### Primary Finding on Indian PPG Datasets
While India possesses significant raw PPG and ECG data initiatives, **no publicly accessible Indian-population dataset currently provides confirmed, patient-level cardiovascular disease labels (such as Coronary Artery Disease, Hypertension, or Cerebrovascular Disease) alongside wearable optical PPG waveforms**. 
- The flagship Indian dataset **Spandan-1M** (Eka Care / AI Kosh / Hugging Face) contains 1,000,000 smartphone-captured PPG recordings; however, it lacks verified clinical diagnostic ground truth.
- The **ICADD** (Indian Coronary Artery Disease Dataset, Bengaluru) provides detailed angiographic reports and coronary lesion grades, but records invasive hemodynamics—specifically Pullback Pressure Gradients (also abbreviated "PPG" in interventional cardiology)—rather than wearable optical photoplethysmograms.

### Selected Clinical Benchmark Dataset
To achieve rigorous, zero-leakage clinical validation without falsifying labels or compromising diagnostic integrity, we identified, acquired, and audited the peer-reviewed **PPG-BP Database** (Liang et al., *Scientific Data*, Nature Publishing Group, 2018):
- **Official Name**: PPG-BP Database (Photoplethysmography Blood Pressure & Cardiovascular Disease Dataset)
- **Source Institution**: Guilin People's Hospital, Guilin, China
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Population**: Guilin, Guangxi, China (**External Clinical Benchmark — explicitly NOT Indian population**)
- **Cohort**: 219 unique subjects, 657 raw waveform recordings (3 segments per subject, 2,100 samples at 1,000 Hz)
- **Clinical Diagnoses**: Confirmed via Hospital Electronic Medical Records (EMR):
  - Healthy Normotensive Controls: 59 subjects
  - Cardiovascular Disease (Hypertension Stages 1–2, Cerebral Infarction, Cerebrovascular Disease): 84 subjects
  - Prehypertension / Metabolic Borderline: 76 subjects

Per strict clinical guidelines, this external benchmark is explicitly labeled as such in all code, metrics, and user interfaces. It is never misrepresented as Indian-population validation.

---

## Detailed Repository Audit & Candidate Evaluations

### 1. Indian Repositories & Initiatives

#### A. AI Kosh / Eka Care Spandan-1M
- **Dataset Name**: Spandan-1M Photoplethysmography Dataset
- **URL**: `https://huggingface.co/datasets/eka-care/spandan-1m` / AI Kosh
- **Population**: India (pan-Indian smartphone users)
- **Size**: 1,000,000 PPG recording segments (8 kHz downsampled / normalized)
- **Cardiovascular Labels**: **None**. Recordings are crowdsourced smartphone camera photoplethysmograms collected during app usage. No clinical diagnostic records, angiography, echocardiograms, or EMR diagnoses are attached.
- **Suitability**: Ingested in our existing pipeline (`data/raw/spandan_subset.json`) for **unsupervised representation learning and normative cluster discovery**, but cannot be used for supervised CVD/CAD classification.

#### B. Indian Coronary Artery Disease Dataset (ICADD - Bengaluru)
- **Dataset Name**: ICADD Angiography & Hemodynamic Cohort
- **Institution**: Tertiary cardiology center, Bengaluru, Karnataka, India
- **Size**: 500+ patient records with confirmed coronary artery stenosis (>50% LAD, LCx, RCA)
- **PPG Signal Availability**: **Incompatible**. The clinical documentation notes "PPG", which in invasive cardiac catheterization denotes *Pullback Pressure Gradient* (fractional flow reserve pullback). It does not contain non-invasive optical photoplethysmograms.

#### C. ICMR (Indian Council of Medical Research) Data Portals
- **Review**: Screened ICMR National Data Repository and public health study archives.
- **Finding**: Epidemiological survey data, metabolic blood panels (ICMR-INDIAB), and 12-lead ECGs (e.g. AIIMS Delhi cohorts) exist; however, continuous wearable PPG waveform repositories with linked CAD angiographic ground truth are not publicly accessible for open model development.

---

### 2. International Biomedical Repositories

#### A. PPG-BP Database (Liang et al., 2018) — **SELECTED**
- **Official Title**: Data from: A photoplethysmography database with related cardiovascular diseases and blood pressure
- **Publication**: *Scientific Data* 5, Article number: 180076 (2018), Nature Publishing Group.
- **Digital Object Identifier**: DOI: 10.1038/sdata.2018.76 / Figshare: 10.6084/m9.figshare.5459299
- **Repository**: Figshare & GitHub (`https://github.com/sanqiaiziji/PPG_BP_dataset`)
- **Institution**: Department of Cardiology & Medical Records, Guilin People's Hospital
- **Population**: Guilin, China (East Asian adult cohort, ages 21–86, 54% male, 46% female)
- **Sensory Protocol**:
  - Device: Micro-sensor transmissive fingertip photoplethysmograph
  - Sampling Frequency: 1,000 Hz
  - Duration: 2.1 seconds per segment (2,100 discrete amplitude readings)
  - Segments: 3 consecutive, motion-artifact-screened segments per subject
- **Ground Truth Disease Labels (EMR Validated)**:
  - Hypertension: Normal (80), Prehypertension (85), Stage 1 Hypertension (34), Stage 2 Hypertension (20)
  - Cerebral Infarction: 20 subjects
  - Cerebrovascular Disease / Insufficiency of Cerebral Blood Supply: 25 subjects
  - Type 2 Diabetes Mellitus: 38 subjects
- **Cardiovascular Target Formulation**:
  - `Class 0 (Healthy Control)`: Strict normotensive (BP < 120/80 mmHg), zero history of stroke, cerebral infarction, or vascular insufficiency ($N = 59$ subjects).
  - `Class 1 (CVD Patient)`: Clinically confirmed vascular pathology: Stage 1/2 Hypertension, Cerebral Infarction, or Cerebrovascular Disease ($N = 84$ subjects).
- **Access Status**: Open Access CC BY 4.0; fully downloaded and ingested into `data/clinical/`.

#### B. MIMIC-III / MIMIC-IV Waveform Database (PhysioNet)
- **URL**: `https://physionet.org/content/mimic3wdb/`
- **Institution**: Beth Israel Deaconess Medical Center, Boston, USA
- **Size**: >10,000 ICU admissions with fingertip PPG waveforms (sampled at 125 Hz)
- **Disease Labels**: ICD-9/ICD-10 codes including CAD (414.01), myocardial infarction (410.x), heart failure (428.0).
- **Access Constraints**: Requires credentialed CITI "Data or Specimens Only Research" training, manual Data Use Agreement (DUA) review, and credentialed identity verification taking 1–2 weeks. Not accessible for immediate open reproducible execution.

#### C. BIDMC PPG & Respiration Dataset (PhysioNet)
- **URL**: `https://physionet.org/content/bidmc/1.0.0/`
- **Size**: 53 patients, 8 minutes each, 125 Hz.
- **Disease Labels**: Critical care monitoring data annotated for respiratory rate and heart rate; no longitudinal or stable CAD/CVD diagnostic labels.

#### D. Syarafi PPG Arrhythmia Dataset (Hugging Face / CinC 2015)
- **URL**: `https://huggingface.co/datasets/Syarafi/PPG-Arrhythmia-Dataset`
- **Size**: Bedside monitor PPG segments from PhysioNet Computing in Cardiology Challenge 2015.
- **Labels**: False alarm suppression (Asystole, Bradycardia, Tachycardia, Ventricular Fibrillation). Focuses on acute rhythm alarms in ICU, not outpatient/wearable vascular disease screening.

---

## Dataset Selection & Compliance Matrix

| Candidate Dataset | Origin | Labeled CVD/CAD? | Raw PPG Waveforms? | Accessibility / Terms | Selected Role |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Spandan-1M** | India | ❌ (Unlabeled) | ✅ (1M recordings) | Open (Hugging Face) | Unsupervised Pretraining & Real Indian Signal Profiling |
| **ICADD** | India | ✅ (CAD Angiography) | ❌ (Invasive Pullback) | Institutional Request | Incompatible sensor modality |
| **PPG-BP Database** | China | ✅ (CVD/HTN/Stroke) | ✅ (657 raw segments) | Open (CC BY 4.0) | **Primary Clinical Benchmark Dataset** |
| **MIMIC-III Waveform** | USA | ✅ (ICD-9 CAD/CVD) | ✅ (Bedside ICU) | Restrictive DUA / CITI | Comparative reference only |
| **CinC Arrhythmia** | USA | ⚠️ (Arrhythmia Alarms) | ✅ (Bedside ICU) | Open | Incompatible target (Rhythm vs Vascular) |

---

## Summary of Clinical Ground Truth Formulation

For training and validating our models in Task 2 and Task 3:
1. **Clinical Target**: Binary Cardiovascular Disease Detection (`0` = Healthy Normotensive Control, `1` = Confirmed Cardiovascular Disease: Stage 1/2 Hypertension or Cerebrovascular Disease / Cerebral Infarction).
2. **Subject Splitting**: Zero patient overlap. All splits (`GroupShuffleSplit` on `subject_id`) ensure that segments from the same individual never appear in both training and test sets.
3. **Transparency Commitment**: All evaluation summaries and Streamlit interfaces prominently display that this model is trained on the **Guilin People's Hospital PPG-BP Clinical Benchmark**, demonstrating algorithmic transferability, while noting that real-world deployment in India requires clinical re-validation on an Indian hospital cohort.

