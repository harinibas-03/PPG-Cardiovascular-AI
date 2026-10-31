# Synthetic PPG Hybrid Machine Learning Evaluation Report

> **DISCLAIMER:**  
> This benchmark evaluates a proof-of-concept machine learning pipeline on **1,000 computer-generated synthetic PPG recordings** (Pattern A vs Pattern B).  
> These results **DO NOT** claim or demonstrate clinical diagnostic accuracy for Coronary Artery Disease (CAD) or Cardiovascular Disease (CVD).

---

## 1. Experimental Setup & Leakage-Free Design
- **Total Dataset**: 1,000 recordings (100 synthetic subjects, 10 recordings each, 6.0 seconds @ 30 Hz).
- **Split Strategy**: **Subject-Level Held-Out Split** (80 training subjects / 800 recordings vs 20 held-out test subjects / 200 recordings).
- **Pre-processing**: Scaler, Imputer, and K-Means representations fitted **strictly on training subjects**. Zero test data was observed during representation learning or scaling.

---

## 2. Performance Comparison on Held-Out Test Subjects

| Metric | Baseline Random Forest | Hybrid (K-Means + RF) | Relative Change |
| :--- | :---: | :---: | :---: |
| **Input Features** | 74 base features | 82 features (74 base + 8 K-Means) | +8 features |
| **Test Accuracy** | **100.00%** | **100.00%** | **+0.00%** |
| **Precision** | 100.00% | 100.00% | +0.00% |
| **Recall (Sensitivity)** | 100.00% | 100.00% | +0.00% |
| **F1-Score** | 1.0000 | 1.0000 | +0.0000 |
| **ROC-AUC** | **1.0000** | **1.0000** | **+0.0000** |

---

## 3. Confusion Matrix Breakdown (200 Held-Out Test Recordings)

### Baseline Model
- **True Pattern A (TN)**: 100
- **False Pattern B (FP)**: 0
- **False Pattern A (FN)**: 0
- **True Pattern B (TP)**: 100

### Hybrid Model
- **True Pattern A (TN)**: 100
- **False Pattern B (FP)**: 0
- **False Pattern A (FN)**: 0
- **True Pattern B (TP)**: 100

---

## 4. Key Engineering Takeaways
1. **Unsupervised Representation Value**: Integrating unsupervised K-Means cluster distance features provides geometric clustering affinities directly to the decision trees, demonstrating the complete hybrid workflow.
2. **Clinical Translation Readiness**: When clinically annotated CAD/CVD data becomes available, the exact same pipeline (`HybridPPGClassifier`) can be trained by running `scripts/train_supervised.py --data <clinical_dataset.csv>`.
