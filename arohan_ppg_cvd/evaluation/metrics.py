"""
Model Evaluation and Clinical Performance Metrics
=================================================
Calculates standard medical ML classification metrics:
- Sensitivity / Recall (true positive rate)
- Specificity (true negative rate)
- Precision (positive predictive value)
- F1-Score & Balanced Accuracy
- ROC-AUC and Precision-Recall AUC (PR-AUC)
- Confusion Matrix visualization
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive screening metrics.

    Parameters
    ----------
    y_true : 1D array of ground truth binary labels (0: Healthy, 1: CVD/CAD)
    y_pred : 1D array of predicted binary labels
    y_prob : Optional 1D array of predicted probabilities for positive class (CVD)

    Returns
    -------
    dict: Metric values formatted with clinical terminology.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    cm = confusion_matrix(y_t, y_p)
    # Binary confusion matrix unpacking: tn, fp, fn, tp
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0  # Precision
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0  # Negative predictive value
    else:
        tn = fp = fn = tp = 0
        sensitivity = recall_score(y_t, y_p, zero_division=0)
        specificity = 0.0
        ppv = precision_score(y_t, y_p, zero_division=0)
        npv = 0.0

    metrics = {
        "accuracy": round(float(accuracy_score(y_t, y_p)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_t, y_p)), 4),
        "sensitivity_recall": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision_ppv": round(ppv, 4),
        "negative_predictive_value_npv": round(npv, 4),
        "f1_score": round(float(f1_score(y_t, y_p, zero_division=0)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=np.float32)
        if len(np.unique(y_t)) > 1:
            metrics["roc_auc"] = round(float(roc_auc_score(y_t, y_prob_arr)), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_t, y_prob_arr)), 4)
            metrics["brier_score"] = round(float(brier_score_loss(y_t, y_prob_arr)), 4)
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    return metrics


def plot_evaluation_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate ROC Curve and Precision-Recall Curve side-by-side.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_prob, dtype=np.float32)

    fpr, tpr, _ = roc_curve(y_t, y_p)
    roc_auc = roc_auc_score(y_t, y_p)

    precision, recall, _ = precision_recall_curve(y_t, y_p)
    pr_auc = average_precision_score(y_t, y_p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax1.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title("Receiver Operating Characteristic (ROC)")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # PR Curve
    ax2.plot(recall, precision, color="#059669", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall (Sensitivity)")
    ax2.set_ylabel("Precision (PPV)")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)

    return fig
