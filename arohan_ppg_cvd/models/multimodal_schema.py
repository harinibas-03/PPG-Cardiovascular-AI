"""Patient-Level Multimodal Fusion Schema & Integration Interface.

Designed for future multimodal fusion between the Arohan Wearable PPG AI model
and a clinical-data teammate's model (electronic medical records, lab biomarkers,
and unstructured clinical note embeddings).
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
import json
import numpy as np

@dataclass
class ClinicalLabBiomarkers:
    """Standardized tabular laboratory blood panel and vital signs."""
    systolic_bp_mmhg: Optional[float] = None
    diastolic_bp_mmhg: Optional[float] = None
    total_cholesterol_mg_dl: Optional[float] = None
    ldl_cholesterol_mg_dl: Optional[float] = None
    hdl_cholesterol_mg_dl: Optional[float] = None
    triglycerides_mg_dl: Optional[float] = None
    fasting_glucose_mg_dl: Optional[float] = None
    hba1c_pct: Optional[float] = None
    cardiac_troponin_t_ng_ml: Optional[float] = None
    crp_mg_l: Optional[float] = None

    def to_vector(self) -> np.ndarray:
        """Convert known lab biomarkers to standardized 10-D numeric array."""
        defaults = {
            "systolic_bp_mmhg": 120.0,
            "diastolic_bp_mmhg": 80.0,
            "total_cholesterol_mg_dl": 180.0,
            "ldl_cholesterol_mg_dl": 100.0,
            "hdl_cholesterol_mg_dl": 50.0,
            "triglycerides_mg_dl": 150.0,
            "fasting_glucose_mg_dl": 95.0,
            "hba1c_pct": 5.5,
            "cardiac_troponin_t_ng_ml": 0.01,
            "crp_mg_l": 1.0,
        }
        vals = []
        for k, def_v in defaults.items():
            v = getattr(self, k)
            vals.append(float(v if v is not None else def_v))
        return np.array(vals, dtype=np.float32)

@dataclass
class MultimodalPatientRecord:
    """Unified patient container integrating wearable sensor streams, clinical labs,

    and unstructured clinical text embeddings.
    """
    patient_id: str
    age: float
    sex: str  # 'M', 'F', or 'Other'
    bmi: float
    wearable_ppg_features: np.ndarray  # 74/75-D physiological feature vector
    wearable_waveform_embedding: np.ndarray  # 32-D latent 1D CNN embedding
    clinical_labs: ClinicalLabBiomarkers = field(default_factory=ClinicalLabBiomarkers)
    clinical_text_embedding: Optional[np.ndarray] = None  # e.g. 128-D BioClinicalBERT vector
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a serializable dictionary."""
        return {
            "patient_id": self.patient_id,
            "demographics": {
                "age": self.age,
                "sex": self.sex,
                "bmi": self.bmi,
            },
            "wearable_ppg_features_dim": len(self.wearable_ppg_features),
            "wearable_waveform_embedding_dim": len(self.wearable_waveform_embedding),
            "clinical_labs": asdict(self.clinical_labs),
            "has_clinical_text_embedding": self.clinical_text_embedding is not None,
            "metadata": self.metadata,
        }

    def get_fused_feature_vector(self) -> np.ndarray:
        """Concatenate all available modalities into a single flat vector."""
        sex_binary = 1.0 if str(self.sex).strip().upper() == "M" else 0.0
        demo_vec = np.array([self.age, sex_binary, self.bmi], dtype=np.float32)
        lab_vec = self.clinical_labs.to_vector()
        ppg_feat_vec = np.asarray(self.wearable_ppg_features, dtype=np.float32).flatten()
        ppg_emb_vec = np.asarray(self.wearable_waveform_embedding, dtype=np.float32).flatten()

        parts = [demo_vec, lab_vec, ppg_feat_vec, ppg_emb_vec]

        if self.clinical_text_embedding is not None:
            text_emb_vec = np.asarray(self.clinical_text_embedding, dtype=np.float32).flatten()
            parts.append(text_emb_vec)

        return np.concatenate(parts).astype(np.float32)

class MultimodalFusionCoordinator:
    """Coordinator for multimodal inference and risk synthesis."""

    def __init__(
        self,
        weight_wearable_ppg: float = 0.45,
        weight_clinical_labs: float = 0.35,
        weight_clinical_text: float = 0.20,
    ):
        total = weight_wearable_ppg + weight_clinical_labs + weight_clinical_text
        self.w_ppg = weight_wearable_ppg / total
        self.w_labs = weight_clinical_labs / total
        self.w_text = weight_clinical_text / total

    def compute_late_fusion_risk(
        self,
        ppg_risk_score: float,
        labs_risk_score: float,
        text_risk_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Combine unimodal risk estimates using late fusion / Bayesian belief pooling."""
        if text_risk_score is None:
            norm_w_ppg = self.w_ppg / (self.w_ppg + self.w_labs)
            norm_w_labs = self.w_labs / (self.w_ppg + self.w_labs)
            fused_score = norm_w_ppg * ppg_risk_score + norm_w_labs * labs_risk_score
        else:
            fused_score = (
                self.w_ppg * ppg_risk_score
                + self.w_labs * labs_risk_score
                + self.w_text * text_risk_score
            )

        fused_score = float(np.clip(fused_score, 0.0, 1.0))
        risk_category = "High Risk" if fused_score >= 0.65 else ("Moderate Risk" if fused_score >= 0.35 else "Low Risk")

        return {
            "fused_cardiovascular_risk_score": round(fused_score, 4),
            "risk_category": risk_category,
            "modality_contributions": {
                "wearable_ppg": round(ppg_risk_score, 4),
                "clinical_labs": round(labs_risk_score, 4),
                "clinical_text": round(text_risk_score, 4) if text_risk_score is not None else None,
            },
            "fusion_weights": {
                "wearable_ppg": round(self.w_ppg, 2),
                "clinical_labs": round(self.w_labs, 2),
                "clinical_text": round(self.w_text, 2),
            },
        }

