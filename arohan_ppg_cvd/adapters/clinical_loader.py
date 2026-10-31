"""Clinical Dataset Loader for the PPG-BP Database (Liang et al., 2018).

Provenance:
- Institution: Guilin People's Hospital, Guilin, China
- Reference: Liang et al. (2018) Scientific Data (Nature), 5:180076.
- Population: External Hospital Benchmark (NOT Indian population)
- Confirmed Clinical Diagnoses: Normal Controls, Hypertension, Cerebral Infarction, Cerebrovascular Disease
"""

import os
import glob
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DEFAULT_CLINICAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "clinical"
)

def load_clinical_ppg_bp_dataset(
    clinical_dir: Optional[str] = None,
    include_prehypertension: bool = False,
) -> pd.DataFrame:
    """Load and harmonize the Guilin PPG-BP clinical dataset.

    Returns a DataFrame containing:
    - subject_id (int)
    - segment_idx (int)
    - file_name (str)
    - raw_signal (np.ndarray)
    - sampling_rate (float = 1000.0)
    - duration_sec (float = 2.1)
    - age, sex, height, weight, bmi, sbp, dbp, heart_rate
    - diagnosis_hypertension, diagnosis_diabetes, diagnosis_stroke, diagnosis_cerebrovascular
    - is_healthy_control (bool)
    - is_cvd (bool)
    - binary_target (int: 0=Healthy, 1=CVD)
    """
    if clinical_dir is None:
        clinical_dir = DEFAULT_CLINICAL_DIR

    xlsx_path = os.path.join(clinical_dir, "Data File", "PPG-BP dataset.xlsx")
    if not os.path.exists(xlsx_path):
        xlsx_path = os.path.join(clinical_dir, "PPG-BP dataset.xlsx")
        if not os.path.exists(xlsx_path):
            raise FileNotFoundError(f"PPG-BP metadata spreadsheet not found in {clinical_dir}")

    raw_dir = os.path.join(clinical_dir, "Data File", "0_subject")
    if not os.path.exists(raw_dir):
        raw_dir = os.path.join(clinical_dir, "0_subject")
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"Raw PPG files folder not found in {clinical_dir}")

    meta = pd.read_excel(xlsx_path, header=1)
    meta = meta.rename(columns={
        "subject_ID": "subject_id",
        "Sex(M/F)": "sex",
        "Age(year)": "age",
        "Height(cm)": "height_cm",
        "Weight(kg)": "weight_kg",
        "Systolic Blood Pressure(mmHg)": "sbp_mmhg",
        "Diastolic Blood Pressure(mmHg)": "dbp_mmhg",
        "Heart Rate(b/m)": "hr_bpm",
        "BMI(kg/m^2)": "bmi",
        "Hypertension": "diag_hypertension",
        "Diabetes": "diag_diabetes",
        "cerebral infarction": "diag_stroke",
        "cerebrovascular disease": "diag_cerebrovascular",
    })

    records = []
    for _, row in meta.iterrows():
        sid = int(row["subject_id"])
        
        # Determine clinical ground truth
        is_stroke = pd.notna(row.get("diag_stroke"))
        is_cerebro = pd.notna(row.get("diag_cerebrovascular"))
        htn_val = str(row.get("diag_hypertension"))
        is_htn_stage12 = htn_val in ["Stage 1 hypertension", "Stage 2 hypertension"]
        is_htn_normal = htn_val == "Normal"
        is_prehtn = htn_val == "Prehypertension"
        has_diabetes = pd.notna(row.get("diag_diabetes"))

        # Confirmed CVD: Stage 1/2 HTN, stroke, or cerebrovascular disease
        is_cvd = bool(is_htn_stage12 or is_stroke or is_cerebro)

        # Strict Healthy Normotensive Control: Normal BP, no stroke, no cerebrovascular, no diabetes
        is_healthy = bool(is_htn_normal and not is_stroke and not is_cerebro and not has_diabetes)

        if not is_healthy and not is_cvd:
            if not include_prehypertension:
                # Prehypertension or isolated diabetes without confirmed macrovascular diagnosis
                continue

        target = 1 if is_cvd else 0

        # Find segment files
        for seg in [1, 2, 3]:
            fname = f"{sid}_{seg}.txt"
            fpath = os.path.join(raw_dir, fname)
            if not os.path.exists(fpath):
                continue

            with open(fpath, "r", encoding="utf-8") as f:
                sig_text = f.read().strip()
            
            raw_vals = np.array([float(x) for x in sig_text.split() if x], dtype=np.float32)
            if len(raw_vals) < 100:
                continue

            records.append({
                "subject_id": sid,
                "segment_idx": seg,
                "file_name": fname,
                "raw_signal": raw_vals,
                "sampling_rate": 1000.0,
                "duration_sec": len(raw_vals) / 1000.0,
                "sex": row.get("sex"),
                "age": row.get("age"),
                "height_cm": row.get("height_cm"),
                "weight_kg": row.get("weight_kg"),
                "bmi": row.get("bmi"),
                "sbp_mmhg": row.get("sbp_mmhg"),
                "dbp_mmhg": row.get("dbp_mmhg"),
                "hr_bpm": row.get("hr_bpm"),
                "diag_hypertension": htn_val,
                "diag_diabetes": str(row.get("diag_diabetes")),
                "diag_stroke": str(row.get("diag_stroke")),
                "diag_cerebrovascular": str(row.get("diag_cerebrovascular")),
                "is_healthy_control": is_healthy,
                "is_cvd": is_cvd,
                "binary_target": target,
                "target_name": "Cardiovascular Disease" if target == 1 else "Healthy Control",
            })

    df = pd.DataFrame(records)
    return df

def split_clinical_by_subject(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-grouped train/test split guaranteeing ZERO patient data leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["subject_id"]))
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Verification: Assert subject intersection is empty
    train_subjects = set(train_df["subject_id"].unique())
    test_subjects = set(test_df["subject_id"].unique())
    assert len(train_subjects.intersection(test_subjects)) == 0, "Patient leakage detected!"

    return train_df, test_df

