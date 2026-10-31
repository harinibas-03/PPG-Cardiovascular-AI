"""
Model Persistence and Serialization (Joblib)
============================================
Handles serialization and loading of trained pipelines, feature configurations,
and clinical audit metadata.
"""

from typing import Dict, Any, Optional
import os
import datetime
import joblib
import logging

from .hybrid_pipeline import HybridPPGClassifier

logger = logging.getLogger(__name__)


def save_model_bundle(
    model: HybridPPGClassifier,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save trained model and associated audit metadata to disk.

    Parameters
    ----------
    model : HybridPPGClassifier
        Fitted hybrid model.
    save_path : str
        Target filepath (.joblib or .pkl).
    metadata : Optional[dict]
        Information about dataset provenance, confirmed label source, validation metrics.

    Returns
    -------
    str : Absolute path to saved file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    meta = {
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "is_unsupervised_fitted": model.is_unsupervised_fitted,
        "is_supervised_fitted": model.is_supervised_fitted,
        "feature_names": model.feature_names,
        "augmented_feature_names": model.augmented_feature_names,
        "n_clusters": model.n_clusters,
        "clinical_disclaimer": (
            "Medical ML Safety Notice: Intended strictly for research/screening assistance. "
            "Model predictions require certified clinician validation."
        ),
    }

    if metadata:
        meta.update(metadata)

    bundle = {
        "model": model,
        "metadata": meta,
    }

    joblib.dump(bundle, save_path)
    logger.info(f"Model bundle successfully saved to: {save_path}")
    return os.path.abspath(save_path)


def load_model_bundle(load_path: str) -> HybridPPGClassifier:
    """
    Load saved model bundle from disk.

    Parameters
    ----------
    load_path : str
        Path to saved joblib file.

    Returns
    -------
    HybridPPGClassifier: Loaded model ready for inference.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file not found at: {load_path}")

    bundle = joblib.load(load_path)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        meta = bundle.get("metadata", {})
        logger.info(f"Loaded model bundle saved at {meta.get('saved_at', 'unknown')}")
        return model
    elif isinstance(bundle, HybridPPGClassifier):
        return bundle
    else:
        raise ValueError(f"Unrecognized model bundle format in {load_path}")
