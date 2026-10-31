"""
Arohan_PPG_CVD
==============
Hybrid Supervised + Unsupervised AI Pipeline for Cardiovascular Disease (CVD/CAD)
Screening using Photoplethysmography (PPG) Signals.

Designed for wearable optical sensors and smartphone photoplethysmography.
Complies with medical ML safety standards: strictly distinguishes exploratory
unsupervised clustering from clinically validated supervised classification.
"""

__version__ = "1.0.0"
__author__ = "Arohan Health ML Research Team"

from .inference.predictor import ArohanPredictor
from .features.extractor import PPGFeatureExtractor
from .preprocessing.filters import bandpass_filter, remove_baseline_wander
from .preprocessing.normalization import normalize_signal
from .preprocessing.segmentation import segment_pulse_waves, segment_fixed_windows
from .models.hybrid_pipeline import HybridPPGClassifier

__all__ = [
    "ArohanPredictor",
    "PPGFeatureExtractor",
    "bandpass_filter",
    "remove_baseline_wander",
    "normalize_signal",
    "segment_pulse_waves",
    "segment_fixed_windows",
    "HybridPPGClassifier",
]
