"""
Unified PPG Feature Extractor
=============================
Integrates signal conditioning, SQI validation, peak detection, pulse wave segmentation,
morphological analysis, derivative (APG/VPG) extraction, and statistical/spectral HRV profiling.
"""

from typing import Dict, Any, List, Optional, Union
import logging
import numpy as np
import pandas as pd

from ..preprocessing.filters import clean_ppg_signal
from ..preprocessing.normalization import normalize_signal
from ..preprocessing.segmentation import detect_systolic_peaks, segment_pulse_waves, segment_fixed_windows
from ..preprocessing.sqi import evaluate_signal_quality
from .waveform import extract_waveform_features
from .derivatives import extract_derivative_features
from .hrv_stats import extract_hrv_features, extract_statistical_and_spectral_features

logger = logging.getLogger(__name__)


class PPGFeatureExtractor:
    """
    Extracts high-dimensional physiological and statistical feature vectors from PPG signals.
    """

    def __init__(
        self,
        default_fs: float = 30.0,
        filter_lowcut: float = 0.5,
        filter_highcut: float = 8.0,
        enforce_sqi: bool = False,
        normalization_method: str = "z_score",
    ):
        """
        Parameters
        ----------
        default_fs : float
            Default sampling rate in Hz (e.g. 30 Hz for smartphone optical sensor, 50-100 Hz for wearable).
        filter_lowcut : float
            Butterworth bandpass lower cutoff (default 0.5 Hz).
        filter_highcut : float
            Butterworth bandpass higher cutoff (default 8.0 Hz).
        enforce_sqi : bool
            If True, returns None or empty features when signal fails SQI acceptability thresholds.
        normalization_method : str
            Normalization technique: 'z_score', 'min_max', or 'robust'.
        """
        self.default_fs = default_fs
        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        self.enforce_sqi = enforce_sqi
        self.normalization_method = normalization_method

    def extract_from_signal(
        self,
        raw_signal: Union[np.ndarray, List[float]],
        fs: Optional[float] = None,
        file_id: Optional[str] = None,
        bpm_reference: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single 1D PPG signal and return engineered feature dictionary.

        Parameters
        ----------
        raw_signal : np.ndarray or list
            Raw 1D photoplethysmogram readings.
        fs : Optional[float]
            Sampling frequency (defaults to self.default_fs if None).
        file_id : Optional[str]
            Optional tracking ID for recording/patient.
        bpm_reference : Optional[float]
            Optional hardware reference heart rate from device.

        Returns
        -------
        dict containing extracted features and signal quality metadata, or None if SQI fails.
        """
        sampling_rate = float(fs if fs is not None else self.default_fs)
        sig = np.asarray(raw_signal, dtype=np.float32).ravel()

        if sig.size < int(sampling_rate * 2.0):
            logger.warning("Signal too short for feature extraction (less than 2 seconds).")
            return None

        # 1. Conditioning & Filtering
        filtered_sig = clean_ppg_signal(
            sig,
            fs=sampling_rate,
            lowcut=self.filter_lowcut,
            highcut=self.filter_highcut,
        )

        # 2. Normalization
        norm_sig = normalize_signal(filtered_sig, method=self.normalization_method)

        # 3. Signal Quality Index (SQI)
        is_acceptable, sqi_metrics = evaluate_signal_quality(norm_sig, fs=sampling_rate)
        if self.enforce_sqi and not is_acceptable:
            logger.warning(f"Signal rejected due to poor SQI: {sqi_metrics}")
            return None

        # 4. Systolic Peak Detection & HRV
        peaks, _ = detect_systolic_peaks(norm_sig, fs=sampling_rate)
        hrv_features = extract_hrv_features(peaks, fs=sampling_rate)

        # 5. Beat Segmentation & Morphological Analysis
        beats = segment_pulse_waves(norm_sig, fs=sampling_rate)
        waveform_features = extract_waveform_features(beats, fs=sampling_rate)

        # 6. Derivative (APG & VPG) Feature Extraction
        derivative_features = extract_derivative_features(beats, fs=sampling_rate)

        # 7. Statistical & Spectral Features
        stat_spec_features = extract_statistical_and_spectral_features(norm_sig, fs=sampling_rate)

        # Combine into complete feature vector
        features = {}
        if file_id is not None:
            features["file_id"] = str(file_id)

        features["sampling_rate"] = sampling_rate
        features["duration_sec"] = round(len(sig) / sampling_rate, 2)
        features["detected_peaks_count"] = int(len(peaks))
        features["segmented_beats_count"] = int(len(beats))

        if bpm_reference is not None and not np.isnan(bpm_reference):
            features["device_bpm_ref"] = float(bpm_reference)

        # Attach all sub-dictionaries
        features.update(sqi_metrics)
        features.update(hrv_features)
        features.update(waveform_features)
        features.update(derivative_features)
        features.update(stat_spec_features)

        return features

    def extract_batch(
        self,
        records: List[Dict[str, Any]],
        fs: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Process a list of records (e.g. Spandan records or dictionaries) into a DataFrame.

        Parameters
        ----------
        records : list of dict
            Each dict must have 'signal' (or 'ppg') and optionally 'file_id', 'bpm', 'sampling_rate'.
        fs : Optional[float]
            Fallback sampling rate.

        Returns
        -------
        pd.DataFrame: Table where each row is a sample and columns are extracted features.
        """
        rows = []
        for r in records:
            sig = r.get("signal", r.get("ppg"))
            file_id = r.get("file_id")
            bpm = r.get("bpm")
            rec_fs = r.get("sampling_rate", fs)

            feat = self.extract_from_signal(
                raw_signal=sig,
                fs=rec_fs,
                file_id=file_id,
                bpm_reference=bpm,
            )
            if feat is not None:
                rows.append(feat)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Ensure numeric columns are cleanly cast
        meta_cols = {"file_id"}
        for col in df.columns:
            if col not in meta_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df
