"""
Spandan-1M Dataset Adapter
==========================
Adapter for the Indian Spandan-1M-V1.0-raw dataset (from Eka Care / AI Kosh / Hugging Face).

Dataset Schema:
- file_id: Unique recording identifier (string/int)
- bpm: Device-estimated heart rate (float/int)
- ppg: List of dictionaries or nested arrays with optical and motion channels:
       {'r': red, 'g': green, 'b': blue, 't': timestamp_ms, 'x': accel_x, 'y': accel_y, 'z': accel_z}

Safety Notice:
Spandan-1M contains NO confirmed clinical diagnosis labels (CAD, MI, Arrhythmia, etc.).
This adapter extracts raw signals and recording metadata strictly for exploratory
analysis, signal processing, feature engineering, and unsupervised clustering.
DO NOT fabricate clinical disease labels or infer coronary artery disease from 'bpm'.
"""

from typing import Dict, Any, List, Optional, Tuple, Generator, Union
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SpandanAdapter:
    """
    Adapter to parse, stream, and structure records from Spandan-1M-V1.0-raw.
    """

    def __init__(self, primary_channel: str = "g", fallback_sampling_rate: float = 30.0):
        """
        Parameters
        ----------
        primary_channel : str
            PPG optical channel to extract. 'g' (green) is recommended for smartphone
            contact PPG due to higher absorption by oxyhemoglobin and hemoglobin.
            Options: 'g', 'r', 'b', or 'all'.
        fallback_sampling_rate : float
            Sampling rate in Hz used if timestamp array 't' is absent or non-uniform.
            Spandan phone recordings are typically ~30 Hz (camera frame rate).
        """
        if primary_channel not in ("g", "r", "b", "all"):
            raise ValueError(f"Invalid primary_channel '{primary_channel}'. Must be 'g', 'r', 'b', or 'all'.")
        self.primary_channel = primary_channel
        self.fallback_sampling_rate = fallback_sampling_rate

    def parse_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a single Spandan record into structured numpy arrays.

        Parameters
        ----------
        record : dict
            Dictionary with keys 'file_id', 'bpm', and 'ppg'.

        Returns
        -------
        dict with:
            - file_id: str
            - bpm: float
            - signal: np.ndarray (1D optical PPG signal, usually green channel)
            - rgb: np.ndarray shape (N, 3) if available
            - timestamps: np.ndarray (in seconds)
            - sampling_rate: float (estimated or fallback)
            - accel: np.ndarray shape (N, 3) with [x, y, z] motion if present
        """
        file_id = str(record.get("file_id", "unknown"))
        bpm = float(record.get("bpm", np.nan)) if record.get("bpm") is not None else np.nan
        raw_ppg = record.get("ppg", [])

        if isinstance(raw_ppg, str):
            try:
                raw_ppg = json.loads(raw_ppg)
            except Exception as e:
                logger.error(f"Failed to parse string ppg in file_id {file_id}: {e}")
                raw_ppg = []

        if not raw_ppg:
            return {
                "file_id": file_id,
                "bpm": bpm,
                "signal": np.array([], dtype=np.float32),
                "rgb": np.empty((0, 3), dtype=np.float32),
                "timestamps": np.array([], dtype=np.float32),
                "sampling_rate": self.fallback_sampling_rate,
                "accel": np.empty((0, 3), dtype=np.float32),
            }

        # Format 1: List of dicts [{'r':..., 'g':..., 'b':..., 't':..., 'x':..., 'y':..., 'z':...}]
        if isinstance(raw_ppg, list) and len(raw_ppg) > 0 and isinstance(raw_ppg[0], dict):
            df_ppg = pd.DataFrame(raw_ppg)
            channels = {col: df_ppg[col].to_numpy(dtype=np.float32) for col in df_ppg.columns}
        # Format 2: Dict of lists {'r': [...], 'g': [...], ...}
        elif isinstance(raw_ppg, dict):
            channels = {k: np.asarray(v, dtype=np.float32) for k, v in raw_ppg.items()}
        # Format 3: Raw 1D or 2D array
        elif isinstance(raw_ppg, (list, np.ndarray)):
            arr = np.asarray(raw_ppg, dtype=np.float32)
            if arr.ndim == 1:
                channels = {self.primary_channel: arr}
            elif arr.ndim == 2 and arr.shape[1] >= 3:
                channels = {"r": arr[:, 0], "g": arr[:, 1], "b": arr[:, 2]}
            else:
                channels = {self.primary_channel: arr.ravel()}
        else:
            raise ValueError(f"Unsupported PPG format for file_id {file_id}: {type(raw_ppg)}")

        # Extract timestamps and compute effective sampling rate
        timestamps = None
        fs = self.fallback_sampling_rate
        if "t" in channels and len(channels["t"]) > 1:
            raw_t = channels["t"]
            # Detect whether timestamps are in milliseconds or seconds
            dt = np.diff(raw_t)
            positive_dt = dt[dt > 0]
            if len(positive_dt) > 0:
                median_dt = np.median(positive_dt)
                if median_dt > 1.0:  # likely milliseconds
                    timestamps = (raw_t - raw_t[0]) / 1000.0
                    fs = 1000.0 / median_dt
                else:  # likely seconds
                    timestamps = raw_t - raw_t[0]
                    fs = 1.0 / median_dt

        # Ensure fs is reasonable (typically 15 Hz to 250 Hz for optical devices)
        if not (10.0 <= fs <= 500.0):
            fs = self.fallback_sampling_rate

        n_samples = len(channels.get(self.primary_channel, next(iter(channels.values()))))
        if timestamps is None or len(timestamps) != n_samples:
            timestamps = np.arange(n_samples, dtype=np.float32) / fs

        # Optical signal
        if self.primary_channel in channels:
            signal = channels[self.primary_channel]
        elif "g" in channels:
            signal = channels["g"]
        elif "r" in channels:
            signal = channels["r"]
        else:
            first_key = next(iter(channels.keys()))
            signal = channels[first_key]

        # RGB combined if available
        if all(k in channels for k in ("r", "g", "b")):
            rgb = np.column_stack([channels["r"], channels["g"], channels["b"]])
        else:
            rgb = np.column_stack([signal])

        # Accelerometer motion channels
        if all(k in channels for k in ("x", "y", "z")):
            accel = np.column_stack([channels["x"], channels["y"], channels["z"]])
        else:
            accel = np.empty((n_samples, 0), dtype=np.float32)

        return {
            "file_id": file_id,
            "bpm": bpm,
            "signal": signal.astype(np.float32),
            "rgb": rgb.astype(np.float32),
            "timestamps": timestamps.astype(np.float32),
            "sampling_rate": float(fs),
            "accel": accel.astype(np.float32),
        }

    @staticmethod
    def load_stream_subset(
        repo_id: str = "ekacare/spandan-1M-V1.0-raw",
        split: str = "train",
        max_samples: int = 50,
        token: Optional[str] = None,
        config_name: Optional[str] = "batch_0",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a small subset of records from Hugging Face without downloading the 43 GB dataset.

        Parameters
        ----------
        repo_id : str
            Hugging Face repository ID.
        split : str
            Dataset split (usually 'train').
        max_samples : int
            Maximum number of records to stream.
        token : Optional[str]
            Hugging Face auth token if repository requires authentication.
        config_name : Optional[str]
            Dataset configuration name (e.g. 'batch_0').

        Yields
        ------
        dict: Parsed Spandan record dictionary.
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError("Please install `datasets`: py -m pip install datasets")

        logger.info(f"Connecting to Hugging Face dataset '{repo_id}' (config: '{config_name}') in streaming mode...")
        if config_name:
            ds = load_dataset(repo_id, config_name, split=split, streaming=True, token=token)
        else:
            ds = load_dataset(repo_id, split=split, streaming=True, token=token)

        adapter = SpandanAdapter()
        count = 0
        for item in ds:
            parsed = adapter.parse_record(item)
            yield parsed
            count += 1
            if count >= max_samples:
                break
        logger.info(f"Successfully streamed {count} records from {repo_id}.")

    @staticmethod
    def load_from_json(filepath: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load local JSON or JSONL subset of Spandan data.
        """
        adapter = SpandanAdapter()
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            # Check if JSONL
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                data = json.load(f)
                if max_samples:
                    data = data[:max_samples]
                for item in data:
                    records.append(adapter.parse_record(item))
            else:
                for idx, line in enumerate(f):
                    if max_samples and idx >= max_samples:
                        break
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        records.append(adapter.parse_record(item))
        return records
