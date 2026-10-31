"""
PPG File Upload Parser and Validator
====================================
Robust ingestion, format parsing, and clinical signal validation for user-uploaded
photoplethysmogram (PPG) recordings in CSV and JSON formats.

Supports:
- Indian Spandan-1M raw recording format (nested optical RGB & motion accelerometer dicts).
- Preprocessed Spandan JSON subsets (with 'signal', 'sampling_rate', 'file_id', 'bpm').
- Time-series CSV files (single or multi-column, with optical PPG, green channel, or plethysmogram).
- Auto-detection of sampling frequency (fs) from metadata or timestamp differences.
- Strict input validation: duration, missing values, flatlining, and sampling rate checks.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
import io
import json
import logging
import numpy as np
import pandas as pd

from .spandan_adapter import SpandanAdapter

logger = logging.getLogger(__name__)


class UploadValidationError(ValueError):
    """Raised when an uploaded file fails clinical or structural validation."""
    pass


def parse_and_validate_upload(
    file_bytes_or_buffer: Union[bytes, io.BytesIO, io.StringIO, str],
    filename: str,
    manual_fs: float = 30.0,
    primary_channel: str = "g",
) -> Dict[str, Any]:
    """
    Parse and validate an uploaded CSV or JSON PPG file.

    Parameters
    ----------
    file_bytes_or_buffer : bytes, BytesIO, StringIO, or str
        The raw uploaded file content.
    filename : str
        Original filename of the upload (used to determine extension and file ID).
    manual_fs : float
        Fallback sampling rate (in Hz) entered by the user if not found in metadata.
    primary_channel : str
        Default optical channel to extract when multi-channel optical data is present ('g', 'r', 'b').

    Returns
    -------
    dict with:
        - 'signal': np.ndarray (1D float32 amplitude array)
        - 'sampling_rate': float (effective sampling frequency in Hz)
        - 'fs_auto_detected': bool (True if extracted from file metadata or timestamps)
        - 'file_id': str (recording identifier)
        - 'device_bpm': Optional[float] (hardware reference heart rate if present)
        - 'duration_sec': float (signal length in seconds)
        - 'n_samples': int (number of valid samples)
        - 'channel_used': str (optical channel identifier)
        - 'warnings': List[str] (non-fatal advisory notices)

    Raises
    ------
    UploadValidationError
        If the file is malformed, unreadable, empty, flatlined, or physiologically invalid.
    """
    warnings: List[str] = []
    fname_lower = filename.lower().strip()

    if not fname_lower.endswith((".csv", ".json", ".txt")):
        raise UploadValidationError(
            f"Unsupported file format '{filename}'. Only .csv and .json files are supported."
        )

    # 1. Read bytes/string content
    content_bytes: bytes
    if isinstance(file_bytes_or_buffer, bytes):
        content_bytes = file_bytes_or_buffer
    elif hasattr(file_bytes_or_buffer, "read"):
        read_val = file_bytes_or_buffer.read()
        content_bytes = read_val if isinstance(read_val, bytes) else read_val.encode("utf-8")
        if hasattr(file_bytes_or_buffer, "seek"):
            file_bytes_or_buffer.seek(0)
    elif isinstance(file_bytes_or_buffer, str):
        content_bytes = file_bytes_or_buffer.encode("utf-8")
    else:
        raise UploadValidationError("Invalid upload payload type: unable to read file stream.")

    if len(content_bytes) == 0:
        raise UploadValidationError("Uploaded file is empty (0 bytes).")

    # 2. Format-Specific Parsing
    raw_signal: Optional[np.ndarray] = None
    detected_fs: Optional[float] = None
    file_id = filename
    device_bpm: Optional[float] = None
    channel_used = primary_channel

    if fname_lower.endswith(".csv") or fname_lower.endswith(".txt"):
        sig, fs_found, fid_found, bpm_found, ch_found = _parse_csv(
            content_bytes, filename, primary_channel=primary_channel
        )
        raw_signal = sig
        detected_fs = fs_found
        if fid_found:
            file_id = fid_found
        if bpm_found is not None:
            device_bpm = bpm_found
        if ch_found:
            channel_used = ch_found

    elif fname_lower.endswith(".json"):
        sig, fs_found, fid_found, bpm_found, ch_found = _parse_json(
            content_bytes, filename, primary_channel=primary_channel, fallback_fs=manual_fs
        )
        raw_signal = sig
        detected_fs = fs_found
        if fid_found:
            file_id = fid_found
        if bpm_found is not None:
            device_bpm = bpm_found
        if ch_found:
            channel_used = ch_found

    # 3. Determine Effective Sampling Rate
    fs_auto_detected = False
    if detected_fs is not None and 5.0 <= detected_fs <= 1000.0:
        effective_fs = float(detected_fs)
        fs_auto_detected = True
    else:
        if manual_fs is None or manual_fs <= 0:
            raise UploadValidationError(
                f"Sampling frequency must be positive. Received invalid manual_fs: {manual_fs}"
            )
        effective_fs = float(manual_fs)

    # 4. Strict Signal Validation
    if raw_signal is None or len(raw_signal) == 0:
        raise UploadValidationError("No optical PPG signal could be extracted from the uploaded file.")

    raw_signal = np.asarray(raw_signal, dtype=np.float32).ravel()

    # NaN / Inf Handling
    nan_mask = np.isnan(raw_signal) | np.isinf(raw_signal)
    nan_count = int(np.sum(nan_mask))
    total_len = len(raw_signal)

    if nan_count == total_len:
        raise UploadValidationError("The uploaded signal contains only NaN or Inf values.")

    if nan_count > 0:
        nan_pct = (nan_count / total_len) * 100.0
        if nan_pct > 10.0:
            raise UploadValidationError(
                f"Signal contains excessive missing/corrupted data ({nan_pct:.1f}% NaN/Inf). "
                "Maximum allowable missing rate is 10%."
            )
        warnings.append(f"Interpolated {nan_count} ({nan_pct:.1f}%) missing NaN/Inf samples.")
        # Linearly interpolate missing points
        x = np.arange(total_len)
        valid = ~nan_mask
        raw_signal[nan_mask] = np.interp(x[nan_mask], x[valid], raw_signal[valid])

    # Duration Check
    duration_sec = total_len / effective_fs
    if duration_sec < 2.0:
        raise UploadValidationError(
            f"Signal duration is too short: {duration_sec:.2f} seconds ({total_len} samples at {effective_fs:.1f} Hz). "
            "Minimum required duration is 2.0 seconds to extract reliable cardiac cycles."
        )

    # Amplitude Range & Flatline Check
    ptp_val = float(np.ptp(raw_signal))
    std_val = float(np.std(raw_signal))
    if ptp_val < 1e-4 or std_val < 1e-5:
        raise UploadValidationError(
            "Uploaded signal appears flatlined or has near-zero amplitude variance "
            f"(Peak-to-Peak = {ptp_val:.2e}, Std = {std_val:.2e})."
        )

    # Sampling Rate Range Validation
    if not (10.0 <= effective_fs <= 500.0):
        warnings.append(
            f"Sampling frequency ({effective_fs:.1f} Hz) is outside the typical physiological PPG range (10–500 Hz)."
        )

    return {
        "signal": raw_signal,
        "sampling_rate": round(effective_fs, 3),
        "fs_auto_detected": fs_auto_detected,
        "file_id": str(file_id),
        "device_bpm": device_bpm,
        "duration_sec": round(duration_sec, 2),
        "n_samples": total_len,
        "channel_used": channel_used,
        "warnings": warnings,
    }


def _parse_csv(
    content_bytes: bytes,
    filename: str,
    primary_channel: str = "g",
) -> Tuple[np.ndarray, Optional[float], Optional[str], Optional[float], str]:
    """Parse a CSV byte payload into signal and metadata."""
    try:
        # Detect delimiter and read DataFrame
        text = content_bytes.decode("utf-8", errors="replace")
        sample_line = text.splitlines()[0] if text.splitlines() else ""
        delimiter = ";" if ";" in sample_line and "," not in sample_line else (
            "\t" if "\t" in sample_line and "," not in sample_line else ","
        )
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except Exception as e:
        raise UploadValidationError(f"Could not parse CSV file: {e}")

    if df.empty:
        raise UploadValidationError("Uploaded CSV file is empty.")

    # Check for metadata columns
    fs_found: Optional[float] = None
    file_id_found: Optional[str] = None
    bpm_found: Optional[float] = None

    for col in df.columns:
        c_clean = str(col).strip().lower()
        if c_clean in ("sampling_rate", "sampling_frequency", "fs", "frequency_hz"):
            try:
                fs_val = float(df[col].dropna().iloc[0])
                if 5.0 <= fs_val <= 1000.0:
                    fs_found = fs_val
            except Exception:
                pass
        elif c_clean in ("file_id", "recording_id", "patient_id", "id"):
            try:
                file_id_found = str(df[col].dropna().iloc[0])
            except Exception:
                pass
        elif c_clean in ("bpm", "heart_rate", "device_bpm", "hr"):
            try:
                bpm_val = float(df[col].dropna().iloc[0])
                if 30.0 <= bpm_val <= 240.0:
                    bpm_found = bpm_val
            except Exception:
                pass

    # Check for timestamp column to calculate sampling rate if not directly specified
    time_cols = [c for c in df.columns if str(c).strip().lower() in ("t", "time", "timestamp", "timestamps", "time_ms", "time_s")]
    if fs_found is None and time_cols:
        t_col = time_cols[0]
        try:
            t_vals = pd.to_numeric(df[t_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(t_vals) > 2:
                dt = np.diff(t_vals)
                pos_dt = dt[dt > 0]
                if len(pos_dt) > 0:
                    med_dt = float(np.median(pos_dt))
                    if med_dt > 1.0:  # milliseconds
                        computed_fs = 1000.0 / med_dt
                    else:  # seconds
                        computed_fs = 1.0 / med_dt
                    if 5.0 <= computed_fs <= 1000.0:
                        fs_found = computed_fs
        except Exception:
            pass

    # Select signal column
    meta_names = {
        "sampling_rate", "sampling_frequency", "fs", "frequency_hz",
        "file_id", "recording_id", "patient_id", "id",
        "bpm", "heart_rate", "device_bpm", "hr",
        "t", "time", "timestamp", "timestamps", "time_ms", "time_s",
        "accel_x", "accel_y", "accel_z", "x", "y", "z",
    }

    # Preferred optical candidate names in order of priority
    priority_candidates = [
        "signal", primary_channel, "g", "green", "ppg", "pleth", "plethysmogram",
        "optical", "amplitude", "val", "value", "raw",
    ]

    selected_col = None
    for cand in priority_candidates:
        match = [c for c in df.columns if str(c).strip().lower() == cand]
        if match:
            selected_col = match[0]
            break

    # If no priority candidate matched, take the first non-metadata numeric column
    if selected_col is None:
        numeric_cols = [
            c for c in df.columns
            if str(c).strip().lower() not in meta_names and pd.to_numeric(df[c], errors="coerce").notnull().sum() > 0
        ]
        if numeric_cols:
            selected_col = numeric_cols[0]
        else:
            # Fall back to first column
            selected_col = df.columns[0]

    try:
        sig_series = pd.to_numeric(df[selected_col], errors="coerce")
        sig_arr = sig_series.to_numpy(dtype=np.float32)
    except Exception as e:
        raise UploadValidationError(f"Could not convert column '{selected_col}' to numeric PPG amplitude: {e}")

    channel_name = str(selected_col)
    return sig_arr, fs_found, file_id_found, bpm_found, channel_name


def _parse_json(
    content_bytes: bytes,
    filename: str,
    primary_channel: str = "g",
    fallback_fs: float = 30.0,
) -> Tuple[np.ndarray, Optional[float], Optional[str], Optional[float], str]:
    """Parse a JSON byte payload (Spandan raw, subset, or generic array) into signal and metadata."""
    try:
        text = content_bytes.decode("utf-8", errors="replace")
        data = json.loads(text)
    except Exception as e:
        raise UploadValidationError(f"Could not parse JSON file: {e}")

    # Case 1: List of records -> extract first record
    if isinstance(data, list):
        if len(data) == 0:
            raise UploadValidationError("Uploaded JSON list is empty.")
        if isinstance(data[0], (int, float)):
            # Case 1A: Simple raw 1D array of PPG samples [12.0, 14.5, ...]
            arr = np.asarray(data, dtype=np.float32)
            return arr, None, filename, None, "1D array"
        else:
            # List of dictionary records, take record 0
            rec = data[0]
    elif isinstance(data, dict):
        rec = data
    else:
        raise UploadValidationError(f"Unsupported JSON root type '{type(data).__name__}'. Expected object or array.")

    # Case 2: Processed Spandan subset dictionary {"file_id": ..., "sampling_rate": ..., "signal": [...], "bpm": ...}
    if "signal" in rec and isinstance(rec["signal"], (list, np.ndarray)):
        arr = np.asarray(rec["signal"], dtype=np.float32)
        fs = float(rec["sampling_rate"]) if "sampling_rate" in rec else None
        fid = str(rec.get("file_id", filename))
        bpm = float(rec["bpm"]) if "bpm" in rec and rec["bpm"] is not None else None
        return arr, fs, fid, bpm, "signal"

    # Case 3: Raw Spandan dictionary with optical/accelerometer payload {"file_id": ..., "bpm": ..., "ppg": [...]}
    if "ppg" in rec:
        adapter = SpandanAdapter(primary_channel=primary_channel, fallback_sampling_rate=fallback_fs)
        parsed = adapter.parse_record(rec)
        arr = parsed["signal"]
        fs = parsed["sampling_rate"]
        fid = parsed["file_id"]
        bpm = parsed["bpm"] if not np.isnan(parsed.get("bpm", np.nan)) else None
        return arr, fs, fid, bpm, primary_channel

    # Case 4: Generic dictionary with other common PPG keys
    for cand in ("ppg", "data", "values", "samples", "waveform", "amplitudes"):
        if cand in rec and isinstance(rec[cand], (list, np.ndarray)):
            arr = np.asarray(rec[cand], dtype=np.float32)
            fs = float(rec["sampling_rate"]) if "sampling_rate" in rec else (
                float(rec["fs"]) if "fs" in rec else None
            )
            fid = str(rec.get("file_id", filename))
            bpm = float(rec["bpm"]) if "bpm" in rec and rec["bpm"] is not None else None
            return arr, fs, fid, bpm, cand

    raise UploadValidationError(
        "Could not locate PPG amplitude values in JSON. Expected keys: 'signal', 'ppg', or 'data'."
    )

