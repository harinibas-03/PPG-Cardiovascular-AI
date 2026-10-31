"""
Unit tests for Spandan-1M dataset adapter.
"""

import pytest
import numpy as np
from arohan_ppg_cvd.adapters.spandan_adapter import SpandanAdapter


def test_spandan_adapter_parse_dict_list():
    adapter = SpandanAdapter(primary_channel="g", fallback_sampling_rate=30.0)

    # Emulate Spandan format: ppg is list of dicts with r, g, b, t, x, y, z
    n = 90  # 3 seconds at 30 Hz
    t = [i * 33.33 for i in range(n)]  # milliseconds
    raw_record = {
        "file_id": "spandan_rec_042",
        "bpm": 74.5,
        "ppg": [
            {
                "r": float(100 + i % 10),
                "g": float(200 + 15 * np.sin(2 * np.pi * 1.2 * (i / 30.0))),
                "b": float(50),
                "t": t[i],
                "x": 0.01,
                "y": 0.02,
                "z": 0.98,
            }
            for i in range(n)
        ],
    }

    parsed = adapter.parse_record(raw_record)
    assert parsed["file_id"] == "spandan_rec_042"
    assert parsed["bpm"] == 74.5
    assert len(parsed["signal"]) == n
    assert parsed["rgb"].shape == (n, 3)
    assert 28.0 <= parsed["sampling_rate"] <= 32.0  # ~30 Hz estimated from t
    assert parsed["accel"].shape == (n, 3)


def test_spandan_adapter_fallback_handling():
    adapter = SpandanAdapter()
    empty_record = {"file_id": "empty", "bpm": None, "ppg": []}
    parsed = adapter.parse_record(empty_record)
    assert parsed["file_id"] == "empty"
    assert len(parsed["signal"]) == 0
    assert parsed["sampling_rate"] == 30.0


def test_spandan_adapter_1d_array_input():
    adapter = SpandanAdapter()
    rec = {"file_id": "raw_arr", "bpm": 65, "ppg": [1.0, 2.0, 3.0, 4.0, 5.0]}
    parsed = adapter.parse_record(rec)
    assert len(parsed["signal"]) == 5
