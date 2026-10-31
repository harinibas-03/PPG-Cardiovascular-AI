"""
Spandan-1M Small Subset Downloader
==================================
Streams and saves a tiny subset of the Spandan-1M-V1.0-raw dataset from Hugging Face
without downloading the massive 43 GB dataset.

Usage:
    py scripts/download_spandan_subset.py --max-samples 50 --output data/raw/spandan_subset.json
"""

import os
import sys
import argparse
import json
import logging

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arohan_ppg_cvd.adapters.spandan_adapter import SpandanAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_spandan_subset")


def main():
    parser = argparse.ArgumentParser(description="Download a small subset of Spandan-1M via streaming.")
    parser.add_argument("--repo-id", type=str, default="ekacare/spandan-1M-V1.0-raw", help="Hugging Face repo ID")
    parser.add_argument("--config", type=str, default="batch_0", help="Dataset config name (e.g. 'batch_0')")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=50, help="Number of records to stream (default: 50)")
    parser.add_argument("--output", type=str, default="data/raw/spandan_subset.json", help="Path to save output JSON")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional Hugging Face access token")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    logger.info(f"Target: Stream {args.max_samples} records from {args.repo_id} (config: {args.config})...")
    try:
        stream = SpandanAdapter.load_stream_subset(
            repo_id=args.repo_id,
            config_name=args.config,
            split=args.split,
            max_samples=args.max_samples,
            token=args.hf_token,
        )

        records = []
        for item in stream:
            # Convert numpy arrays to serializable lists
            rec = {
                "file_id": item["file_id"],
                "bpm": item["bpm"],
                "sampling_rate": item["sampling_rate"],
                "signal": item["signal"].tolist(),
            }
            records.append(rec)
            if len(records) % 10 == 0:
                logger.info(f"Streamed {len(records)}/{args.max_samples} samples...")

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        logger.info(f"Successfully saved {len(records)} samples to {args.output}")

    except Exception as e:
        logger.error(f"Failed to stream dataset from '{args.repo_id}': {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
