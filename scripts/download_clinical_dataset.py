"""Download and unpack the PPG-BP Clinical Benchmark Dataset (Liang et al., 2018).

Dataset Details:
- Source: Guilin People's Hospital / Figshare / GitHub sanqiaiziji/PPG_BP_dataset
- Reference: Liang Y, Chen Z, Liu G, Elgendi M. (2018) Scientific Data (Nature), 5:180076.
- Population: Guilin, China (External Clinical Benchmark - NOT Indian population)
- Diseases: Hypertension, Cerebral Infarction, Diabetes, Healthy Controls
"""

import os
import sys
import urllib.request
import zipfile
import pandas as pd

CLINICAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clinical")
ZIP_URL = "https://raw.githubusercontent.com/sanqiaiziji/PPG_BP_dataset/master/PPG-BP_Database.zip"
TABLE_URL = "https://raw.githubusercontent.com/sanqiaiziji/PPG_BP_dataset/master/Table_1.xlsx"

def download_file(url: str, dest_path: str):
    print(f"Downloading {url} -> {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
    print(f"Saved {dest_path} ({os.path.getsize(dest_path)} bytes)")

def main():
    os.makedirs(CLINICAL_DIR, exist_ok=True)
    zip_path = os.path.join(CLINICAL_DIR, "PPG-BP_Database.zip")
    table_path = os.path.join(CLINICAL_DIR, "Table_1.xlsx")
    
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000000:
        download_file(ZIP_URL, zip_path)
    else:
        print(f"Zip file already present: {zip_path}")
        
    if not os.path.exists(table_path) or os.path.getsize(table_path) < 10000:
        download_file(TABLE_URL, table_path)
    else:
        print(f"Metadata table already present: {table_path}")
        
    # Unpack zip
    raw_dir = os.path.join(CLINICAL_DIR, "raw_ppg_txt")
    if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) < 100:
        print("Extracting PPG-BP_Database.zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(CLINICAL_DIR)
        print("Extracted successfully.")
        
    # Inspect metadata table
    df_meta = pd.read_excel(table_path)
    print("\nMetadata Table Preview:")
    print(df_meta.head())
    print("\nColumns:", df_meta.columns.tolist())
    print("\nShape:", df_meta.shape)
    if "Diagnosis" in df_meta.columns or any("diagnos" in c.lower() for c in df_meta.columns):
        diag_col = [c for c in df_meta.columns if "diagnos" in c.lower()][0]
        print(f"\nDiagnosis Value Counts ({diag_col}):")
        print(df_meta[diag_col].value_counts())

if __name__ == "__main__":
    main()

