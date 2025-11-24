# src/preprocess.py
import os
import pandas as pd
import numpy as np

# Resolve paths relative to project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROC_DIR = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

RAW_PATH = os.path.join(RAW_DIR, "synthetic_telemetry_data.csv")
PROC_PATH = os.path.join(PROC_DIR, "telemetry_clean.csv")

def main():
    print("[preprocess] Loading raw telemetry data...")
    df = pd.read_csv(RAW_PATH)

    # Basic sanity checks & trims
    df["Product Type"] = df["Product Type"].astype(str).str.strip()
    df["Event Type"] = df["Event Type"].astype(str).str.strip()
    df["User ID"] = df["User ID"].astype(str).str.strip()

    # Parse timestamp
    df["Time of Event"] = pd.to_datetime(df["Time of Event"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Time of Event"])
    if len(df) < before:
        print(f"[preprocess] Dropped {before - len(df)} rows with invalid timestamps.")

    # Time features
    df["event_date"] = df["Time of Event"].dt.date
    df["event_hour"] = df["Time of Event"].dt.hour
    df["event_dow"] = df["Time of Event"].dt.dayofweek  # 0=Mon..6=Sun
    df["is_weekend"] = df["event_dow"].isin([5, 6]).astype(int)

    # Column order
    df = df[[
        "Product Type", "Event Type", "Time of Event",
        "event_date", "event_hour", "event_dow", "is_weekend",
        "User ID"
    ]]

    df.to_csv(PROC_PATH, index=False)
    print(f"[preprocess] Saved -> {PROC_PATH}")
    print(f"[preprocess] Final shape: {df.shape}")

if __name__ == "__main__":
    main()
