# src/preprocess.py
import os
import pandas as pd
import numpy as np

RAW_PATH = os.path.join("data", "raw", "support2.csv")
PROC_DIR = os.path.join("data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)
PROC_PATH = os.path.join(PROC_DIR, "support2_clean.csv")

def main():
    print("[load] Reading raw dataset...")
    df = pd.read_csv(RAW_PATH)

    print(f"[info] Original shape: {df.shape}")

    # --- Type fixing ---
    # 1. Strip column names
    df.columns = df.columns.str.strip()

    # 2. Convert obvious numeric columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass

    # 3. Handle missing values
    # Numeric → fill with median; Categorical → fill with mode
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)

    # 4. Encode categorical columns (if any)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print(f"[encode] Encoding categorical columns: {list(cat_cols)}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 5. Save cleaned dataset
    df.to_csv(PROC_PATH, index=False)
    print(f"[save] Cleaned dataset saved at: {PROC_PATH}")
    print(f"[done] Final shape: {df.shape}")

if __name__ == "__main__":
    main()
