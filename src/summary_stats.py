# src/summary_stats.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- resolve paths relative to project root ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(ROOT_DIR, "data", "processed")
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
VIS_DIR = os.path.join(ROOT_DIR, "visuals")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

PROC_PATH = os.path.join(PROC_DIR, "support2_clean.csv")
RAW_PATH = os.path.join(RAW_DIR, "support2.csv")

def describe_safe(df: pd.DataFrame) -> pd.DataFrame:
    # compatible across pandas versions
    try:
        return df.describe(include="all")
    except TypeError:
        # very old pandas sometimes chokes on include='all'
        num = df.describe(include=[np.number])
        obj = df.describe(include=[object])
        return pd.concat([num, obj], axis=1)

def save_corr_heatmap(df_num: pd.DataFrame, out_png: str, out_csv: str):
    if df_num.shape[1] < 2:
        print("[info] Skipping correlation: fewer than 2 numeric columns.")
        return
    corr = df_num.corr(method="pearson")
    corr.to_csv(out_csv)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title("Numeric Feature Correlation Heatmap")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[save] Correlations -> {out_csv}")
    print(f"[save] Heatmap -> {out_png}")

def main():
    print("[load] Loading cleaned dataset...")
    df_proc = pd.read_csv(PROC_PATH)
    print(f"[info] Processed shape: {df_proc.shape}")

    # --- Summary on processed ---
    desc = describe_safe(df_proc)
    desc.to_csv(os.path.join(REPORT_DIR, "support2_summary_statistics.csv"))
    print("[save] Summary (processed) -> reports/support2_summary_statistics.csv")

    # --- Cardinality (processed) ---
    card_df = df_proc.nunique().sort_values(ascending=False).reset_index()
    card_df.columns = ["column", "n_unique"]
    card_df.to_csv(os.path.join(REPORT_DIR, "support2_cardinality.csv"), index=False)
    print("[save] Cardinality (processed) -> reports/support2_cardinality.csv")

    # --- Missingness: raw vs processed (if raw exists) ---
    if os.path.exists(RAW_PATH):
        df_raw = pd.read_csv(RAW_PATH)
        miss_raw = (df_raw.isna().mean() * 100).round(4).sort_values(ascending=False).reset_index()
        miss_raw.columns = ["column", "missing_pct_raw"]
        miss_proc = (df_proc.isna().mean() * 100).round(4).sort_values(ascending=False).reset_index()
        miss_proc.columns = ["column", "missing_pct_processed"]

        miss = miss_raw.merge(miss_proc, on="column", how="outer").fillna(0.0)
        miss.to_csv(os.path.join(REPORT_DIR, "support2_missingness_raw_vs_processed.csv"), index=False)
        print("[save] Missingness (raw vs processed) -> reports/support2_missingness_raw_vs_processed.csv")
    else:
        # fallback: just processed
        miss_proc = (df_proc.isna().mean() * 100).round(4).sort_values(ascending=False).reset_index()
        miss_proc.columns = ["column", "missing_pct_processed"]
        miss_proc.to_csv(os.path.join(REPORT_DIR, "support2_missingness.csv"), index=False)
        print("[save] Missingness (processed) -> reports/support2_missingness.csv")

    # --- Correlations on processed numeric columns ---
    num_df = df_proc.select_dtypes(include=[np.number])
    save_corr_heatmap(
        num_df,
        out_png=os.path.join(VIS_DIR, "correlation_heatmap.png"),
        out_csv=os.path.join(REPORT_DIR, "support2_numeric_correlations.csv"),
    )

    print("[done] Summary statistics complete.")

if __name__ == "__main__":
    main()
