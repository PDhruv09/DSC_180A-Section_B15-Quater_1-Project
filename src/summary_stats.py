# src/summary_stats.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Paths relative to repo root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
VIS_DIR = os.path.join(ROOT_DIR, "visuals")
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

PROC_PATH = os.path.join(PROC_DIR, "telemetry_clean.csv")

def main():
    print("[summary] Loading cleaned telemetry data...")
    df = pd.read_csv(PROC_PATH, parse_dates=["Time of Event"])
    print(f"[summary] Shape: {df.shape}")

    # ---- Basic summary ----
    summary = df.describe(include="all")
    summary.to_csv(os.path.join(REPORT_DIR, "telemetry_summary_statistics.csv"))
    print("[summary] -> reports/telemetry_summary_statistics.csv")

    # ---- Core aggregates ----
    # By Product
    by_product = (
        df.groupby("Product Type")
          .size()
          .rename("event_count")
          .reset_index()
          .sort_values("event_count", ascending=False)
    )
    by_product.to_csv(os.path.join(REPORT_DIR, "telemetry_event_counts_by_product.csv"), index=False)
    print("[summary] -> reports/telemetry_event_counts_by_product.csv")

    # By Event Type
    by_event = (
        df.groupby("Event Type")
          .size()
          .rename("event_count")
          .reset_index()
          .sort_values("event_count", ascending=False)
    )
    by_event.to_csv(os.path.join(REPORT_DIR, "telemetry_event_counts_by_type.csv"), index=False)
    print("[summary] -> reports/telemetry_event_counts_by_type.csv")

    # Events per day
    df["event_date"] = pd.to_datetime(df["event_date"])
    by_day = (
        df.groupby("event_date")
          .size()
          .rename("event_count")
          .reset_index()
          .sort_values("event_date")
    )
    by_day.to_csv(os.path.join(REPORT_DIR, "telemetry_events_per_day.csv"), index=False)
    print("[summary] -> reports/telemetry_events_per_day.csv")

    # Unique users per product (useful for add/remove-by-user DP later)
    users_per_product = (
        df.groupby("Product Type")["User ID"]
          .nunique()
          .rename("unique_users")
          .reset_index()
          .sort_values("unique_users", ascending=False)
    )
    users_per_product.to_csv(os.path.join(REPORT_DIR, "telemetry_unique_users_per_product.csv"), index=False)
    print("[summary] -> reports/telemetry_unique_users_per_product.csv")

    # ---- Error rates & Z-scores (matches notebook formulas) ----
    error_counts = (
        df[df["Event Type"] == "error"]
        .groupby("Product Type")
        .size()
        .rename("Error Count")
    )
    total_counts = (
        df.groupby("Product Type")
        .size()
        .rename("Total Count")
    )
    err = pd.concat([error_counts, total_counts], axis=1).fillna(0)
    err["Error Rate"] = err["Error Count"] / err["Total Count"]
    # z-score of error rates across product types
    # if a single product type exists, zscore returns NaN; handle that:
    if err["Error Rate"].nunique() > 1:
        err["Error Rate Z-Score"] = zscore(err["Error Rate"])
    else:
        err["Error Rate Z-Score"] = 0.0

    err.sort_values("Error Rate", ascending=False, inplace=True)
    err.to_csv(os.path.join(REPORT_DIR, "telemetry_error_rates_zscores.csv"))
    print("[summary] -> reports/telemetry_error_rates_zscores.csv")

    # ---- Plots ----
    # Event type distribution
    plt.figure(figsize=(6, 4))
    plt.bar(by_event["Event Type"], by_event["event_count"])
    plt.xlabel("Event Type")
    plt.ylabel("Count")
    plt.title("Event Type Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "event_type_distribution.png"), dpi=200)
    plt.close()

    # Product type distribution
    plt.figure(figsize=(6, 4))
    plt.bar(by_product["Product Type"], by_product["event_count"])
    plt.xlabel("Product Type")
    plt.ylabel("Count")
    plt.title("Product Type Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "product_type_distribution.png"), dpi=200)
    plt.close()

    # Events over time (daily)
    plt.figure(figsize=(8, 4))
    plt.plot(by_day["event_date"], by_day["event_count"])
    plt.xlabel("Date")
    plt.ylabel("Event Count")
    plt.title("Events Per Day")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "events_over_time.png"), dpi=200)
    plt.close()

    # Error rate per product (bar)
    plt.figure(figsize=(7, 4))
    plt.bar(err.index.astype(str), err["Error Rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Error Rate")
    plt.title("Error Rate by Product Type")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "error_rate_by_product.png"), dpi=200)
    plt.close()

    # Z-score per product (bar)
    plt.figure(figsize=(7, 4))
    plt.bar(err.index.astype(str), err["Error Rate Z-Score"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Z-Score (Error Rate)")
    plt.title("Error Rate Z-Score by Product Type")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "error_rate_zscore_by_product.png"), dpi=200)
    plt.close()

    print("[summary] Saved plots -> visuals/")
    print("[summary] Done.")

if __name__ == "__main__":
    main()
