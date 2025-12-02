"""
dp_eval.py

Runs the Gaussian DP mechanism multiple times and evaluates utility
via:
    - L_infinity error on z-scores
    - IOU on top sets (products with z > 0)

This is the "privacy evaluation" part:
we KEEP the mechanism fixed but vary the randomness, and see how much
its outputs deviate from the non-private ground truth.
"""

import os
import json
import numpy as np
import pandas as pd

from dp_Gaussian_mechanism import (
    load_clean_telemetry,
    build_user_level_primary_product,
    compute_true_user_level_counts,
    add_gaussian_noise_to_counts,
    get_top_set_from_zscores,
    DEFAULT_EPSILON,
    DEFAULT_DELTA,
)

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
#  Metric helpers (L_inf, IOU)
# -----------------------------

def compute_l_inf_error(true_z: pd.Series, dp_z: pd.Series) -> float:
    """
    Compute L_infinity error between true and DP z-score vectors.

    We align indices first (product types), then take:
        max_P |Z_true[P] - Z_dp[P]|
    """
    # Align by index (product type)
    true_aligned, dp_aligned = true_z.align(dp_z, join="inner")
    return float(np.max(np.abs(true_aligned - dp_aligned)))


def compute_iou(true_set: set, dp_set: set) -> float:
    """
    Compute Intersection over Union (IOU) between two sets of product types.
    IOU = |true ∩ dp| / |true ∪ dp|
    """
    intersection = len(true_set.intersection(dp_set))
    union = len(true_set.union(dp_set))
    return intersection / union if union > 0 else 0.0


def summarize_metric_array(values: list[float]) -> dict:
    """
    Given a list of metric values (e.g., 100 L_inf values),
    compute:
        - max
        - min
        - 5%, 50%, 95% quantiles

    and return them in a dict.
    """
    arr = np.asarray(values)
    summary = {
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "5%": float(np.percentile(arr, 5)),
        "50%": float(np.percentile(arr, 50)),
        "95%": float(np.percentile(arr, 95)),
    }
    return summary


# -----------------------------
#  Main evaluation loop
# -----------------------------

def main(
    num_runs: int = 100,
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
    base_seed: int = 12345,
):
    """
    Run the DP mechanism `num_runs` times with different random seeds
    and evaluate L_infinity and IOU for each run.

    Outputs:
        - reports/dp_eval_runs.csv: one row per run with L_inf and IOU
        - reports/dp_eval_summary.json: quantiles for both metrics
    """

    print("[dp_eval] Loading cleaned telemetry and building TRUE stats...")
    df = load_clean_telemetry()
    primary = build_user_level_primary_product(df)
    true_counts = compute_true_user_level_counts(primary)

    # Extract true z-scores and true top set once; they don't change
    z_true = true_counts["z_true"]
    true_top_set = get_top_set_from_zscores(z_true)

    print("[dp_eval] TRUE z-scores:")
    print(z_true)
    print(f"[dp_eval] TRUE top set (z_true > 0): {true_top_set}")

    l_inf_values = []
    iou_values = []

    # For reproducibility across runs, we seed with base_seed + run_idx
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print(f"[dp_eval] Run {run_idx+1}/{num_runs} (seed={seed})")

        # Apply DP mechanism with this seed
        noisy_counts = add_gaussian_noise_to_counts(
            true_counts, epsilon=epsilon, delta=delta, random_state=seed
        )

        # Extract DP z-scores and DP top set
        z_dp = noisy_counts["z_dp"]
        dp_top_set = get_top_set_from_zscores(z_dp)

        # Compute metrics
        l_inf = compute_l_inf_error(z_true, z_dp)
        iou = compute_iou(true_top_set, dp_top_set)

        l_inf_values.append(l_inf)
        iou_values.append(iou)

    # Turn into DataFrame and save all runs
    runs_df = pd.DataFrame(
        {
            "run": np.arange(num_runs),
            "L_inf": l_inf_values,
            "IOU": iou_values,
        }
    )
    runs_csv_path = os.path.join(REPORT_DIR, "dp_eval_runs.csv")
    runs_df.to_csv(runs_csv_path, index=False)
    print(f"[dp_eval] Saved per-run metrics -> {runs_csv_path}")

    # Summaries
    l_inf_summary = summarize_metric_array(l_inf_values)
    iou_summary = summarize_metric_array(iou_values)

    summary = {
        "epsilon": epsilon,
        "delta": delta,
        "num_runs": num_runs,
        "L_inf_summary": l_inf_summary,
        "IOU_summary": iou_summary,
    }

    summary_path = os.path.join(REPORT_DIR, "dp_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[dp_eval] Saved summary -> {summary_path}")
    print("[dp_eval] L_inf summary:", l_inf_summary)
    print("[dp_eval] IOU summary:  ", iou_summary)


if __name__ == "__main__":
    # Default: 100 runs, epsilon=2.0, delta=1e-6
    main()
