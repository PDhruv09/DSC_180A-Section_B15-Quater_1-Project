"""
dp_mechanism.py

Implements a user-level (adds/removes-one-user) Gaussian DP mechanism for the telemetry dataset, BY HAND.

Pipeline of the file:
1) Loads cleaned telemetry data (telemetry_clean.csv).
2) Collapse to user-level "primary product" contributions of the data.
3) Compute user-level counts per product:
   - U_total_P = # of users primarily using product P
   - U_error_P = # of those users who ever had an 'error' on P
4) Compute TRUE error rates and z-scores.
5) Add Gaussian noise with analytically chosen sigma to U_total and U_error.
6) Recompute DP error rates and DP z-scores for comparsion.
7) Return both true and DP z-scores and the corresponding "top sets" 
    top sets refers to (products with z-score > 0).

For clarity this file does NOT generate synthetic data; it only produces DP-perturbed aggregates that can be used to 
generate synthetic data or for direct evaluation (L_inf, IOU) in dp_eval_gaussiant.py.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import zscore

# -----------------------------
#  Global config / constants
#  to make sure that directory are the same even if you run on different computer
# -----------------------------

# Project paths (relative to repo root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# Clean telemetry file produced by preprocess.py
CLEAN_PATH = os.path.join(PROC_DIR, "telemetry_clean.csv")

# Default DP parameters from the jupyter notebook given by the Professor
DEFAULT_EPSILON = 2.0
DEFAULT_DELTA = 1e-6

# L2 sensitivity of the user-level count vector:
# We will design the pipeline so that each user affects exactly ONE product,
# and for that product:
#   U_total_P changes by at most 1,
#   U_error_P changes by at most 1.
# So for the concatenated vector [U_total_P, U_error_P], the difference
# in L2 norm between neighboring datasets is sqrt(1^2 + 1^2) = sqrt(2).
L2_SENSITIVITY = np.sqrt(2.0)


# -----------------------------
#  Helper: Load cleaned data
# -----------------------------

def load_clean_telemetry():
    """
    Load the cleaned telemetry CSV produced by preprocess.py.

    Expected columns:
        - Product Type
        - Event Type
        - Time of Event
        - event_date
        - event_hour
        - event_dow
        - is_weekend
        - User ID
    """
    df = pd.read_csv(CLEAN_PATH, parse_dates=["Time of Event"])
    return df


# -----------------------------
#  User-level primary product
# -----------------------------

def build_user_level_primary_product(df: pd.DataFrame):
    """
    Collapse the telemetry data to ONE row per (User ID, Product Type),
    then pick a SINGLE "primary product" per user.

    Why do we do this?
    - We want USER-LEVEL DP (add/remove one user).
    - If a user touches many products, they would otherwise affect many
      product-specific counts, increasing sensitivity.
    - To keep sensitivity analysis simple and explicit, we assign each user
      to a single "primary product" and ignore their contributions to others
      in the DP mechanism (we can still use full data for non-private stats).

    Primary product definition:
    - For each user, we choose the product where they have the MOST events.
    - If there is a tie, pandas' idxmax picks the first occurrence.

    Returns a DataFrame with the following columns:
        - User ID
        - Product Type (primary product)
        - n_events (events for that user on that primary product)
        - any_error (bool: did they ever have an 'error' on that product?)
    """

    # Group by (User ID, Product Type) to compute per-user-per-product stats
    user_prod = (
        df.groupby(["User ID", "Product Type"])
          .agg(
              n_events=("Event Type", "size"),
              any_error=("Event Type", lambda x: (x == "error").any()),
          )
          .reset_index()
    )

    # For each user, pick the row with max n_events (primary product)
    idx = user_prod.groupby("User ID")["n_events"].idxmax()
    primary = user_prod.loc[idx].copy()

    # Sanity check: one row per user
    assert primary["User ID"].is_unique, "Primary-product table must have one row per user."

    return primary


# -----------------------------
#  True user-level counts and z-scores
# -----------------------------

def compute_true_user_level_counts(primary: pd.DataFrame):
    """
    Given the primary-product per-user DataFrame, compute TRUE user-level counts:
        U_total_P = # users whose primary product is P
        U_error_P = # of those users who have any_error == True on P

    Then compute:
        ErrorRate_P = U_error_P / U_total_P
        Z_true_P = z-score of ErrorRate_P across products.

    Returns a DataFrame indexed by Product Type with columns:
        - U_total
        - U_error
        - error_rate_true
        - z_true
    """

    # Total users per product (primary assignment)
    U_total = (
        primary.groupby("Product Type")["User ID"]
               .nunique()
               .rename("U_total")
    )

    # Users per product with at least one error on that product
    primary_error = primary[primary["any_error"]]
    U_error = (
        primary_error.groupby("Product Type")["User ID"]
                     .nunique()
                     .rename("U_error")
    )

    # Combine into one table, fill missing error counts with 0
    counts = pd.concat([U_total, U_error], axis=1).fillna(0)
    counts["U_total"] = counts["U_total"].astype(int)
    counts["U_error"] = counts["U_error"].astype(int)

    # Error rate per product (handle division by zero explicitly)
    counts["error_rate_true"] = 0.0
    nonzero_mask = counts["U_total"] > 0
    counts.loc[nonzero_mask, "error_rate_true"] = (
        counts.loc[nonzero_mask, "U_error"] / counts.loc[nonzero_mask, "U_total"]
    )

    # Z-score of error rates across products.
    # If there's only one unique rate, zscore is undefined -> set to 0.
    if counts["error_rate_true"].nunique() > 1:
        counts["z_true"] = zscore(counts["error_rate_true"])
    else:
        counts["z_true"] = 0.0

    return counts


# -----------------------------
#  Gaussian DP mechanism
# -----------------------------

def gaussian_sigma(l2_sensitivity: float, epsilon: float, delta: float):
    """
    Compute the Gaussian noise scale (sigma) for (epsilon, delta)-DP
    using the standard analytic bound from the basic Gaussian mechanism:

        sigma >= (Δ_2 / ε) * sqrt(2 * ln(1.25 / δ))

    where:
        Δ_2 = L2 sensitivity of the vector-valued function.
    """
    return (l2_sensitivity / epsilon) * np.sqrt(2.0 * np.log(1.25 / delta))


def add_gaussian_noise_to_counts(
    counts: pd.DataFrame,
    epsilon: float = DEFAULT_EPSILON,
    delta: float = DEFAULT_DELTA,
    random_state: int | None = None,
):
    """
    Apply the Gaussian mechanism to the user-level counts (U_total, U_error)
    for each product type.

    We treat the vector f(D) = [U_total_P, U_error_P]_P as our query.
    By construction of primary products, adding/removing one user changes
    U_total and U_error for a single product P by at most 1 each, so the
    L2 sensitivity is sqrt(1^2 + 1^2) = sqrt(2).

    We then add N(0, sigma^2) noise independently to EACH coordinate.

    Returns a new DataFrame with additional columns:
        - U_total_noisy
        - U_error_noisy
        - error_rate_dp
        - z_dp
    """

    rng = np.random.default_rng(seed=random_state)

    # Copy to avoid mutating input
    noisy = counts.copy()

    # Compute sigma from (epsilon, delta, L2_sensitivity)
    sigma = gaussian_sigma(L2_SENSITIVITY, epsilon, delta)
    print(f"[dp_mechanism] Using sigma = {sigma:.4f} for Gaussian noise.")

    # Add Gaussian noise to U_total and U_error
    for col in ["U_total", "U_error"]:
        noise = rng.normal(loc=0.0, scale=sigma, size=len(noisy))
        noisy[f"{col}_noisy"] = noisy[col].astype(float) + noise

        # Post-processing:
        # - Negative counts do NOT make sense -> clamp at 0.
        # - We do NOT round to integer here; we can leave them real-valued
        #   since they are only used to form rates.
        noisy[f"{col}_noisy"] = noisy[f"{col}_noisy"].clip(lower=0.0)

    # Compute DP error rates using noisy counts.
    noisy["error_rate_dp"] = 0.0
    nonzero = noisy["U_total_noisy"] > 0
    noisy.loc[nonzero, "error_rate_dp"] = (
        noisy.loc[nonzero, "U_error_noisy"] / noisy.loc[nonzero, "U_total_noisy"]
    )

    # Compute DP z-scores over products.
    if noisy["error_rate_dp"].nunique() > 1:
        noisy["z_dp"] = zscore(noisy["error_rate_dp"])
    else:
        noisy["z_dp"] = 0.0

    return noisy


# -----------------------------
#  Top-set helpers
# -----------------------------

def get_top_set_from_zscores(z_series: pd.Series, threshold: float = 0.0):
    """
    Given a Series of z-scores indexed by product type,
    return the set of product types whose z-score > threshold.

    By default, threshold = 0.0 as in the assignment.
    """
    mask = z_series > threshold
    return set(z_series.index[mask])


# -----------------------------
#  Main entry point (for quick manual testing)
# -----------------------------

def main():
    """
    - Load cleaned telemetry.
    - Build primary-product user-level table.
    - Compute TRUE counts / error rates / z-scores.
    - Apply Gaussian DP to get NOISY counts / z-scores.
    - Save a combined CSV with both true and DP stats to reports/.
    """

    print("[dp_mechanism] Loading cleaned telemetry...")
    df = load_clean_telemetry()
    print(f"[dp_mechanism] Clean shape: {df.shape}")

    print("[dp_mechanism] Building user-level primary product table...")
    primary = build_user_level_primary_product(df)
    print(f"[dp_mechanism] Primary table shape: {primary.shape}")

    print("[dp_mechanism] Computing TRUE user-level counts and z-scores...")
    true_counts = compute_true_user_level_counts(primary)

    print("[dp_mechanism] Applying Gaussian DP to counts...")
    noisy_counts = add_gaussian_noise_to_counts(
        true_counts, epsilon=DEFAULT_EPSILON, delta=DEFAULT_DELTA, random_state=42
    )

    # Merge true and DP stats into one DataFrame for inspection
    merged = noisy_counts.copy()  # already has U_total, U_error, error_rate_true, z_true

    out_path = os.path.join(REPORT_DIR, "dp_user_level_counts_and_zscores.csv")
    merged.to_csv(out_path)
    print(f"[dp_mechanism] Saved DP + true stats -> {out_path}")

    true_top = get_top_set_from_zscores(merged["z_true"])
    dp_top = get_top_set_from_zscores(merged["z_dp"])
    print(f"[dp_mechanism] TRUE top set (z_true > 0): {true_top}")
    print(f"[dp_mechanism] DP   top set (z_dp > 0):   {dp_top}")


if __name__ == "__main__":
    main()