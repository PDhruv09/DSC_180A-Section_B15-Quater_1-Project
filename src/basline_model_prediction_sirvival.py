# src/Baseline_mdel_prediction_surival.py
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss, brier_score_loss, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

# --- make all paths relative to repo root ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
VIS_DIR = os.path.join(ROOT_DIR, "visuals")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

PROC_PATH = os.path.join(DATA_DIR, "support2_clean.csv")

TARGET_CANDIDATES = ["death", "hospdead", "surv6m", "surv2m"]
DROP_IF_PRESENT = ["d.time", "dtime", "slos", "los", "length_of_stay"]

def pick_target(df):
    cols_lower = {c.lower(): c for c in df.columns}
    for c in TARGET_CANDIDATES:
        if c in cols_lower:
            return cols_lower[c]
    raise ValueError(f"No target found. Expected one of {TARGET_CANDIDATES}")

def make_mortality_target(y):
    lname = y.name.lower()
    if lname.startswith("surv"):
        return 1 - y.astype(int)
    if y.dtype.kind not in "biu":
        y = y.astype(str).str.strip().str.lower().map(
            {"1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0}
        ).fillna(y)
    y = pd.to_numeric(y, errors="coerce")
    uniq = set(y.dropna().unique().tolist())
    if not uniq.issubset({0, 1}):
        y = (y > 0).astype(int)
    return y

def reliability_plot(y_true, y_prob, out_path):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 4))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram (Baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    print("[baseline] Loading processed dataset...")
    df = pd.read_csv(PROC_PATH)
    print(f"[baseline] Shape: {df.shape}")

    target_col = pick_target(df)
    print(f"[baseline] Target: {target_col}")

    y_raw = df[target_col].copy()
    X = df.drop(columns=[target_col])

    drop_cols = [c for c in X.columns if c.lower() in DROP_IF_PRESENT]
    if drop_cols:
        X = X.drop(columns=drop_cols)
        print(f"[baseline] Dropped columns: {drop_cols}")

    y = make_mortality_target(y_raw)
    if y.isna().any():
        keep = ~y.isna()
        X, y = X.loc[keep], y.loc[keep]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int), test_size=0.2, random_state=42, stratify=y.astype(int)
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)

    prob_test = clf.predict_proba(X_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)

    metrics = {
        "script": "Baseline_mdel_prediction_surival.py",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "auc": float(roc_auc_score(y_test, prob_test)),
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "log_loss": float(log_loss(y_test, prob_test)),
        "brier": float(brier_score_loss(y_test, prob_test))
    }

    metrics_path = os.path.join(REPORT_DIR, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] Metrics -> {metrics_path}")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, prob_test)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={metrics['auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Baseline ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "baseline_roc_curve.png"), dpi=200)
    plt.close()

    # Reliability
    reliability_plot(y_test, prob_test, os.path.join(VIS_DIR, "baseline_reliability.png"))

    # Feature importances
    coefs = pd.Series(clf.coef_[0], index=X.columns)
    imp_df = pd.DataFrame({
        "feature": coefs.index,
        "abs_coef": coefs.abs().values,
        "coef": coefs.values
    }).sort_values("abs_coef", ascending=False)
    imp_df.to_csv(os.path.join(REPORT_DIR, "baseline_feature_importance.csv"), index=False)
    print("[baseline] Done — results saved in reports/ and visuals/")

if __name__ == "__main__":
    main()
