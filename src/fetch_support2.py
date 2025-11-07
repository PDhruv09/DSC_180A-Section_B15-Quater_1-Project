# src/fetch_support2.py
import os
import sys
import json
import subprocess

def ensure_pkg(pkg: str):
    try:
        __import__(pkg)
    except ImportError:
        print(f"[setup] '{pkg}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def main():
    # 1) deps
    ensure_pkg("ucimlrepo")
    ensure_pkg("pandas")

    from ucimlrepo import fetch_ucirepo
    import pandas as pd

    # 2) folders
    RAW_DIR = os.path.join("data", "raw")
    os.makedirs(RAW_DIR, exist_ok=True)

    # 3) fetch
    print("[fetch] Downloading SUPPORT2 (UCI id=880)...")
    support2 = fetch_ucirepo(id=880)
    X = support2.data.features
    y = support2.data.targets

    # 4) save data
    combined = pd.concat([X, y], axis=1)

    f_features = os.path.join(RAW_DIR, "support2_features.csv")
    f_targets  = os.path.join(RAW_DIR, "support2_targets.csv")
    f_combined = os.path.join(RAW_DIR, "support2.csv")
    f_metadata = os.path.join(RAW_DIR, "support2_metadata.json")
    f_vars     = os.path.join(RAW_DIR, "support2_variables.csv")

    X.to_csv(f_features, index=False)
    y.to_csv(f_targets, index=False)
    combined.to_csv(f_combined, index=False)

    # 5) save metadata/variables
    try:
        with open(f_metadata, "w", encoding="utf-8") as fh:
            # support2.metadata is a dict-like object
            json.dump(support2.metadata, fh, indent=2)
    except Exception as e:
        print(f"[warn] Could not write metadata JSON: {e}")

    try:
        support2.variables.to_csv(f_vars, index=False)
    except Exception as e:
        print(f"[warn] Could not write variables CSV: {e}")

    # 6) minimal confirmations
    print("[done] Saved:")
    print(f"  - Features : {f_features}  shape={X.shape}")
    print(f"  - Targets  : {f_targets}   shape={y.shape}")
    print(f"  - Combined : {f_combined}  shape={combined.shape}")
    print(f"  - Metadata : {f_metadata}")
    print(f"  - Variables: {f_vars}")

if __name__ == "__main__":
    main()
