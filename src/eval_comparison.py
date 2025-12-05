"""
eval_comparison.py

Runs dp_eval_gaussian.py and dp_eval_laplace.py in order to compare summary statistics.

"""

import os
import json
import numpy as np
import pandas as pd

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# load Gaussian evaluation results
gaussian_eval_path = os.path.join(REPORT_DIR, "dp_eval_summary_gaussian.json")
with open(gaussian_eval_path, "r") as f:
    gaussian_eval = json.load(f)

# load Gaussian evaluation runs
gaussian_runs_path = os.path.join(REPORT_DIR, "dp_eval_runs_gaussian.csv")
gaussian_df = pd.read_csv(gaussian_runs_path)

# load Laplace evaluation results
laplace_eval_path = os.path.join(REPORT_DIR, "dp_eval_summary_laplace.json")
with open(laplace_eval_path, "r") as f:
    laplace_eval = json.load(f)

# load Laplace evaluation runs
laplace_runs_path = os.path.join(REPORT_DIR, "dp_eval_runs_laplace.csv")
laplace_df = pd.read_csv(laplace_runs_path)

print(gaussian_df, laplace_df)
