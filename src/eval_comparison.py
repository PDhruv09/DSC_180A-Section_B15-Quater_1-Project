"""
eval_comparison.py

Compare L_inf and IOU over runs for Gaussian and Laplace mechanisms

"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# create 2x2 plot grid
plot, axes = plt.subplots(2, 2, figsize=(14, 12))

# create graph for Gaussian L_inf over runs, sync y-axis limit
axes[0, 0].scatter(gaussian_df['run'], gaussian_df['L_inf'])
axes[0, 0].set_title("Gaussian L_inf over runs")
axes[0, 0].set_ylim(0, 0.2)

# create graph for Laplace L_inf over runs, sync y-axis limit
axes[0, 1].scatter(laplace_df['run'], laplace_df['L_inf'], color='red')
axes[0, 1].set_title("Laplace L_inf over runs")
axes[0, 1].set_ylim(0, 0.2)

# create graph for Gaussian IOU over runs
axes[1, 0].scatter(gaussian_df['run'], gaussian_df['IOU'])
axes[1, 0].set_title("Gaussian IOU over runs")

# create graph for Laplace IOU over runs
axes[1, 1].scatter(laplace_df['run'], laplace_df['IOU'], color='red')
axes[1, 1].set_title("Laplace L_inf over runs")

VISUALS_DIR = os.path.join(ROOT_DIR, "visuals")
img_path = os.path.join(VISUALS_DIR, "eval_comp_plots.png")
plt.savefig(img_path)