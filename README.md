# DSC_180A Section_B15 Quater_1 Project  
**Title:** Evaluating Differentially Private Synthetic Data Generation Using the SUPPORT2 Clinical Dataset  
**Team:** Dhruv Patel , Reva Agrawal, and Jordan Lambino 
**Meentor:** Prof. Yu-Xiang Wang  

## Overview
This project investigates the feasibility and accuracy of generating synthetic datasets that preserve privacy while maintaining statistical utility, using SUPPORT2 clinical dataset as a case study. Our primary goal is to explore how well differentially private mechanisms, including baseline noise addition and advanced synthetic data generation methods can reproduce the essential attributes and distributions of sensitive data without compromising individual privacy. By comparing real and synthetic data on measures such as correlation, attribute similarity, and predictive performance, we aim to evaluate the trade-off between privacy protection and data utility. The results will help determine whether differential privacy can support the release of realistic yet privacy-preserving clinical data for healthcare research and analysis.

## Repository Structure
- `data/` — Raw, processed, and synthetic versions of the dataset.
- `notebooks/` — Jupyter notebooks for data exploration, preprocessing, modeling, and evaluation.
- `src/` — Python modules for data processing and synthetic generation.
- `reports/` — Abstract, introduction, and final report files.
- `visuals/` — Plots and visualizations of key results.
- `scripts/` — Shell scripts for running experiments and reproducibility.

## Dependencies
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Data Fetching (SUPPORT2)
This project programmatically fetches the SUPPORT2 dataset from the UCI ML Repository using the `ucimlrepo` package for reproducibility.

### 1) Create & activate a virtual environment
#### PowerShell (Windows)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Git Bash (Windows)
```bash
python -m venv .venv
source .venv/Scripts/activate
```

#### macOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install base dependencies
```bash
pip install -r requirements.txt
```

> If `ucimlrepo` is not already installed, the fetch script will install it automatically.

### 3) Fetch and save the dataset
```bash
python src/fetch_support2.py
```
This will create:
- `data/raw/support2_features.csv`
- `data/raw/support2_targets.csv`
- `data/raw/support2.csv` (features + targets combined)
- `data/raw/support2_metadata.json`
- `data/raw/support2_variables.csv`
You can now load `data/raw/support2.csv` in notebooks and pipelines for further processing.

## Data Preprocessing
Cleans and prepares the dataset for modeling.
```bash
python src/preprocess.py
```
This script:

- Cleans column names and converts data types
- Handles missing values (median for numeric, mode for categorical)
- One-hot encodes categorical features
- Saves the processed file as:
    - data/processed/support2_clean.
    
## Summary Statistics
Generates descriptive statistics and correlation analyses for the cleaned dataset.
```bash
python src/summary_stats.py
```
Creates:
- `reports/support2_summary_statistics.csv`
- `reports/support2_cardinality.csv`
- `reports/support2_missingness_raw_vs_processed.csv`
- `reports/support2_numeric_correlations.csv`
- `visuals/correlation_heatmap.png`

## Baseline Model Prediction (Survival)
Builds and evaluates a logistic regression baseline model for mortality prediction using the SUPPORT2 dataset.
```bash
python src/Baseline_mdel_prediction_surival.py
```
This script:
- Detects the correct survival or mortality target (death, hospdead, surv6m, or surv2m)
- Converts survival values to mortality when needed (1 - survival)
- Trains a logistic regression model with balanced class weights
- Computes metrics (AUC, accuracy, log loss, and Brier score)
- Produces visualizations for ROC and calibration curves

Outputs:
- `reports/baseline_metrics.json`
- `reports/baseline_feature_importance.csv`
- `visuals/baseline_roc_curve.png`
- `visuals/baseline_reliability.png`

## Next Steps
Future extensions will involve:
- Generating synthetic datasets using differentially private mechanisms
- Comparing real vs. synthetic model performance
- Evaluating privacy–utility trade-offs
- Documenting reproducible privacy-safe data release pipelines
