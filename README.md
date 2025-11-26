# DSC_180A Section_B15 Quater_1 Project  
**Title:** Differentially Private Telemetry Data Analysis – Non-Private Exploration 

**Team:** Dhruv Patel, Reva Agrawal, and Jordan Lambino 

**Mentor:** Prof. Yu-Xiang Wang  

## Overview
This project investigates the feasibility and accuracy of generating synthetic datasets that preserve privacy while maintaining statistical utility, using a synthetic telemetry event log dataset as a case study. The dataset contains user-level event logs for multiple product types, with attributes for product type, event type, timestamp, and user ID.

Our primary goal is to explore how well differentially private mechanisms and synthetic data generation methods can reproduce key analytical quantities — such as error counts and error rates per product — without compromising user-level privacy. By comparing real and synthetic (or DP-noisy) results on metrics like per-product error rate and average error rate across products, we aim to evaluate the trade-off between privacy protection and data utility.

## 📂 Repository Structure
- `data/` — Raw, processed, and (future) synthetic versions of the telemetry dataset.  
  - `data/raw/` — Original `synthetic_telemetry_data.csv` provided by the instructor.  
  - `data/processed/` — Cleaned version with parsed timestamps and derived features.  
  - `data/synthetic/` — To be used for storing differentially private synthetic datasets.  
- `notebooks/` — Jupyter notebooks for exploratory analysis and experimentation (e.g., `README.ipynb`).  
- `src/` — Python scripts for preprocessing, summary statistics, and baseline metrics.  
- `reports/` — CSV and JSON outputs containing summary statistics and baseline metrics.  
- `visuals/` — Plots and figures for telemetry event counts and error rates.  
- `scripts/` — (Optional) Shell scripts for running the pipeline end-to-end. 

## Dependencies
Install dependencies with:
```bash
pip install -r requirements.txt
```

## 🗃️ Data (Synthetic Telemetry)
The dataset contains synthetic telemetry event logs with the following attributes:
Product Type: Categorical attribute (e.g. A, B, C, D, E, F, Others).
Event Type: Event kind — open, close, save, reset, error.
Time of Event: Timestamp between May 1, 2024 and July 31, 2024.
User ID: Anonymized user identifier.

## Data Preprocessing
Cleans and prepares the dataset for modeling.
```bash
python src/preprocess.py
```
This script:

- Outputs: data/processed/telemetry_clean.csv

- Columns added:
- - event_date: Date of event
- - event_hour: Hour of event (0–23)
- - event_dow: Day of week (0=Mon–6=Sun)
- - is_weekend: Boolean (1 if weekend)
    
## Summary Statistics
### 📊 Statistical Formulas
The following equations are implemented to calculate core telemetry metrics:
1. Error Count per Product Type
$$
\text{ErrorCount}_P = \sum_i 1\{\text{EventType}_i = \text{error},\ \text{ProductType}_i = P\}
$$
2. Total Event Count per Product Type
$$
\text{TotalCount}_P = \sum_i 1\{\text{ProductType}_i = P\}
$$
3. Error Rate
$$
\text{ErrorRate}_P = \frac{\text{ErrorCount}_P}{\text{TotalCount}_P}
$$
4. Z-Score
$$
Z_P = \frac{\text{ErrorRate}_P - \overline{\text{ErrorRate}}}{\text{SD}(\text{ErrorRate})}
$$

Product types with positive z-scores perform worse than average, meaning they have higher error rates relative to other product types.

Generates descriptive statistics and correlation analyses for the cleaned dataset.
```bash
python src/summary_stats.py
```
Creates:
- Generates frequency tables:
- - Events by product type
- - Events by event type
- - Events per day
- - Unique users per product type

Calculates error rates and z-scores using:
- Saves results in reports/ and plots in visuals/.
- Output reports:
- - telemetry_summary_statistics.csv
- - telemetry_event_counts_by_product.csv
- - telemetry_event_counts_by_type.csv
- - telemetry_events_per_day.csv
- - telemetry_unique_users_per_product.csv
- - telemetry_error_rates_zscores.csv

- Output visuals:
- - event_type_distribution.png
- - product_type_distribution.png
- - events_over_time.png
- - error_rate_by_product.png
- - error_rate_zscore_by_product.png
