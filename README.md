# DSC_180A Section_B15 Quarter_1 Project  
**Title:** Balancing Privacy and Utility: Differentially Private Synthetic Telemetry Data Generation

**Team & Mentor:** 
| Name             | Role    | GitHub       |
|------------------|---------|--------------|
| Dhruv Patel      | Student | [@PDhruv09](https://github.com/PDhruv09)       |
| Reva Agrawal     | Student | [@agrawalreva](https://github.com/agrawalreva) |
| Jordan Lambino   | Student | [@jordanlambino](https://github.com/jordanlambino)  |
| Yu-Xiang Wang    | Advisor | [@yuxiangw](https://github.com/yuxiangw)|

---

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Data](#️-data-synthetic-telemetry)
- [Usage](#-usage)
  - [Running Python Scripts](#running-python-scripts)
  - [Using Jupyter Notebooks](#using-jupyter-notebooks)
- [Statistical Methods](#-statistical-methods)
- [Evaluation Metrics](#-evaluation-metrics)
- [Results](#-results)

---

## Overview
This project investigates the feasibility and accuracy of generating synthetic datasets that preserve privacy while maintaining statistical utility, using a synthetic telemetry event log dataset as a case study. The dataset contains user-level event logs for multiple product types, with attributes for product type, event type, timestamp, and user ID.

Our primary goal is to explore how well differentially private mechanisms (both **Gaussian** and **Laplace**) can reproduce key analytical quantities such as error counts and error rates per product without compromising user-level privacy. By comparing real and DP-noisy results on metrics like per-product error rate and average error rate across products, we evaluate the trade-off between privacy protection and data utility.

### Key Research Questions
1. How accurately can differentially private mechanisms preserve statistical utility?
2. What is the optimal balance between privacy budget (ε) and data utility?
3. Which DP mechanism (Gaussian vs. Laplace) performs better for telemetry data?
4. How do L∞ error and IOU (Intersection over Union) metrics vary across different privacy budgets?

For a detailed discussion of these questions and our methodology, please refer to the full report [here](https://drive.google.com/file/d/1AS2MzcOv4TvUNJKeGbtr0ebVXijjNVdq/view?usp=sharing)
---

## Repository Structure
```
DSC_180A-Section_B15-Quarter_1-Project/
├── data/
│   ├── raw/                                    # Original synthetic telemetry data
│   │   └── synthetic_telemetry_data.csv
│   ├── processed/                              # Cleaned and preprocessed data
│   │   └── telemetry_clean.csv
│   └── synthetic/                              # Differentially private synthetic datasets
│
├── notebooks/
│   ├── telemetry_exploration.ipynb             # Initial EDA and baseline analysis
│   ├── dp_Gaussian_mechanism_and_eval.ipynb    # Gaussian DP implementation & evaluation
│   └── dp_Laplace_mechanism_and_eval.ipynb     # Laplace DP implementation & evaluation
│
├── src/
│   ├── preprocess.py                           # Data preprocessing script
│   ├── summary_stats.py                        # Summary statistics generation
│   ├── dp_Gaussian_mechanism.py                # Gaussian DP mechanism implementation
│   ├── dp_Laplace_mechanism.py                 # Laplace DP mechanism implementation
│   ├── dp_eval_gaussian.py                     # Gaussian DP evaluation (100 runs)
│   └── dp_eval_laplace.py                      # Laplace DP evaluation (100 runs)
│   └── eval_comparison.py                      # Comparison of evluation run by both mechanism and thier plots
│
├── reports/                                    # Generated CSV/JSON reports
│   ├── telemetry_summary_statistics.csv
│   ├── telemetry_event_counts_by_product.csv
│   ├── telemetry_event_counts_by_type.csv
│   ├── telemetry_events_per_day.csv
│   ├── telemetry_unique_users_per_product.csv
│   ├── telemetry_error_rates_zscores.csv
│   ├── dp_user_level_counts_and_zscores.csv         # Gaussian DP results
│   ├── dp_user_level_counts_and_zscores_laplace.csv # Laplace DP results
│   ├── dp_eval_runs.csv                             # Gaussian evaluation runs
│   ├── dp_eval_runs_laplace.csv                     # Laplace evaluation runs
│   ├── dp_eval_summary.json                         # Gaussian evaluation summary
│   └── dp_eval_summary_laplace.json                 # Laplace evaluation summary
│
├── visuals/                                    # Generated plots and figures
│   ├── event_type_distribution.png
│   ├── product_type_distribution.png
│   ├── events_over_time.png
│   ├── error_rate_by_product.png
│   └── error_rate_zscore_by_product.png
│
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Setup
1. **Clone the repository:**
```bash
git clone https://github.com/PDhruv09/DSC_180A-Section_B15-Quarter_1-Project.git
cd DSC_180A-Section_B15-Quarter_1-Project
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Required Packages
- `pandas` — Data manipulation and analysis
- `numpy` — Numerical computing
- `matplotlib` — Plotting and visualization
- `scipy` — Scientific computing (for z-scores and noise generation)
- `jupyter` — Interactive notebooks

---

## Getting Started

### Pipeline
Run the complete analysis pipeline with:

```bash
# 1. Preprocess the data
python src/preprocess.py

# 2. Generate summary statistics and visualizations
python src/summary_stats.py

# 3. Run Gaussian DP mechanism
python src/dp_Gaussian_mechanism.py

# 4. Run Laplace DP mechanism
python src/dp_Laplace_mechanism.py

# 5. Evaluate Gaussian DP (100 runs with different seeds)
python src/dp_eval_gaussian.py

# 6. Evaluate Laplace DP (100 runs with different seeds)
python src/dp_eval_laplace.py
```

**Alternatively**, explore the analysis interactively using Jupyter notebooks (see [Using Jupyter Notebooks](#using-jupyter-notebooks)).

---

## Data (Synthetic Telemetry)
The dataset contains synthetic telemetry event logs with the following attributes:

| Attribute       | Description                                          | Type        |
|-----------------|------------------------------------------------------|-------------|
| **Product Type**| Categorical product identifier (A, B, C, D, E, F, Others) | Categorical |
| **Event Type**  | Event kind: `open`, `close`, `save`, `reset`, `error` | Categorical |
| **Time of Event** | Timestamp between May 1, 2024 and July 31, 2024   | Datetime    |
| **User ID**     | Anonymized user identifier                           | String      |

### Data Characteristics
- **Date Range:** May 1, 2024 – July 31, 2024 (92 days)
- **Product Types:** 7 categories (A, B, C, D, E, F, Others)
- **Event Types:** 5 categories (open, close, save, reset, error)
- **Privacy Unit:** Individual users (user-level differential privacy)

---

## Usage

### Running Python Scripts

#### 1. Data Preprocessing (`preprocess.py`)
Clean and prepare the dataset for analysis:
```bash
python src/preprocess.py
```

**What it does:**
- Loads raw telemetry data from `data/raw/synthetic_telemetry_data.csv`
- Parses timestamps and validates data entries
- Extracts temporal features for time-based analysis
- Saves processed data to `data/processed/telemetry_clean.csv`

**Generated Features:**
- `event_date`: Date of event (YYYY-MM-DD)
- `event_hour`: Hour of event (0–23)
- `event_dow`: Day of week (0=Monday, 6=Sunday)
- `is_weekend`: Boolean flag (1 if Saturday/Sunday, 0 otherwise)

---

#### 2. Summary Statistics (`summary_stats.py`)
Generate descriptive statistics and visualizations:
```bash
python src/summary_stats.py
```

**What it does:**
- Calculates frequency distributions by product and event type
- Computes error rates and z-scores per product
- Generates time-series analysis of events
- Creates visualizations saved to `visuals/`

**Output Files:**
- **Reports:**
  - `telemetry_summary_statistics.csv` — Overall dataset statistics
  - `telemetry_event_counts_by_product.csv` — Events per product
  - `telemetry_event_counts_by_type.csv` — Events per type
  - `telemetry_events_per_day.csv` — Daily event counts
  - `telemetry_unique_users_per_product.csv` — User counts per product
  - `telemetry_error_rates_zscores.csv` — Error rates with z-scores

- **Visualizations:**
  - `event_type_distribution.png`
  - `product_type_distribution.png`
  - `events_over_time.png`
  - `error_rate_by_product.png`
  - `error_rate_zscore_by_product.png`

---

#### 3. Gaussian DP Mechanism (`dp_Gaussian_mechanism.py`)
Apply Gaussian noise for (ε, δ)-differential privacy:
```bash
python src/dp_Gaussian_mechanism.py
```

**What it does:**
- Loads cleaned telemetry data
- Builds user-level "primary product" assignments (one product per user)
- Computes true user-level counts: `U_total` and `U_error` per product
- Adds Gaussian noise with scale σ = (Δ₂/ε) × √(2ln(1.25/δ))
- Recomputes DP error rates and z-scores
- Saves results to `reports/dp_user_level_counts_and_zscores.csv`

**Default Parameters:**
- ε (epsilon) = 2.0
- δ (delta) = 1e-6
- L₂ sensitivity = √2 (user affects one product in both U_total and U_error)

---

#### 4. Laplace DP Mechanism (`dp_Laplace_mechanism.py`)
Apply Laplace noise for pure ε-differential privacy:
```bash
python src/dp_Laplace_mechanism.py
```

**What it does:**
- Loads cleaned telemetry data
- Builds user-level "primary product" assignments
- Computes true user-level counts: `U_total` and `U_error` per product
- Adds Laplace noise with scale b = Δ₁/ε
- Recomputes DP error rates and z-scores
- Saves results to `reports/dp_user_level_counts_and_zscores_laplace.csv`

**Default Parameters:**
- ε (epsilon) = 2.0
- L₁ sensitivity = 2 (user affects one product: +1 in U_total, +1 in U_error)

---

#### 5. Gaussian DP Evaluation (`dp_eval_gaussian.py`)
Run comprehensive evaluation of Gaussian mechanism:
```bash
python src/dp_eval_gaussian.py
```

**What it does:**
- Runs Gaussian DP mechanism 100 times with different random seeds
- Computes L∞ error on z-scores for each run
- Computes IOU (Intersection over Union) on "top sets" (products with z > 0)
- Generates quantile summaries (min, 5%, 50%, 95%, max)

**Output Files:**
- `reports/dp_eval_runs.csv` — Per-run metrics
- `reports/dp_eval_summary.json` — Statistical summary

---

#### 6. Laplace DP Evaluation (`dp_eval_laplace.py`)
Run comprehensive evaluation of Laplace mechanism:
```bash
python src/dp_eval_laplace.py
```

**What it does:**
- Runs Laplace DP mechanism 100 times with different random seeds
- Computes L∞ error on z-scores for each run
- Computes IOU on "top sets" for each run
- Generates quantile summaries

**Output Files:**
- `reports/dp_eval_runs_laplace.csv` — Per-run metrics
- `reports/dp_eval_summary_laplace.json` — Statistical summary

---

### Using Jupyter Notebooks

For **interactive exploration and experimentation**, launch Jupyter:

```bash
jupyter notebook
```

Then navigate to the `notebooks/` directory and open:

#### 1. **`telemetry_exploration.ipynb`**
**Purpose:** Initial exploratory data analysis and baseline metrics

**Contents:**
- Data loading and structure inspection
- Descriptive statistics and distributions
- Event patterns over time
- User behavior analysis
- Baseline error rate calculations
- Initial visualization of product performance

---

#### 2. **`dp_Gaussian_mechanism_and_eval.ipynb`**
**Purpose:** Interactive Gaussian DP mechanism implementation and evaluation

**Contents:**
- User-level primary product assignment
- True count computation (U_total, U_error)
- Gaussian noise addition with (ε, δ)-DP guarantees
- Single-run DP z-score visualization
- Multi-run evaluation (100 iterations)
- L∞ error and IOU metric computation
- Privacy-utility trade-off analysis
- Quantile summaries and distribution plots

**Key Features:**
- Step-by-step explanation of Gaussian mechanism
- Sensitivity analysis (L₂ norm = √2)
- Comparison of true vs. DP z-scores
- Interactive parameter tuning

---

#### 3. **`dp_Laplace_mechanism_and_eval.ipynb`**
**Purpose:** Interactive Laplace DP mechanism implementation and evaluation

**Contents:**
- User-level primary product assignment
- True count computation (U_total, U_error)
- Laplace noise addition with pure ε-DP
- Single-run DP z-score visualization
- Multi-run evaluation (100 iterations)
- L∞ error and IOU metric computation
- Privacy-utility trade-off analysis
- Quantile summaries and distribution plots

**Key Features:**
- Step-by-step explanation of Laplace mechanism
- Sensitivity analysis (L₁ norm = 2)
- Comparison of true vs. DP z-scores
- Pure ε-DP (no δ parameter needed)
- Side-by-side comparison with Gaussian mechanism

---

## Statistical Methods

### Core Metrics

#### 1. **Error Count per Product Type**
```
ErrorCount_P = Σ 1{EventType_i = "error" AND ProductType_i = P}
```
Total number of error events for product type P.

#### 2. **Total Event Count per Product Type**
```
TotalCount_P = Σ 1{ProductType_i = P}
```
Total number of all events for product type P.

#### 3. **Error Rate**
```
ErrorRate_P = ErrorCount_P / TotalCount_P
```
Proportion of events that are errors for product type P.

#### 4. **Z-Score (Standardized Error Rate)**
```
Z_P = (ErrorRate_P - μ) / σ
```
Where:
- μ = mean error rate across all products
- σ = standard deviation of error rates

**Interpretation:** Products with **positive z-scores** have higher-than-average error rates and may indicate quality issues.

---

### User-Level DP Formulation

To achieve **user-level differential privacy**, we:

1. **Assign each user a "primary product"** based on where they have the most events
2. **Compute user-level counts:**
   - `U_total_P` = # of users whose primary product is P
   - `U_error_P` = # of those users who had ≥1 error on P

3. **Apply DP noise** to these user-level counts (not raw event counts)

**Why user-level?**
- Adding/removing one user changes counts for only ONE product
- Clear, bounded sensitivity: L₂ = √2 (Gaussian) or L₁ = 2 (Laplace)
- Stronger privacy guarantee than event-level DP

---

### Differential Privacy Mechanisms

#### **Gaussian Mechanism (ε, δ)-DP**
Adds noise drawn from Gaussian distribution:
```
DP_Value = TrueValue + N(0, σ²)
```
where:
```
σ = (Δ₂ / ε) × √(2 × ln(1.25 / δ))
```

**Parameters:**
- Δ₂ = √2 (L₂ sensitivity for user-level counts)
- ε = privacy budget (default: 2.0)
- δ = failure probability (default: 1e-6)

**Properties:**
- Provides approximate DP: (ε, δ)-DP
- Smaller noise than Laplace for same ε when δ > 0
- Suitable when small failure probability is acceptable

---

#### **Laplace Mechanism (pure ε-DP)**
Adds noise drawn from Laplace distribution:
```
DP_Value = TrueValue + Lap(b)
```
where:
```
b = Δ₁ / ε
```

**Parameters:**
- Δ₁ = 2 (L₁ sensitivity for user-level counts)
- ε = privacy budget (default: 2.0)

**Properties:**
- Provides pure ε-DP (no δ)
- Heavier tails than Gaussian (more extreme outliers possible)
- Stronger privacy guarantee (no failure probability)

---

## Evaluation Metrics

### 1. **L∞ Error on Z-Scores**
```
L∞ = max_P |Z_true[P] - Z_dp[P]|
```
Measures the **worst-case difference** between true and DP z-scores across all products.

**Interpretation:**
- Lower is better (indicates utility preservation)
- Critical for identifying whether DP distorts the relative ranking of products

---

### 2. **IOU (Intersection over Union) on Top Sets**
```
IOU = |TopSet_true ∩ TopSet_dp| / |TopSet_true ∪ TopSet_dp|
```
where `TopSet = {P : Z_P > 0}` (products with above-average error rates)

**Interpretation:**
- Ranges from 0 (no overlap) to 1 (perfect match)
- Measures whether DP preserves the **set of problematic products**
- High IOU means DP correctly identifies which products need attention

---

### 3. **Quantile Summaries**
For both L∞ and IOU, we compute:
- **Min, Max:** Range of values across 100 runs
- **5%, 50%, 95% percentiles:** Distribution of metric values

This characterizes the **variability** introduced by randomness in the DP mechanism.

---

## Results

### What We Analyze

1. **Privacy-Utility Trade-off:**
   - How does utility (L∞, IOU) degrade as ε decreases?
   - At what ε value does utility become unacceptable?

2. **Mechanism Comparison:**
   - Does Gaussian or Laplace perform better for this dataset?
   - How do noise distributions affect z-score stability?

3. **Variability Across Runs:**
   - How much do L∞ and IOU vary due to randomness?
   - Are results consistent or highly variable?

4. **Top Set Preservation:**
   - Can DP reliably identify products with high error rates?
   - How often does IOU = 1.0 (perfect preservation)?

### Comparisons
- **Metrics Compared:** Linf and IOU values across 100 independent runs
- **Side-by-Side Analysis:** 
  - One plot overlays Laplace vs. Gaussian Linf results
  - Another plot overlays Laplace vs. Gaussian IOU results
- **Purpose:** To evaluate differences in robustness and accuracy between Laplace and Gaussian mechanisms
- **Interpretation:** Visualizations highlight distributional shifts, variance, and relative performance across mechanisms

### Output Locations
- **Quantitative Results:** `reports/dp_eval_summary_gaussian.json` and `reports/dp_eval_summary_laplace.json`
- **Post-Run Data:** `reports/dp_eval_runs_gaussian.csv` and `reports/dp_eval_runs_laplace.csv`
- **Visualizations:** Generated in notebooks with distribution plots, boxplots, and comparison charts

---

## Future Work

1. **Synthetic Data Generation:**
   - Use DP counts to generate fully synthetic telemetry datasets
   - Validate synthetic data utility for downstream ML tasks

2. **Advanced DP Mechanisms:**
   - Implement private feature selection
   - Test adaptive composition for multiple queries
   - Explore privacy amplification via sampling

3. **Multi-Query Scenarios:**
   - Sequential composition over time windows
   - Privacy budget allocation strategies

4. **Production Deployment:**
   - Scale to production telemetry systems
   - Develop automated DP reporting pipelines
   - Create privacy-preserving dashboards

---

## Contact

For questions or feedback:
- **Dhruv Patel** - [dhp005@ucsd.edu]
- **Reva Agrawal** - [ragrawal@ucsd.edu]
- **Jordan Lambino** - [jlambino@ucsd.edu]

---

## Acknowledgments

- **Advisor:** Yu-Xiang Wang, UC San Diego
- **Course:** DSC 180A (Data Science Capstone), Fall 2024
- **Dataset:** Synthetic telemetry data provided by course instructor
- **References:** Gaussian and Laplace mechanisms from differential privacy literature

---

**Last Updated:** December 2025
