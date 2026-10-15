# Casino Segmentation Pipeline

An end-to-end customer segmentation and bonus-allocation workflow for casino player data. It trains a clustering model on a customer-level extract, validates it, saves a deployment artefact, then scores new data in batch and recommends a bonus amount per customer.

> **Data note:** this repository contains code only. Raw extracts, trained artefacts, figures and prediction outputs are excluded via `.gitignore` because they contain customer data.

## Scripts

Run them in this order:

| # | Script | Purpose |
|---|--------|---------|
| 1 | `Casino_Segementation_Pipeline.py` | **Training.** Loads the customer extract, runs the full segmentation workflow and saves the deployment artefacts. |
| 2 | `Python_Batch_Prediction_Test_Code.py` | **Batch scoring.** Loads the saved artefacts, scores a new extract, checks for drift and writes segment assignments with bonus recommendations. |
| 3 | `Bonus_Lookup.py` | **Lookup.** Reads the latest saved scoring output and prints segment and bonus details for specific Customer_IDs. |

## What the training pipeline does

The training script is organised in numbered steps, each printing a banner to the console:

- **Data preparation:** EDA, data-structure detection, categorical encoding, scaling, redundancy removal, and feature selection (sequential backward elimination, with business-protected features).
- **Leakage checks:** contamination and look-ahead checks, with conditional investigation of any flags.
- **Clustering tendency and distance choice:** Hopkins statistic, then distance metric selection (Euclidean, Gower, Mahalanobis).
- **Algorithm selection and fitting:** K-Means, hierarchical (Ward), DBSCAN, HDBSCAN, and a GMM candidate, with a decision matrix for choosing between them.
- **Validation:** internal validity indices, assignment-confidence calibration, bootstrap stability (ARI/Jaccard), permutation significance, multi-seed robustness.
- **Interpretation and sign-off:** cluster profiling, business alignment scorecard, KPI validation, final validation gate, optional sub-cluster discovery.
- **Deployment:** saves `deployment_artifacts.joblib`, `training_baseline.json` and `segment_personas.json` into `artifacts/`, then demonstrates drift monitoring.

## What batch scoring does

1. Loads and normalises the raw extract, then aggregates it to customer level.
2. Assigns each customer to a segment and sub-segment with a confidence value.
3. Computes a **bonus worthiness score** from four components (value, recency, responsiveness, engagement), weighted using PCA.
4. Combines three bonus rules (business rule, profitability rule, segment-aware multiplier) into a recommended bonus amount.
5. Structures eligible bonuses into Aviator flights where applicable.
6. Compares the new data to the saved training baseline to flag drift.
7. Writes `segment_assignments_<timestamp>.csv` to `artifacts/batch_predictions/`.

Customers excluded from segmentation (for example the zero-cash-funded cohort) are kept in the output with an exclusion reason and no bonus.

## Requirements

- Python 3.9+ (developed with Anaconda)
- `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `statsmodels`, `joblib`
- `hdbscan`, `umap-learn`

```powershell
conda install -c conda-forge hdbscan umap-learn
pip install statsmodels
```

## Configuration

File paths are currently hardcoded near the top of each script and point to a local Windows folder. Before running, edit these to match your machine:

- `Casino_Segementation_Pipeline.py`: `data_dir` (and the input CSV path used by `file_path`)
- `Python_Batch_Prediction_Test_Code.py`: `NEW_DATA_PATH`, `ARTIFACT_PATH`, `BASELINE_PATH`, `OUTPUT_DIR`
- `Bonus_Lookup.py`: `OUTPUT_DIR`

Other settings, such as `RANDOM_STATE`, the protected feature list and the clustering thresholds, are constants at the top of each script.

## Usage

```powershell
# 1. Train and save artefacts
python Casino_Segementation_Pipeline.py

# 2. Score a new extract
python Python_Batch_Prediction_Test_Code.py

# 3. Look up specific customers from the latest scoring run
python Bonus_Lookup.py 11111111 22222222
```

`Bonus_Lookup.py` only reads the most recent saved output, so run step 2 first and re-run it if you need fresher data.

## Expected inputs

A customer-activity CSV with columns such as `Customer_ID`, `Country_Code`, `Total_Deposit_Amount`, `Cash_Stake_Casino`, `Cash_Stake_Sports`, `GGR`, `NGR`, `Bonus_Given`, `Bet_Count`, `Game_Type`, `Supplier` and the date fields (`Reg_Date`, `Last_Login_Date`, `Activity_Date`, and so on). See `load_and_normalize()` in the batch script for the full list of expected columns.

## Repository layout

```
.
├── Casino_Segementation_Pipeline.py        # training pipeline
├── Python_Batch_Prediction_Test_Code.py    # batch scoring and bonus recommendation
├── Bonus_Lookup.py                         # per-customer lookup
├── .gitignore
└── README.md
```

Generated at runtime (not tracked): `artifacts/`, `figures/`, `exports/`.
