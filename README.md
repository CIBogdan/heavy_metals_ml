# Heavy Metals Risk Modeling — ML classification + ANN regression

A clean, reproducible Data Science repository for:
- **Risk classification** (multi-class + binary “serious vs not serious”) from notification-style records
- **Concentration regression** (`Conc`) **per metal** using **ANN** with tabular predictors

Designed to **recruiters** (modular code, CLI, tests, CI-ready) and to be usable **academically**
(reproducible pipelines, data statement, citation metadata).

---

## What this project does

### 1) Classification
Given structured notification records (e.g., metal, product category, notifier, origin + numeric indices),
the pipeline trains:
- a **multi-class** model (RandomForest by default) for `Risk_decision`
- an optional **binary** model for **serious** risk (derived from labels)

Outputs:
- metrics (accuracy / F1 / confusion matrices)
- trained models (`joblib`)
- plots (PNG)

### 2) Regression (per metal)
For each selected metal (e.g., Pb/Cd/Hg/As), the pipeline trains an **ANN** to predict:
- `Conc` (concentration)

Outputs:
- metrics (MAE / RMSE / R²)
- learning curves
- saved Keras models

---

## Repository structure

```
heavy-metals-ml/
  src/heavymetals/             # reusable package (pip install -e .)
  scripts/                     # CLI entry points (thin wrappers)
  notebooks/                   # demo notebook + original colab
  tests/                       # minimal unit tests
  docs/                        # academic notes + pipeline figure
  outputs/                     # generated (gitignored)
  data/                        # local datasets (gitignored)
```

---

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2) Put your dataset locally
Place a CSV at:
```
data/df_clean.csv
```

Your file should contain at minimum:
- `Metal`, `Product_category`, `Notified_by`, `Origin`
- `Risk_decision` (for classification)
- `Conc` (for regression)

### 3) Train classification
```bash
heavymetals-train-cls --data data/df_clean.csv --out outputs/classification
```

### 4) Train regression (per metal)
```bash
heavymetals-train-reg --data data/df_clean.csv --out outputs/regression --metals Pb Cd Hg As
```

---

## Reproducibility

- All random seeds are controlled via `config.yaml` (see `configs/`).
- Outputs are versioned by timestamped run folders.
- Models and encoders are persisted.

---

## Data statement

This repository intentionally **does not** ship the dataset.
If you use RASFF-derived or proprietary datasets, follow your license / GDPR / institutional policy.

---

## How to cite

A `CITATION.cff` is included. On GitHub, it will appear under **“Cite this repository”**.

---

## License
MIT (see `LICENSE`).
