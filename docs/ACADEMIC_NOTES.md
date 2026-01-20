# Academic notes (methods & reporting)

This repository is structured to support paper-style reproducibility.

## Suggested Methods text (adapt)
- Data cleaning: missing-value handling, outlier treatment (per metal), and encoding strategy
- Classification: train/test split with stratification; RF hyperparameters; metrics (macro-F1)
- Regression: metal-specific ANN; early stopping; MAE/RMSE/R²; saved model artifacts

## Reporting checklist
- Random seed(s) documented (config.yaml)
- Exact package versions (requirements / environment.yml)
- Train/test split strategy
- Class imbalance handling (RF class_weight)
- Uncertainty: confidence intervals via bootstrap (optional extension)
