#!/usr/bin/env python
"""
Make paper-ready figures (300 dpi) for heavy_metals_ml.

Outputs (default: outputs/paper_figures/):
- classification/
  - confusion_multiclass.png (if multiclass target available)
  - confusion_binary.png
  - roc_binary.png
  - feature_importance_rf.png
- regression/
  - ann_metrics.csv
  - mae_by_metal.png
  - pred_vs_obs_<metal>.png (if metal column exists)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    auc,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def set_mpl_defaults():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def coerce_binary(y: pd.Series) -> pd.Series | None:
    if y is None:
        return None
    yy = y.copy()

    if pd.api.types.is_numeric_dtype(yy):
        uniq = pd.Series(yy.dropna().unique())
        if set(uniq.tolist()) <= {0, 1} and len(uniq) == 2:
            return yy.astype(int)

    yy_str = yy.astype(str).str.strip().str.lower()
    mapping = {
        "serious": 1,
        "not serious": 0,
        "not_serious": 0,
        "non-serious": 0,
        "non serious": 0,
        "0": 0,
        "1": 1,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
    }
    mapped = yy_str.map(mapping)
    if mapped.notna().sum() >= int(0.8 * len(mapped.dropna())):
        return mapped.astype("Int64")

    if yy_str.str.contains("serious").any():
        return yy_str.apply(lambda s: 1 if ("serious" in s and "not" not in s) else 0).astype(
            "Int64"
        )

    return None


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, numeric_cols, categorical_cols


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return []


def run_classification(
    df: pd.DataFrame, outdir: Path, seed: int, test_size: float
) -> dict[str, str]:
    cls_dir = outdir / "classification"
    ensure_dir(cls_dir)

    y_multi_col = find_first_existing(
        df, ["Risk_decision", "risk_decision", "Decision", "decision", "Class", "class"]
    )
    y_bin_col = find_first_existing(
        df, ["Risk_binary", "risk_binary", "Serious", "serious", "Binary", "binary"]
    )
    conc_col = find_first_existing(
        df, ["Conc", "conc", "Concentration", "concentration", "Value", "value"]
    )

    exclude = set([c for c in [y_multi_col, y_bin_col, conc_col] if c is not None])
    X = df.drop(columns=list(exclude), errors="ignore").copy()
    X = X.dropna(axis=1, how="all")

    if X.shape[1] < 2:
        raise RuntimeError("Too few feature columns after excluding targets. Check dataset schema.")

    pre, _, _ = build_preprocessor(X)

    used = {}

    # Multi-class (optional)
    if y_multi_col is not None:
        y_multi = df[y_multi_col].astype(str).str.strip()
        mask = y_multi.notna() & (y_multi != "") & X.notna().any(axis=1)
        X_m, y_m = X.loc[mask], y_multi.loc[mask]
        if y_m.nunique() >= 3 and len(y_m) >= 50:
            X_train, X_test, y_train, y_test = train_test_split(
                X_m, y_m, test_size=test_size, random_state=seed, stratify=y_m
            )
            clf = RandomForestClassifier(
                n_estimators=500, random_state=seed, n_jobs=-1, class_weight="balanced"
            )
            pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            labels = sorted(y_m.unique())
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            fig, ax = plt.subplots(figsize=(7.2, 6.2))
            disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
            ax.set_title("Multi-class Confusion Matrix")
            fig.tight_layout()
            fig.savefig(cls_dir / "confusion_multiclass.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            used["multiclass_target"] = y_multi_col
        else:
            used["multiclass_target"] = f"{y_multi_col} (skipped)"
    else:
        used["multiclass_target"] = "not found"

    # Binary (required if possible)
    yb = None
    if y_bin_col is not None:
        yb = coerce_binary(df[y_bin_col])
        used["binary_target_raw"] = y_bin_col
    elif y_multi_col is not None:
        yb = coerce_binary(df[y_multi_col])
        used["binary_target_raw"] = f"derived from {y_multi_col}"
    else:
        used["binary_target_raw"] = "not found"

    if yb is None or yb.dropna().nunique() != 2:
        used["binary_target"] = "not found/convertible"
        return used

    mask = yb.notna() & X.notna().any(axis=1)
    X_b = X.loc[mask].copy()
    y_b = yb.loc[mask].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_b, y_b, test_size=test_size, random_state=seed, stratify=y_b
    )

    clf = RandomForestClassifier(
        n_estimators=800, random_state=seed, n_jobs=-1, class_weight="balanced"
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    # Confusion matrix
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not serious", "serious"])
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Binary Confusion Matrix")
    fig.tight_layout()
    fig.savefig(cls_dir / "confusion_binary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ROC
    y_score = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="RF").plot(ax=ax)
    ax.set_title(f"Binary ROC Curve (AUC = {roc_auc:.3f})")
    fig.tight_layout()
    fig.savefig(cls_dir / "roc_binary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Feature importance
    pre_fitted = pipe.named_steps["pre"]
    names = get_feature_names(pre_fitted)
    importances = pipe.named_steps["clf"].feature_importances_
    if len(names) == len(importances):
        fi = (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(25)
        )
        fig, ax = plt.subplots(figsize=(7.6, 6.4))
        ax.barh(fi["feature"][::-1], fi["importance"][::-1])
        ax.set_title("RF Feature Importance (Top 25)")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(cls_dir / "feature_importance_rf.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    used["binary_target"] = used.get("binary_target_raw", "binary")
    return used


def run_regression(df: pd.DataFrame, outdir: Path, seed: int, test_size: float) -> dict[str, str]:
    reg_dir = outdir / "regression"
    ensure_dir(reg_dir)

    conc_col = find_first_existing(
        df, ["Conc", "conc", "Concentration", "concentration", "Value", "value"]
    )
    if conc_col is None:
        raise RuntimeError("Concentration column not found (expected 'Conc' or similar).")

    metal_col = find_first_existing(df, ["Metal", "metal", "Element", "element"])

    exclude = {conc_col}
    for c in [
        find_first_existing(
            df, ["Risk_decision", "risk_decision", "Decision", "decision", "Class", "class"]
        ),
        find_first_existing(
            df, ["Risk_binary", "risk_binary", "Serious", "serious", "Binary", "binary"]
        ),
    ]:
        if c is not None:
            exclude.add(c)
    if metal_col is not None:
        exclude.add(metal_col)

    X_all = df.drop(columns=list(exclude), errors="ignore").copy().dropna(axis=1, how="all")
    y_all = pd.to_numeric(df[conc_col], errors="coerce")

    mask_all = y_all.notna() & X_all.notna().any(axis=1)
    X_all, y_all = X_all.loc[mask_all].copy(), y_all.loc[mask_all].astype(float)

    pre, _, _ = build_preprocessor(X_all)

    def fit_one(X: pd.DataFrame, y: pd.Series, label: str):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        reg = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            random_state=seed,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.15,
        )
        pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = {
            "group": label,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "MSE": float(mean_squared_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred)),
        }
        return metrics, y_test.to_numpy(), y_pred

    rows = []

    if metal_col is not None:
        for metal, dfg in df.loc[mask_all].groupby(metal_col):
            X_m = X_all.loc[dfg.index]
            y_m = y_all.loc[dfg.index]
            if len(y_m) < 30:
                continue
            m, y_true, y_pred = fit_one(X_m, y_m, str(metal))
            rows.append(m)

            fig, ax = plt.subplots(figsize=(6.0, 5.2))
            ax.scatter(y_true, y_pred, alpha=0.7)
            mn = float(np.nanmin([y_true.min(), y_pred.min()]))
            mx = float(np.nanmax([y_true.max(), y_pred.max()]))
            ax.plot([mn, mx], [mn, mx], linestyle="--")
            ax.set_title(f"Predicted vs Observed (MLP) — {metal}")
            ax.set_xlabel("Observed concentration")
            ax.set_ylabel("Predicted concentration")
            fig.tight_layout()
            fig.savefig(reg_dir / f"pred_vs_obs_{metal}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
    else:
        m, _, _ = fit_one(X_all, y_all, "ALL")
        rows.append(m)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(reg_dir / "ann_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.bar(metrics_df["group"].astype(str), metrics_df["MAE"])
    ax.set_title("Regression error by group (MAE)")
    ax.set_xlabel("Group")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(reg_dir / "mae_by_metal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "conc_col": conc_col,
        "metal_col": metal_col if metal_col is not None else "not found",
        "n_rows_reg": str(len(y_all)),
        "n_groups": str(metrics_df.shape[0]),
    }


def main():
    set_mpl_defaults()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to df_clean.csv (local).")
    ap.add_argument("--out", default="outputs/paper_figures", help="Output directory.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    outdir = Path(args.out)
    ensure_dir(outdir)

    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded: {data_path} | shape={df.shape}")
    print(f"[INFO] Output dir: {outdir.resolve()}")

    used_cls = run_classification(df, outdir=outdir, seed=args.seed, test_size=args.test_size)
    print("[INFO] Classification:", used_cls)

    used_reg = run_regression(df, outdir=outdir, seed=args.seed, test_size=args.test_size)
    print("[INFO] Regression:", used_reg)

    print("[DONE] Paper-ready figures generated.")


if __name__ == "__main__":
    main()
