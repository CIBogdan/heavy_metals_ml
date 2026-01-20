from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._shared import load_config_and_data, run_dir
from ..plots import plot_learning_curve
from ..preprocessing import FeatureSpec, infer_numeric_columns
from ..regression_ann import save_ann, train_ann_regression
from joblib import dump


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV dataset")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--out", default="outputs/regression")
    p.add_argument("--metals", nargs="*", default=[], help="Metals to train (e.g., Pb Cd Hg As)")
    args = p.parse_args()

    cfg, df = load_config_and_data(args.config, args.data)
    out_root = run_dir(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    metal_col = cfg.columns.metal_column
    metals = args.metals or sorted(df[metal_col].dropna().unique().tolist())

    exclude = [cfg.columns.target_multiclass, cfg.columns.target_regression]
    numeric = cfg.columns.numeric or infer_numeric_columns(df, exclude=exclude)
    spec = FeatureSpec(categorical=cfg.columns.categorical, numeric=numeric)

    ann_params = {
        "epochs": cfg.ann.epochs,
        "batch_size": cfg.ann.batch_size,
        "learning_rate": cfg.ann.learning_rate,
        "hidden_units": cfg.ann.hidden_units or [128, 64, 32],
        "dropout": cfg.ann.dropout,
        "patience": cfg.ann.patience,
    }

    for m in metals:
        df_m = df[df[metal_col] == m].copy()
        df_m = df_m.dropna(subset=[cfg.columns.target_regression])
        if len(df_m) < 25:
            print(f"Skip {m}: too few rows ({len(df_m)})")
            continue

        out = out_root / f"metal_{m}"
        out.mkdir(parents=True, exist_ok=True)

        prep_pipe, model, res, hist = train_ann_regression(
            df=df_m,
            target=cfg.columns.target_regression,
            spec=spec,
            seed=cfg.seed,
            ann_params=ann_params,
        )

        # save artifacts
        dump(prep_pipe, out / "preprocessor.joblib")
        save_ann(model, out)
        (out / "metrics.json").write_text(json.dumps(res.__dict__, indent=2), encoding="utf-8")
        plot_learning_curve(hist, out / "learning_curve.png", title=f"Learning curve — {m}")

        print(f"Done {m}: MAE={res.mae:.4g}, RMSE={res.rmse:.4g}, R2={res.r2:.4g}")

    print(f"Saved outputs to: {out_root}")


if __name__ == "__main__":
    main()
