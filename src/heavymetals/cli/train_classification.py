from __future__ import annotations

import argparse
import json

from ..classification import save_pipeline, train_random_forest_multiclass
from ..plots import plot_confusion_matrix
from ..preprocessing import FeatureSpec, infer_numeric_columns
from ._shared import load_config_and_data, run_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV dataset")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--out", default="outputs/classification")
    args = p.parse_args()

    cfg, df = load_config_and_data(args.config, args.data)
    out = run_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # features
    exclude = [cfg.columns.target_multiclass, cfg.columns.target_regression]
    numeric = cfg.columns.numeric or infer_numeric_columns(df, exclude=exclude)
    spec = FeatureSpec(categorical=cfg.columns.categorical, numeric=numeric)

    rf_params = {
        "n_estimators": cfg.rf.n_estimators,
        "max_depth": cfg.rf.max_depth,
        "class_weight": cfg.rf.class_weight,
        "n_jobs": -1,
    }

    pipe, res = train_random_forest_multiclass(
        df=df,
        target=cfg.columns.target_multiclass,
        spec=spec,
        seed=cfg.seed,
        rf_params=rf_params,
    )

    # save
    save_pipeline(pipe, str(out / "model_multiclass.joblib"))
    (out / "metrics_multiclass.json").write_text(json.dumps(res.report, indent=2), encoding="utf-8")

    labels = list(res.report.keys())
    labels = [x for x in labels if x not in ("accuracy", "macro avg", "weighted avg")]
    plot_confusion_matrix(
        res.confusion_matrix, labels, out / "cm_multiclass.png", "Confusion Matrix — Multi-class"
    )

    print(f"Saved outputs to: {out}")


if __name__ == "__main__":
    main()
