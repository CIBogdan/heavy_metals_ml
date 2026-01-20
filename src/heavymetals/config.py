from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class ColumnConfig:
    target_multiclass: str
    target_regression: str
    metal_column: str
    categorical: List[str]
    numeric: List[str]


@dataclass(frozen=True)
class RFConfig:
    n_estimators: int = 600
    max_depth: Optional[int] = None
    class_weight: Optional[str] = "balanced_subsample"


@dataclass(frozen=True)
class ANNConfig:
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_units: List[int] = None  # type: ignore
    dropout: float = 0.15
    patience: int = 12


@dataclass(frozen=True)
class AppConfig:
    seed: int
    columns: ColumnConfig
    rf: RFConfig
    ann: ANNConfig

    @staticmethod
    def from_yaml(path: str | Path) -> "AppConfig":
        p = Path(path)
        data: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))

        cols = data["columns"]
        col_cfg = ColumnConfig(
            target_multiclass=cols["target_multiclass"],
            target_regression=cols["target_regression"],
            metal_column=cols["metal_column"],
            categorical=list(cols.get("categorical", [])),
            numeric=list(cols.get("numeric", [])),
        )

        cls = data.get("models", {}).get("classification", {})
        rf_cfg = RFConfig(**cls.get("random_forest", {}))

        reg = data.get("models", {}).get("regression", {}).get("ann", {})
        ann_cfg = ANNConfig(
            epochs=int(reg.get("epochs", 80)),
            batch_size=int(reg.get("batch_size", 64)),
            learning_rate=float(reg.get("learning_rate", 1e-3)),
            hidden_units=list(reg.get("hidden_units", [128, 64, 32])),
            dropout=float(reg.get("dropout", 0.15)),
            patience=int(reg.get("patience", 12)),
        )

        return AppConfig(seed=int(data.get("seed", 42)), columns=col_cfg, rf=rf_cfg, ann=ann_cfg)
