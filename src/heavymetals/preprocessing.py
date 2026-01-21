from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureSpec:
    categorical: list[str]
    numeric: list[str]


def infer_numeric_columns(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """Preprocessor: impute + one-hot for categoricals; impute + scale for numeric."""
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, spec.categorical),
            ("num", num_pipe, spec.numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_xy(
    df: pd.DataFrame,
    target: str,
    spec: FeatureSpec,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[spec.categorical + spec.numeric].copy()
    y = df[target].copy()
    return X, y
