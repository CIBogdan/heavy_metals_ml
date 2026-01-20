from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import FeatureSpec, build_preprocessor, make_xy


@dataclass(frozen=True)
class ClassificationResults:
    report: Dict
    confusion_matrix: np.ndarray
    macro_f1: float


def train_random_forest_multiclass(
    df: pd.DataFrame,
    target: str,
    spec: FeatureSpec,
    seed: int,
    rf_params: Dict,
) -> Tuple[Pipeline, ClassificationResults]:
    X, y = make_xy(df, target, spec)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y if y.nunique() > 1 else None
    )

    model = RandomForestClassifier(random_state=seed, **rf_params)

    pipe = Pipeline(steps=[("prep", build_preprocessor(spec)), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return pipe, ClassificationResults(report=rep, confusion_matrix=cm, macro_f1=float(macro))


def save_pipeline(pipe: Pipeline, path: str) -> None:
    dump(pipe, path)
