from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import FeatureSpec, build_preprocessor, make_xy


@dataclass(frozen=True)
class RegressionResults:
    mae: float
    rmse: float
    r2: float


def build_ann(input_dim: int, hidden_units: list[int], dropout: float, lr: float) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_ann_regression(
    df: pd.DataFrame,
    target: str,
    spec: FeatureSpec,
    seed: int,
    ann_params: dict,
) -> tuple[Pipeline, tf.keras.Model, RegressionResults, tf.keras.callbacks.History]:
    X, y = make_xy(df, target, spec)
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    prep = build_preprocessor(spec)
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)

    tf.keras.utils.set_random_seed(seed)

    model = build_ann(
        input_dim=X_train_t.shape[1],
        hidden_units=list(ann_params.get("hidden_units", [128, 64, 32])),
        dropout=float(ann_params.get("dropout", 0.15)),
        lr=float(ann_params.get("learning_rate", 1e-3)),
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(ann_params.get("patience", 12)),
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_t,
        y_train.to_numpy(),
        validation_split=0.2,
        epochs=int(ann_params.get("epochs", 80)),
        batch_size=int(ann_params.get("batch_size", 64)),
        verbose=0,
        callbacks=callbacks,
    )

    y_pred = model.predict(X_test_t, verbose=0).reshape(-1)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(mean_squared_error(y_test, y_pred, squared=False))
    r2 = float(r2_score(y_test, y_pred))

    # pack preprocessor in sklearn Pipeline for persistence of transforms
    pipe = Pipeline(steps=[("prep", prep)])
    return pipe, model, RegressionResults(mae=mae, rmse=rmse, r2=r2), history


def save_ann(model: tf.keras.Model, outdir: str | Path) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.keras")
