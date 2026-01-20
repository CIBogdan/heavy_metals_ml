from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read dataset with safe defaults."""
    p = Path(path)
    return pd.read_csv(p)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
