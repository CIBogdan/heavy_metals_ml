from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import AppConfig
from ..io import read_csv


def run_dir(out_root: str | Path) -> Path:
    out_root = Path(out_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_root / stamp


def load_config_and_data(
    config_path: str | Path, data_path: str | Path
) -> tuple[AppConfig, pd.DataFrame]:
    cfg = AppConfig.from_yaml(config_path)
    df = read_csv(data_path)
    return cfg, df
