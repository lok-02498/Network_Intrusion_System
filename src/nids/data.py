"""Dataset loading, CSV-to-Parquet conversion, and synthetic demo data."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a Parquet or CSV file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported dataset format '{suffix}'. Use .parquet or .csv.")


def convert_csv_to_parquet(csv_path: str | Path, parquet_path: str | Path) -> Path:
    """Convert a CSV dataset to Parquet format."""
    src = Path(csv_path)
    dst = Path(parquet_path)
    if not src.exists():
        raise FileNotFoundError(f"Source CSV not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    df.to_parquet(dst, index=False)
    return dst


def split_features_and_target(
    df: pd.DataFrame,
    target_column: str = "label",
    drop_columns: Iterable[str] = ("id", "attack_cat", "traffic_type"),
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y, dropping metadata columns."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    cols_to_drop = [col for col in drop_columns if col in df.columns]
    cols_to_drop.append(target_column)
    X = df.drop(columns=cols_to_drop)
    y = df[target_column].astype(int)
    return X, y


def generate_synthetic_traffic(
    n_samples: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic UNSW-NB15-style dataset for offline demos and tests."""
    rng = np.random.default_rng(random_state)
    is_attack = rng.binomial(1, 0.35, size=n_samples)

    proto = rng.choice(["tcp", "udp", "icmp"], size=n_samples, p=[0.65, 0.25, 0.10])
    service = rng.choice(
        ["http", "dns", "ftp", "ssh", "-"],
        size=n_samples,
        p=[0.35, 0.25, 0.10, 0.10, 0.20],
    )
    state = rng.choice(["FIN", "CON", "INT", "REQ"], size=n_samples, p=[0.45, 0.25, 0.20, 0.10])

    dur = np.where(is_attack == 1, rng.exponential(2.5, n_samples), rng.exponential(0.6, n_samples))
    spkts = np.where(is_attack == 1, rng.integers(15, 150, n_samples), rng.integers(1, 30, n_samples))
    dpkts = np.where(is_attack == 1, rng.integers(0, 60, n_samples), rng.integers(1, 35, n_samples))
    sbytes = spkts * rng.integers(60, 400, n_samples)
    dbytes = dpkts * rng.integers(40, 350, n_samples)
    rate = (spkts + dpkts) / np.maximum(dur, 0.01)
    sttl = np.where(is_attack == 1, rng.choice([64, 254], n_samples), rng.choice([31, 62, 64], n_samples))
    dttl = np.where(is_attack == 1, rng.choice([0, 29, 252], n_samples), rng.choice([29, 60, 252], n_samples))

    return pd.DataFrame(
        {
            "id": np.arange(1, n_samples + 1),
            "dur": np.round(dur, 6),
            "proto": proto,
            "service": service,
            "state": state,
            "spkts": spkts,
            "dpkts": dpkts,
            "sbytes": sbytes,
            "dbytes": dbytes,
            "rate": np.round(rate, 4),
            "sttl": sttl,
            "dttl": dttl,
            "label": is_attack,
        }
    )
