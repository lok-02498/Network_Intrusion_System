"""Convert UNSW-NB15 CSV datasets into compressed Parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_csv_to_parquet(csv_path: Path, parquet_path: Path) -> Path:
    """Read a CSV file and write it to Parquet format."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import polars as pl

        pl.read_csv(csv_path).write_parquet(parquet_path)
    except ImportError:
        pd.read_csv(csv_path).to_parquet(parquet_path, index=False)
    return parquet_path


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for converting UNSW-NB15 training and testing CSVs."""
    parser = argparse.ArgumentParser(description="Convert UNSW-NB15 CSV files to Parquet.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Directory containing UNSW-NB15 CSV files.",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="UNSW_NB15_training-set.csv",
        help="Training CSV filename inside --dataset-dir.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="UNSW_NB15_testing-set.csv",
        help="Testing CSV filename inside --dataset-dir.",
    )
    args = parser.parse_args(argv)

    pairs = [
        (args.dataset_dir / args.train_csv, args.dataset_dir / "train_full.parquet"),
        (args.dataset_dir / args.test_csv, args.dataset_dir / "test_full.parquet"),
    ]
    for csv_path, parquet_path in pairs:
        out = convert_csv_to_parquet(csv_path, parquet_path)
        print(f"Converted {csv_path} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
