"""Unit tests for data utilities, preprocessing, and CLI."""

import pytest

from nids.cli import main as cli_main
from nids.data import convert_csv_to_parquet, generate_synthetic_traffic, load_dataset, split_features_and_target


def test_split_features_and_target():
    """Ensure metadata and target columns are removed from X."""
    df = generate_synthetic_traffic(n_samples=20, random_state=1)
    df["attack_cat"] = "Normal"
    X, y = split_features_and_target(df)

    assert "label" not in X.columns
    assert "id" not in X.columns
    assert "attack_cat" not in X.columns
    assert len(X) == len(y) == 20


def test_csv_to_parquet_conversion(tmp_path):
    """Verify CSV-to-Parquet conversion round-trips cleanly."""
    df = generate_synthetic_traffic(n_samples=25, random_state=3)
    csv_path = tmp_path / "sample.csv"
    parquet_path = tmp_path / "sample.parquet"
    df.to_csv(csv_path, index=False)

    out_path = convert_csv_to_parquet(csv_path, parquet_path)
    assert out_path.exists()

    loaded = load_dataset(out_path)
    assert loaded.shape == df.shape


def test_load_dataset_missing_raises(tmp_path):
    """Missing files should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.parquet")


def test_cli_demo_exits_zero(tmp_path, monkeypatch):
    """CLI demo command should run cleanly in a temporary working directory."""
    monkeypatch.chdir(tmp_path)
    exit_code = cli_main(["demo", "--samples", "100"])
    assert exit_code == 0
    assert (tmp_path / "artifacts" / "best_model.joblib").exists()
    assert (tmp_path / "artifacts" / "model_metrics.json").exists()
