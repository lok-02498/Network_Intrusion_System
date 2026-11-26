"""Smoke and sanity tests for the Network Intrusion Detection System."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from praquet_converter import convert_csv_to_parquet


def test_environment_imports():
    """Verify core ML and data dependencies are importable."""
    import joblib
    import numpy
    import pandas
    import pyarrow
    import sklearn

    assert joblib is not None
    assert numpy is not None
    assert pandas is not None
    assert pyarrow is not None
    assert sklearn is not None


def test_parquet_datasets_exist_and_valid():
    """Verify committed Parquet datasets exist, are non-empty, and contain valid labels."""
    train_path = Path("dataset/train_full.parquet")
    test_path = Path("dataset/test_full.parquet")

    assert train_path.exists(), "dataset/train_full.parquet does not exist"
    assert test_path.exists(), "dataset/test_full.parquet does not exist"

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    assert len(df_train) > 0, "Train parquet dataset is empty"
    assert len(df_test) > 0, "Test parquet dataset is empty"
    assert "label" in df_train.columns
    assert "label" in df_test.columns
    assert set(df_train["label"].unique()).issubset({0, 1})


def test_csv_to_parquet_converter(tmp_path):
    """Verify convert_csv_to_parquet round-trips a sample CSV cleanly."""
    sample = pd.DataFrame({"dur": [0.1, 1.2], "proto": ["tcp", "udp"], "label": [0, 1]})
    csv_path = tmp_path / "sample.csv"
    parquet_path = tmp_path / "sample.parquet"
    sample.to_csv(csv_path, index=False)

    out_path = convert_csv_to_parquet(csv_path, parquet_path)
    assert out_path.exists()

    loaded = pd.read_parquet(out_path)
    assert loaded.shape == sample.shape


def test_baseline_pipeline_sample_fit():
    """Verify a lightweight preprocessing + classifier pipeline fits on a sample slice."""
    df = pd.read_parquet("dataset/train_full.parquet").sample(n=200, random_state=42)
    drop_cols = [c for c in ("id", "attack_cat", "traffic_type", "label") if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["label"].astype(int)

    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    clf = Pipeline([("prep", preprocessor), ("lr", LogisticRegression(max_iter=500))])
    clf.fit(X, y)
    preds = clf.predict(X.head(10))
    assert len(preds) == 10
