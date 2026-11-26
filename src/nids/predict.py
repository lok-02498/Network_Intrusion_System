"""Inference and risk-tier scoring for network traffic records."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from nids.config import NIDSConfig


def assign_risk_tier(
    probability: float,
    high_threshold: float = 0.80,
    medium_threshold: float = 0.40,
) -> str:
    """Map an intrusion probability to a human-readable alert tier."""
    if probability >= high_threshold:
        return "HIGH"
    if probability >= medium_threshold:
        return "MEDIUM"
    return "LOW"


def load_model(model_path: str | Path) -> Pipeline:
    """Load a persisted scikit-learn pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}")
    return joblib.load(path)


def predict_traffic(
    df: pd.DataFrame,
    model: Pipeline,
    config: NIDSConfig | None = None,
) -> pd.DataFrame:
    """Score traffic rows and append prediction, intrusion probability, and risk tier."""
    cfg = config or NIDSConfig()
    drop_cols = [col for col in (*cfg.drop_columns, cfg.target_column) if col in df.columns]
    features = df.drop(columns=drop_cols)

    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
    else:
        probabilities = predictions.astype(float)

    result = pd.DataFrame(index=df.index)
    if "id" in df.columns:
        result["id"] = df["id"]
    result["prediction"] = predictions.astype(int)
    result["intrusion_probability"] = probabilities.round(4)
    result["risk_tier"] = [
        assign_risk_tier(
            float(prob),
            high_threshold=cfg.high_risk_threshold,
            medium_threshold=cfg.medium_risk_threshold,
        )
        for prob in probabilities
    ]
    return result
