"""Model training, benchmarking, and artifact persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from nids.config import NIDSConfig
from nids.data import split_features_and_target
from nids.features import build_preprocessor


def build_candidate_models(
    X_train: pd.DataFrame,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Create a dictionary of candidate classification pipelines."""
    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scale_numeric=True)),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(random_state=random_state),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, scale_numeric=False)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=150,
                        max_depth=6,
                        learning_rate=0.1,
                        eval_metric="logloss",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    return models


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute standard binary classification metrics on the test set."""
    y_pred = model.predict(X_test)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if hasattr(model, "predict_proba") and len(y_test.unique()) > 1:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: NIDSConfig | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Train candidate models, select the best by F1 score, and optionally save artifacts."""
    cfg = config or NIDSConfig()
    X_train, y_train = split_features_and_target(
        train_df,
        target_column=cfg.target_column,
        drop_columns=cfg.drop_columns,
    )
    X_test, y_test = split_features_and_target(
        test_df,
        target_column=cfg.target_column,
        drop_columns=cfg.drop_columns,
    )

    models = build_candidate_models(X_train, random_state=cfg.random_state)
    leaderboard: list[dict[str, Any]] = []
    trained_pipelines: dict[str, Pipeline] = {}

    for name, pipeline in models.items():
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        metrics = evaluate_model(pipeline, X_test, y_test)
        leaderboard.append(
            {
                "model": name,
                **{k: round(v, 4) for k, v in metrics.items()},
                "train_seconds": round(elapsed, 3),
            }
        )
        trained_pipelines[name] = pipeline

    leaderboard.sort(key=lambda row: (row["f1"], row["roc_auc"]), reverse=True)
    best_name = leaderboard[0]["model"]
    best_model = trained_pipelines[best_name]

    if save_artifacts:
        cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, cfg.model_path)
        payload = {
            "best_model": best_name,
            "leaderboard": leaderboard,
        }
        Path(cfg.metrics_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "leaderboard": pd.DataFrame(leaderboard),
    }
