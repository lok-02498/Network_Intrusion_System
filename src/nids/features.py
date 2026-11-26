"""Feature identification and preprocessing pipeline builders."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return lists of numeric and categorical column names."""
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical network features."""
    numeric_cols, categorical_cols = get_feature_types(X)

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")
