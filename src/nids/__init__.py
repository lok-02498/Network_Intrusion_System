"""Network Intrusion Detection System (NIDS) package."""

__author__ = "Lokesh Deshwal"
__roll_no__ = "2401421062"

from nids.config import NIDSConfig
from nids.data import convert_csv_to_parquet, generate_synthetic_traffic, load_dataset
from nids.predict import assign_risk_tier, predict_traffic
from nids.train import train_and_evaluate

__all__ = [
    "NIDSConfig",
    "assign_risk_tier",
    "convert_csv_to_parquet",
    "generate_synthetic_traffic",
    "load_dataset",
    "predict_traffic",
    "train_and_evaluate",
]
