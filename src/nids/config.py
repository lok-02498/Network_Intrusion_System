"""Central configuration for the Network Intrusion Detection System."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class NIDSConfig:
    """Configuration defaults for data paths, target, and training."""

    random_state: int = 42
    target_column: str = "label"
    drop_columns: tuple[str, ...] = ("id", "attack_cat", "traffic_type")
    dataset_dir: Path = field(default_factory=lambda: Path("dataset"))
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    train_filename: str = "train_full.parquet"
    test_filename: str = "test_full.parquet"
    model_filename: str = "best_model.joblib"
    metrics_filename: str = "model_metrics.json"
    high_risk_threshold: float = 0.80
    medium_risk_threshold: float = 0.40

    @property
    def train_path(self) -> Path:
        return self.dataset_dir / self.train_filename

    @property
    def test_path(self) -> Path:
        return self.dataset_dir / self.test_filename

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_filename

    @property
    def metrics_path(self) -> Path:
        return self.artifacts_dir / self.metrics_filename
