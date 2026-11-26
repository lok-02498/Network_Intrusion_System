"""Smoke tests for package imports and core NIDS workflow."""

from nids import (
    NIDSConfig,
    assign_risk_tier,
    generate_synthetic_traffic,
    predict_traffic,
    train_and_evaluate,
)


def test_package_imports():
    """Verify core package exports are importable."""
    cfg = NIDSConfig()
    assert cfg.target_column == "label"
    assert assign_risk_tier(0.9) == "HIGH"
    assert assign_risk_tier(0.5) == "MEDIUM"
    assert assign_risk_tier(0.1) == "LOW"


def test_end_to_end_synthetic_smoke(tmp_path):
    """Run a fast end-to-end train + predict smoke test on synthetic traffic."""
    df = generate_synthetic_traffic(n_samples=120, random_state=7)
    train_df = df.iloc[:90].copy()
    test_df = df.iloc[90:].copy()

    cfg = NIDSConfig(artifacts_dir=tmp_path / "artifacts")
    result = train_and_evaluate(train_df, test_df, config=cfg, save_artifacts=True)

    assert cfg.model_path.exists()
    assert cfg.metrics_path.exists()
    assert not result["leaderboard"].empty

    scored = predict_traffic(test_df, result["best_model"], config=cfg)
    assert list(scored.columns) == ["id", "prediction", "intrusion_probability", "risk_tier"]
    assert len(scored) == len(test_df)
