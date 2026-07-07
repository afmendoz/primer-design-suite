"""Fast end-to-end pipeline test: simulate -> featurize -> train -> score.

The key cross-component assertion is that the artifact ``train_model`` saves
loads back through ``primer_core.tools.score.score_candidate`` and yields a
float — i.e. the flagship predictor and the copilot's scoring tool agree on
the artifact contract. Metric values are not asserted (only finiteness).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("pandas")
pytest.importorskip("sklearn")


def _write_config(tmp_path: Path, dataset: Path, features: Path) -> Path:
    config = {
        "data": {
            "seed": 7,
            "n_templates": 6,
            "primers_per_template": 4,
            "template_length": 120,
        },
        "paths": {
            "dataset": str(dataset),
            "feature_table": str(features),
            "model_artifact": str(tmp_path / "model.joblib"),
            "evaluation_report": str(tmp_path / "report.json"),
            "predictions": str(tmp_path / "predictions.csv"),
            "blast_db": str(tmp_path / "no_such_db"),
        },
        "features": {
            "composition": True,
            "thermo": True,
            "structure": False,
            "specificity": False,
        },
        "thermo": {"primer3_settings": {"mv_conc": 50.0, "dv_conc": 3.0}},
        "model": {"name": "random_forest", "params": {"n_estimators": 20, "random_state": 0}},
        "ladder": {"random_forest": {"n_estimators": 20, "random_state": 0}},
        "cv": {"grouping_key": "template_id", "n_splits": 3, "random_state": 0},
        # No mlflow section -> training skips MLflow entirely.
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(config, handle)
    return config_path


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    import pandas as pd

    from predictor.pipeline.featurize import featurize_dataset
    from predictor.pipeline.simulate import simulate_from_config
    from predictor.pipeline.train import train_model
    from primer_core.tools.score import score_candidate

    dataset = tmp_path / "primers.csv"
    features = tmp_path / "features.csv"
    config_path = _write_config(tmp_path, dataset, features)

    # simulate
    simulate_from_config(dataset, config_path)
    assert dataset.exists()
    assert dataset.with_suffix(".provenance.yaml").exists()
    df = pd.read_csv(dataset)
    assert len(df) == 6 * 4

    # featurize
    featurize_dataset(dataset, features, config_path)
    feats = pd.read_csv(features)
    assert len(feats) == 6 * 4
    assert {"primer_id", "template_id", "label", "gc_content", "three_prime_end_dg"} <= set(
        feats.columns
    )

    # train (grouped CV) + save artifact
    artifact_path = tmp_path / "model.joblib"
    metrics = train_model(features, "random_forest", artifact_path, config_path)
    assert math.isfinite(metrics["spearman"])
    assert math.isfinite(metrics["rmse"])
    assert artifact_path.exists()

    # KEY cross-component check: the saved artifact scores through score_candidate.
    feature_cols = [c for c in feats.columns if c not in ("primer_id", "template_id", "label")]
    candidate = {col: float(feats.iloc[0][col]) for col in feature_cols}
    result = score_candidate(candidate, artifact_path)
    assert isinstance(result["predicted_efficiency"], float)
    assert math.isfinite(result["predicted_efficiency"])
