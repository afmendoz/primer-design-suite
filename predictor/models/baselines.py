"""Baseline model builders: ElasticNet and RandomForest.

These are the floor of the modeling ladder (see CLAUDE.md) — simple,
well-understood models that gradient boosting and the CNN must beat to
justify their added complexity. ElasticNet is wrapped in a StandardScaler
pipeline since a linear model needs standardized inputs.
"""

from __future__ import annotations

from typing import Any


def build_elasticnet(**params: Any) -> Any:
    """Construct a scaled ElasticNet regressor for efficiency prediction.

    Args:
        **params: Hyperparameters forwarded to
            ``sklearn.linear_model.ElasticNet`` (e.g. ``alpha``, ``l1_ratio``),
            typically sourced from ``predictor/workflows/configs/config.yaml``.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline`` of
        ``StandardScaler -> ElasticNet``.
    """
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("elasticnet", ElasticNet(**params)),
        ]
    )


def build_random_forest(**params: Any) -> Any:
    """Construct a RandomForest regressor for primer efficiency prediction.

    Args:
        **params: Hyperparameters forwarded to
            ``sklearn.ensemble.RandomForestRegressor`` (e.g. ``n_estimators``,
            ``max_depth``), typically sourced from
            ``predictor/workflows/configs/config.yaml``.

    Returns:
        An unfitted ``sklearn.ensemble.RandomForestRegressor`` estimator.
    """
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**params)
