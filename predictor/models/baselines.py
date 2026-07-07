"""Baseline model builders: ElasticNet and RandomForest.

These are the floor of the modeling ladder (see CLAUDE.md) — simple,
well-understood models that gradient boosting and the CNN must beat to
justify their added complexity.
"""

from __future__ import annotations

from typing import Any


def build_elasticnet(**params: object) -> Any:
    """Construct an ElasticNet regressor for primer efficiency prediction.

    Args:
        **params: Hyperparameters forwarded to
            ``sklearn.linear_model.ElasticNet`` (e.g. ``alpha``, ``l1_ratio``),
            typically sourced from ``predictor/workflows/configs/config.yaml``.

    Returns:
        An unfitted ``sklearn.linear_model.ElasticNet`` estimator.
    """
    raise NotImplementedError


def build_random_forest(**params: object) -> Any:
    """Construct a RandomForest regressor for primer efficiency prediction.

    Args:
        **params: Hyperparameters forwarded to
            ``sklearn.ensemble.RandomForestRegressor`` (e.g. ``n_estimators``,
            ``max_depth``), typically sourced from
            ``predictor/workflows/configs/config.yaml``.

    Returns:
        An unfitted ``sklearn.ensemble.RandomForestRegressor`` estimator.
    """
    raise NotImplementedError
