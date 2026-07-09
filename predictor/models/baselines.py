"""Baseline model builders: ElasticNet and RandomForest.

These are the floor of the modeling ladder (by project convention) — simple,
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


def build_logreg(**params: Any) -> Any:
    """Construct a scaled LogisticRegression classifier (amplify y/n).

    Baseline for head A's classification head; wrapped in a StandardScaler
    pipeline since a linear model needs standardized inputs. ``class_weight``
    defaults to ``"balanced"`` given the ~14% positive rate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    params.setdefault("class_weight", "balanced")
    params.setdefault("max_iter", 1000)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(**params)),
        ]
    )


def build_random_forest_classifier(**params: Any) -> Any:
    """Construct a RandomForest classifier for the amplification label."""
    from sklearn.ensemble import RandomForestClassifier

    params.setdefault("class_weight", "balanced")
    return RandomForestClassifier(**params)
