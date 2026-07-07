"""Gradient boosting model builders: XGBoost and LightGBM.

Per CLAUDE.md, this stage of the modeling ladder is "usually the winner" on
this tabular feature set — worth comparing carefully against both the
baselines and the CNN.
"""

from __future__ import annotations

from typing import Any


def build_xgboost(**params: Any) -> Any:
    """Construct an XGBoost regressor for primer efficiency prediction.

    Args:
        **params: Hyperparameters forwarded to ``xgboost.XGBRegressor``
            (e.g. ``n_estimators``, ``max_depth``, ``learning_rate``),
            typically sourced from
            ``predictor/workflows/configs/config.yaml``.

    Returns:
        An unfitted ``xgboost.XGBRegressor`` estimator.
    """
    import xgboost

    return xgboost.XGBRegressor(**params)


def build_lightgbm(**params: Any) -> Any:
    """Construct a LightGBM regressor for primer efficiency prediction.

    Args:
        **params: Hyperparameters forwarded to ``lightgbm.LGBMRegressor``
            (e.g. ``n_estimators``, ``num_leaves``, ``learning_rate``),
            typically sourced from
            ``predictor/workflows/configs/config.yaml``.

    Returns:
        An unfitted ``lightgbm.LGBMRegressor`` estimator.
    """
    import lightgbm

    return lightgbm.LGBMRegressor(**params)
