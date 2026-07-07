"""Modeling ladder: baselines -> gradient boosting -> small deep model.

Per CLAUDE.md, report the full ladder honestly: ElasticNet / RandomForest
baselines, then XGBoost / LightGBM (usually the winner on this tabular
feature set), then a small PyTorch 1D-CNN / k-mer MLP on raw sequence. If
boosting beats the deep model at the available N, say so plainly.

``MODEL_BUILDERS`` is the single registry mapping a config ``model.name`` to
its builder; ``get_model`` is the helper used by ``train.py`` and
``evaluate.py`` so neither hardcodes the mapping.
"""

from __future__ import annotations

from typing import Any, Callable

from predictor.models.baselines import build_elasticnet, build_random_forest
from predictor.models.boosting import build_lightgbm, build_xgboost
from predictor.models.cnn import build_kmer_mlp, build_sequence_cnn

MODEL_BUILDERS: dict[str, Callable[..., Any]] = {
    "elasticnet": build_elasticnet,
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "cnn": build_sequence_cnn,
    "kmer_mlp": build_kmer_mlp,
}

# The tabular ladder used for the honest baseline-vs-boosting comparison
# (excludes the optional deep models, which need torch and raw sequence).
LADDER_MODELS: tuple[str, ...] = ("elasticnet", "random_forest", "xgboost", "lightgbm")


def get_model(name: str, params: dict[str, Any] | None = None) -> Any:
    """Build an unfitted estimator by registry name.

    Args:
        name: Registry key (see ``MODEL_BUILDERS``).
        params: Hyperparameters forwarded to the builder.

    Returns:
        An unfitted estimator.

    Raises:
        ValueError: If ``name`` is not a known model.
    """
    if name not in MODEL_BUILDERS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](**(params or {}))


__all__ = ["MODEL_BUILDERS", "LADDER_MODELS", "get_model"]
