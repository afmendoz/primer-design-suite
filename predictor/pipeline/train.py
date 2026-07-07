"""Training stage: feature table -> trained model artifact.

Trains a model from the modeling ladder in ``predictor/models`` (baselines,
boosting, or CNN) on a featurized dataset. Cross-validation must be grouped
by gene/template (see the ``grouping_key`` in the dataset's provenance
sidecar / config) so a primer and its sequence neighbors never span train and
test folds — CLAUDE.md forbids reporting ungrouped CV metrics. Reports
Spearman correlation and RMSE, not bare R².

CLI-able: intended to be invoked directly or via
``predictor/workflows/Snakefile``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def train_model(
    feature_table_path: str | Path,
    model_name: str,
    output_model_path: str | Path,
    config_path: str | Path = "predictor/workflows/configs/config.yaml",
) -> dict[str, Any]:
    """Train a model with grouped cross-validation and persist the artifact.

    Uses grouped CV (grouping by gene/template, per the dataset's
    ``grouping_key``) to prevent leakage between train and test folds.

    Args:
        feature_table_path: Path to the feature table produced by
            ``featurize.py``.
        model_name: Which model builder to use (e.g. ``"elasticnet"``,
            ``"random_forest"``, ``"xgboost"``, ``"lightgbm"``, ``"cnn"``);
            see ``predictor/models``.
        output_model_path: Destination path for the serialized trained model.
        config_path: Path to the pipeline config controlling model
            hyperparameters and CV grouping key.

    Returns:
        A metrics dict with grouped-CV ``spearman`` and ``rmse`` (and any
        per-fold breakdown), reported honestly per the modeling ladder in
        CLAUDE.md — a suspiciously high score should be investigated, not
        celebrated.

    Raises:
        FileNotFoundError: If ``feature_table_path`` or ``config_path`` is
            missing.
        ValueError: If ``model_name`` is not recognized.
    """
    raise NotImplementedError


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module."""
    parser = argparse.ArgumentParser(description="Train a primer efficiency model.")
    parser.add_argument("--features", required=True, help="Path to feature table")
    parser.add_argument("--model", required=True, help="Model name (see predictor/models)")
    parser.add_argument("--output", required=True, help="Path to write trained model artifact")
    parser.add_argument(
        "--config",
        default="predictor/workflows/configs/config.yaml",
        help="Path to pipeline config YAML",
    )
    return parser


def main() -> None:
    """CLI entrypoint: ``python -m predictor.pipeline.train``."""
    args = _build_arg_parser().parse_args()
    train_model(args.features, args.model, args.output, args.config)


if __name__ == "__main__":
    main()
