"""Evaluation stage: trained model -> grouped CV metrics + SHAP report.

Re-runs (or loads cached) grouped cross-validation results and computes SHAP
importances. Per CLAUDE.md, SHAP importances must land on biologically
sensible features (3'-ΔG, dimer ΔG, GC-clamp); if the top features are
nonsensical, that is a signal of leakage or a featurization bug, not a result
to report.

CLI-able: intended to be invoked directly or via
``predictor/workflows/Snakefile``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def evaluate_model(
    model_path: str | Path,
    feature_table_path: str | Path,
    output_report_path: str | Path,
    config_path: str | Path = "predictor/workflows/configs/config.yaml",
) -> dict[str, Any]:
    """Compute grouped CV metrics and SHAP importances for a trained model.

    Args:
        model_path: Path to a serialized trained model artifact.
        feature_table_path: Path to the feature table used for evaluation.
        output_report_path: Destination path for the written evaluation
            report (metrics + SHAP summary).
        config_path: Path to the pipeline config controlling the CV grouping
            key and evaluation settings.

    Returns:
        A report dict with grouped-CV ``spearman``, ``rmse``, and a SHAP
        feature-importance summary.

    Raises:
        FileNotFoundError: If any input path is missing.
    """
    raise NotImplementedError


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module."""
    parser = argparse.ArgumentParser(description="Evaluate a trained primer efficiency model.")
    parser.add_argument("--model", required=True, help="Path to trained model artifact")
    parser.add_argument("--features", required=True, help="Path to feature table")
    parser.add_argument("--output", required=True, help="Path to write evaluation report")
    parser.add_argument(
        "--config",
        default="predictor/workflows/configs/config.yaml",
        help="Path to pipeline config YAML",
    )
    return parser


def main() -> None:
    """CLI entrypoint: ``python -m predictor.pipeline.evaluate``."""
    args = _build_arg_parser().parse_args()
    evaluate_model(args.model, args.features, args.output, args.config)


if __name__ == "__main__":
    main()
