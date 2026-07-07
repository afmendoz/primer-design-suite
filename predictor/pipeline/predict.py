"""Prediction stage: trained model + new candidates -> predicted efficiency.

This is the pipeline-side counterpart to ``primer_core.tools.score.
score_candidate`` — the same trained model artifact is loaded and applied
here for batch scoring outside the agent context.

CLI-able: intended to be invoked directly or via
``predictor/workflows/Snakefile``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def predict(
    model_path: str | Path,
    feature_table_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Run batch prediction for a feature table against a trained model.

    Args:
        model_path: Path to a serialized trained model artifact.
        feature_table_path: Path to a feature table of candidates to score.
        output_path: Destination path for predictions.

    Returns:
        Path to the written predictions file.

    Raises:
        FileNotFoundError: If ``model_path`` or ``feature_table_path`` is
            missing.
    """
    raise NotImplementedError


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module."""
    parser = argparse.ArgumentParser(description="Batch-predict primer efficiency.")
    parser.add_argument("--model", required=True, help="Path to trained model artifact")
    parser.add_argument("--features", required=True, help="Path to feature table")
    parser.add_argument("--output", required=True, help="Path to write predictions")
    return parser


def main() -> None:
    """CLI entrypoint: ``python -m predictor.pipeline.predict``."""
    args = _build_arg_parser().parse_args()
    predict(args.model, args.features, args.output)


if __name__ == "__main__":
    main()
