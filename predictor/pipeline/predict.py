"""Prediction stage: trained model + feature table -> predicted efficiency.

This is the pipeline-side counterpart to
``primer_core.tools.score.score_candidate`` — the same joblib artifact is
loaded and applied here for batch scoring outside the agent context, using
the artifact's ``feature_names`` to select and order columns.

CLI-able: invoked directly or via ``predictor/workflows/Snakefile``.
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
        model_path: Path to a serialized flagship joblib artifact.
        feature_table_path: Path to a CSV feature table of candidates.
        output_path: Destination CSV path for predictions.

    Returns:
        Path to the written predictions file (columns ``primer_id``,
        ``predicted_efficiency``).

    Raises:
        FileNotFoundError: If ``model_path`` or ``feature_table_path`` missing.
    """
    import joblib
    import pandas as pd

    model_path = Path(model_path)
    feature_table_path = Path(feature_table_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    if not feature_table_path.exists():
        raise FileNotFoundError(f"feature table not found: {feature_table_path}")

    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        model = artifact["model"]
        feature_names = artifact["feature_names"]
    else:
        model = artifact
        feature_names = None

    df = pd.read_csv(feature_table_path)
    cols = (
        feature_names
        if feature_names is not None
        else [c for c in df.columns if c not in ("primer_id", "template_id", "label")]
    )
    x = df[cols].to_numpy(dtype=float)
    preds = model.predict(x)

    out = pd.DataFrame(
        {"primer_id": df["primer_id"], "predicted_efficiency": [float(p) for p in preds]}
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
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
