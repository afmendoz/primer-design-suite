"""Candidate scoring tool — calls the trained flagship predictor model.

This is the bridge between the two halves of the story: the ``predictor``
flagship model (trained via ``predictor/pipeline/train.py``) is loaded here
and invoked to score a fully-featurized primer candidate for predicted
efficiency. The copilot agent must call this rather than asserting an
efficiency estimate itself (see CLAUDE.md).

Artifact contract: ``model_path`` is a ``joblib`` dump of EITHER a fitted
sklearn-style estimator, OR a dict ``{"model": estimator, "feature_names":
[...]}``. When ``feature_names`` is provided the candidate features are
ordered accordingly; otherwise the candidate feature keys are sorted for a
deterministic column order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def score_candidate(
    candidate_features: dict[str, Any],
    model_path: str | Path,
) -> dict[str, Any]:
    """Score a featurized primer candidate using the trained flagship model.

    Args:
        candidate_features: Feature dict for one candidate, as produced by
            ``predictor/pipeline/featurize.py`` (composition, thermo,
            structure, and specificity features combined).
        model_path: Path to a serialized trained model artifact (see the
            artifact contract in this module's docstring).

    Returns:
        A report dict with key ``predicted_efficiency`` (float).

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ValueError: If the artifact declares a ``feature_names`` entry that is
            missing from ``candidate_features``.
    """
    import joblib

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")

    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):
        estimator = artifact["model"]
        feature_names = artifact["feature_names"]
        try:
            row = [candidate_features[name] for name in feature_names]
        except KeyError as exc:
            raise ValueError(
                f"candidate_features is missing required feature: {exc.args[0]!r}"
            ) from exc
    else:
        estimator = artifact
        feature_names = sorted(candidate_features)
        row = [candidate_features[name] for name in feature_names]

    prediction = estimator.predict([row])
    return {"predicted_efficiency": float(prediction[0])}
