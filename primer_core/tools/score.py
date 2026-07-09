"""Candidate scoring tools — bridge to the trained predictor heads.

Two entry points:

* ``score_candidate(features, model_path)`` — the low-level primitive: load one
  joblib artifact and predict from a feature dict. Used by the predictor
  pipeline and by ``score_dual_head``.
* ``score_dual_head(primer_seq, template_seq, classifier_path, regressor_path)``
  — the copilot-facing scorer. It featurizes the primer (with its template, so
  the primer-template annealing features are available) and returns BOTH head-A
  signals: ``amplify_probability`` (classification) and ``predicted_efficiency``
  (regression), each with explicit in-domain caveats. Per CLAUDE.md the copilot
  never asserts these numbers itself — they come from here.

Artifact contract: a joblib dump of EITHER a fitted sklearn-style estimator, OR
a dict ``{"model": estimator, "feature_names": [...]}``. With ``feature_names``
the candidate features are ordered accordingly; otherwise keys are sorted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_IN_DOMAIN_CAVEAT = (
    "scores are in-domain to openPrimeR designed IGHV primers (a set pre-filtered "
    "for a GC clamp and no self-dimers); treat as out-of-domain extrapolation for "
    "primers or templates unlike that set"
)


def _prepare(model_path: str | Path, candidate_features: dict[str, Any]):
    """Load an artifact and build its ordered feature row from a feature dict."""
    import joblib

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model artifact not found: {model_path}")
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        estimator = artifact["model"]
        feature_names = artifact["feature_names"]
    else:
        estimator = artifact
        feature_names = sorted(candidate_features)
    try:
        row = [candidate_features[name] for name in feature_names]
    except KeyError as exc:
        raise ValueError(
            f"candidate_features is missing required feature: {exc.args[0]!r}"
        ) from exc
    return estimator, row


def score_candidate(candidate_features: dict[str, Any], model_path: str | Path) -> dict[str, Any]:
    """Predict efficiency for a featurized candidate from one model artifact.

    Returns:
        ``{"predicted_efficiency": float}``.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        ValueError: If a declared ``feature_names`` entry is missing.
    """
    estimator, row = _prepare(model_path, candidate_features)
    return {"predicted_efficiency": float(estimator.predict([row])[0])}


def _amplify_probability(model_path: str | Path, candidate_features: dict[str, Any]) -> float:
    """P(amplify) from a classifier artifact's ``predict_proba``."""
    estimator, row = _prepare(model_path, candidate_features)
    return float(estimator.predict_proba([row])[0][1])


def score_dual_head(
    primer_seq: str,
    template_seq: str | None = None,
    classifier_path: str | Path | None = None,
    regressor_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score a primer with the head-A models: P(amplify) + efficiency + caveats.

    Featurizes ``primer_seq`` (with ``template_seq`` when given, enabling the
    primer-template annealing features the head-A models rely on), then applies
    whichever head artifacts are provided.

    Args:
        primer_seq: Primer sequence (5'->3').
        template_seq: Template to featurize annealing against. If omitted, the
            annealing features are absent and head-A models that need them are
            skipped with a note.
        classifier_path: Head-A amplification classifier artifact.
        regressor_path: Head-A efficiency regressor artifact.

    Returns:
        A dict with any of ``amplify_probability`` / ``predicted_efficiency``
        that could be computed, plus ``features``, ``caveats``, and ``notes``.
    """
    from primer_core.featurize import featurize_primer

    feats = featurize_primer(primer_seq, template_seq)
    out: dict[str, Any] = {"features": feats, "caveats": [_IN_DOMAIN_CAVEAT]}
    notes: list[str] = []
    if template_seq is None:
        notes.append("no template provided; primer-template annealing features unavailable")

    if classifier_path:
        try:
            out["amplify_probability"] = _amplify_probability(classifier_path, feats)
        except (ValueError, FileNotFoundError) as exc:
            notes.append(f"classifier skipped: {exc}")
    if regressor_path:
        try:
            out["predicted_efficiency"] = score_candidate(feats, regressor_path)[
                "predicted_efficiency"
            ]
        except (ValueError, FileNotFoundError) as exc:
            notes.append(f"regressor skipped: {exc}")
    if notes:
        out["notes"] = notes
    return out
