"""Tests for primer_core.tools: design, score, specificity.

design uses the real primer3-py engine; score fits a tiny sklearn model;
specificity is guarded on the availability of the ``blastn`` binary.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from primer_core.tools.design import design_primers
from primer_core.tools.score import score_candidate

# ~300 bp synthetic template with reasonable base composition.
SAMPLE_TEMPLATE = (
    "GTACGATCCAGTGCTAGCATCGATCGATCGTAGCTAGCTAGCATCGATCGATCGATCGTAGCTAGCTAGC"
    "ATCGGATCCTAGCTAGCTAGCATCGATCGGCTAGCTAGCATCGATCGATCGGCTAGCTAGCATCGATCGT"
    "AGCTAGCTAGCATCGATCGATCGGCTAGCTAGCATCGATCGTAGCTAGCTAGCATCGATCGATCGGCTAG"
    "CTAGCATCGATCGTAGCTAGCTAGCATCGATCGATCGGCTAGCTAGCATCGATCGTAGCTAGCTAGCATC"
    "GATCGATCGGCTAGCTAGCATCG"
)


def test_design_primers_returns_candidates() -> None:
    candidates = design_primers(SAMPLE_TEMPLATE)
    assert len(candidates) >= 1
    first = candidates[0]
    assert first["left_sequence"]
    assert first["right_sequence"]
    assert isinstance(first["product_size"], int)


def test_score_candidate_returns_float(tmp_path: Path) -> None:
    import joblib
    from sklearn.linear_model import LinearRegression

    feature_names = ["gc_content", "three_prime_end_dg", "homodimer_dg"]
    x_train = [[0.5, -2.0, -1.0], [0.6, -3.0, -0.5], [0.4, -1.0, -2.0]]
    y_train = [0.8, 0.9, 0.6]
    model = LinearRegression().fit(x_train, y_train)

    artifact_path = tmp_path / "model.joblib"
    joblib.dump({"model": model, "feature_names": feature_names}, artifact_path)

    candidate = {"gc_content": 0.55, "three_prime_end_dg": -2.5, "homodimer_dg": -0.8}
    report = score_candidate(candidate, artifact_path)
    assert isinstance(report["predicted_efficiency"], float)


def test_score_candidate_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        score_candidate({"a": 1.0}, tmp_path / "nope.joblib")


def test_score_candidate_missing_feature(tmp_path: Path) -> None:
    import joblib
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit([[0.0, 1.0]], [0.5])
    artifact_path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "feature_names": ["a", "b"]}, artifact_path)
    with pytest.raises(ValueError):
        score_candidate({"a": 1.0}, artifact_path)  # missing "b"


@pytest.mark.skipif(shutil.which("blastn") is None, reason="blastn not installed")
def test_check_specificity_smoke() -> None:  # pragma: no cover - needs blastn + DB
    from primer_core.tools.specificity import check_specificity

    # Without a real DB this should raise RuntimeError pointing at the DB path.
    with pytest.raises(RuntimeError):
        check_specificity("ACGTACGTACGTACGTACGT", db_path="/nonexistent/db")
