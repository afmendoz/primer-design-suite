"""Tests for primer_core.tools: design, score, specificity.

design uses the real primer3-py engine; score fits a tiny sklearn model;
specificity is guarded on the availability of the ``blastn`` binary.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from primer_core.tools.design import design_primers
from primer_core.tools.score import score_candidate, score_dual_head

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


def test_score_dual_head_returns_both_signals(tmp_path: Path) -> None:
    import joblib
    from sklearn.linear_model import LinearRegression, LogisticRegression

    feats = ["gc_content", "length", "tm"]
    xr = [[0.5, 20, 60.0], [0.6, 22, 62.0], [0.4, 18, 58.0]]
    reg = LinearRegression().fit(xr, [0.6, 0.8, 0.3])
    clf = LogisticRegression().fit(xr, [0, 1, 0])
    reg_path = tmp_path / "reg.joblib"
    clf_path = tmp_path / "clf.joblib"
    joblib.dump({"model": reg, "feature_names": feats}, reg_path)
    joblib.dump({"model": clf, "feature_names": feats}, clf_path)

    out = score_dual_head("ACGTACGTACGTACGTACGT", classifier_path=clf_path, regressor_path=reg_path)
    assert isinstance(out["amplify_probability"], float)
    assert isinstance(out["predicted_efficiency"], float)
    assert out["caveats"]  # in-domain caveat present
    assert any("no template" in n for n in out.get("notes", []))  # noted missing template


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


def test_check_specificity_local_requires_db() -> None:
    from primer_core.tools.specificity import check_specificity

    with pytest.raises(ValueError):
        check_specificity("ACGTACGTACGTACGTACGT")  # local mode needs db_path


@pytest.mark.skipif(
    shutil.which("blastn") is None or shutil.which("makeblastdb") is None,
    reason="BLAST+ not installed",
)
def test_check_specificity_local_db(tmp_path: Path) -> None:
    import subprocess

    from primer_core.tools.specificity import check_specificity

    fasta = tmp_path / "refs.fasta"
    fasta.write_text(
        ">subj_a\nGGGGATTGACTACGACGCGCTCATTTACGTACGT\n"
        ">subj_b\nTTTTATTGACTACGACGCGCTCATGGGGCCACGT\n"
    )
    db = tmp_path / "db"
    subprocess.run(
        ["makeblastdb", "-in", str(fasta), "-dbtype", "nucl", "-out", str(db)],
        check=True,
        capture_output=True,
    )
    primer = "ATTGACTACGACGCGCTCAT"  # exact substring of both subjects
    rep = check_specificity(primer, db_path=str(db))
    assert rep["source"].startswith("local:")
    assert rep["off_target_hit_count"] >= 2  # hits both subjects
    # excluding one subject lowers the off-target count
    rep2 = check_specificity(primer, db_path=str(db), exclude_target_id="subj_a")
    assert rep2["off_target_hit_count"] < rep["off_target_hit_count"]


def test_parse_blast_record_remote_schema() -> None:
    from types import SimpleNamespace

    from primer_core.tools.specificity import _parse_blast_record

    hsp = SimpleNamespace(
        identities=18, align_length=20, query_start=1, query_end=20, expect=1e-3, bits=36.2
    )
    aln = SimpleNamespace(accession="ACC123", hit_id="gi|1|ACC123", hsps=[hsp])
    hits = _parse_blast_record(SimpleNamespace(alignments=[aln]))
    assert len(hits) == 1
    assert hits[0]["subject_id"] == "ACC123"
    assert hits[0]["pident"] == pytest.approx(90.0)
    assert hits[0]["qstart"] == 1 and hits[0]["qend"] == 20
    assert hits[0]["bitscore"] == 36.2
