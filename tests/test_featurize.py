"""Tests for the shared featurizer, openPrimeR loader, and classifier registry."""

import pandas as pd
import pytest

from predictor.models import get_classifier
from primer_core.featurize import ANNEALING_FEATURES, INTRINSIC_FEATURES, featurize_primer
from primer_core.io.openprimer import load_openprimer, load_templates

PRIMER = "ATTGACTACGACGCGCTCAT"
TEMPLATE = "GGGGATTGACTACGACGCGCTCATTTACGTACGT"


def test_featurize_intrinsic_only():
    f = featurize_primer(PRIMER)
    assert set(f) == set(INTRINSIC_FEATURES)
    assert all(isinstance(v, float) for v in f.values())


def test_featurize_with_template_adds_annealing():
    f = featurize_primer(PRIMER, TEMPLATE)
    assert set(f) == set(INTRINSIC_FEATURES) | set(ANNEALING_FEATURES)
    assert f["mismatch_count"] == 0.0  # exact substring
    assert f["annealing_dg"] < 0.0


def test_load_openprimer_cleans_and_labels(tmp_path):
    fasta = tmp_path / "t.fasta"
    fasta.write_text(
        f">ACC|IGHVX-1*01|Homo sapiens|F|EXON|1..2|n nt|1| | | | |n| | |\n{TEMPLATE}\n"
    )
    csv = tmp_path / "m.csv"
    pd.DataFrame(
        {
            "Primer": ["p1", "p2", "p3"],
            "Primer_Sequence": [PRIMER.lower(), "ACGTN", PRIMER.lower()],  # p2 degenerate
            "Template": ["IGHVX-1*01", "IGHVX-1*01", "IGHV-missing*01"],  # p3 uncovered
            "Group": ["G", "G", "G"],
            "primer_efficiency": [0.9, 0.9, 0.1],
        }
    ).to_csv(csv, index=False)

    df = load_openprimer(csv, fasta)
    assert list(df.columns) == [
        "primer_id",
        "primer_seq",
        "template_id",
        "group",
        "template_seq",
        "efficiency",
        "amplified",
    ]
    assert len(df) == 1  # degenerate + uncovered dropped
    assert df["amplified"].iloc[0] == 1  # 0.9 >= 0.5
    assert df["template_seq"].iloc[0] == TEMPLATE


def test_load_templates_parses_allele(tmp_path):
    fasta = tmp_path / "t.fasta"
    fasta.write_text(">ACC|IGHV1-2*02|Homo|F|EXON\nACGTACGT\n")
    assert load_templates(fasta) == {"IGHV1-2*02": "ACGTACGT"}


def test_get_classifier_registry():
    assert get_classifier("logreg") is not None
    with pytest.raises(ValueError):
        get_classifier("does_not_exist")
