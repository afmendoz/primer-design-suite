"""Tests for primer-template annealing features."""

import pytest

from primer_core.features import annealing as an

# Primer sits at offset 4 on the sense strand of this template.
TEMPLATE = "GGGGATTGACTACGACGCGCTCATTTACGTACGTACGT"
PRIMER = "ATTGACTACGACGCGCTCAT"  # exact 20-mer substring at offset 4


def test_perfect_site_found():
    site = an.best_binding_site(PRIMER, TEMPLATE)
    assert site["strand"] == "+"
    assert site["start"] == 4
    assert site["mismatches"] == 0
    assert site["site"] == PRIMER


def test_mismatch_counts():
    mm = PRIMER[:-1] + ("G" if PRIMER[-1] != "G" else "A")  # flip 3' base
    assert an.mismatch_count(mm, TEMPLATE) == 1
    assert an.three_prime_mismatches(mm, TEMPLATE, n=1) == 1
    # an internal mismatch should not count as a 3' mismatch
    internal = PRIMER[:5] + ("A" if PRIMER[5] != "A" else "C") + PRIMER[6:]
    assert an.mismatch_count(internal, TEMPLATE) == 1
    assert an.three_prime_mismatches(internal, TEMPLATE, n=1) == 0


def test_annealing_dg_perfect_is_strongly_negative():
    dg = an.annealing_delta_g(PRIMER, TEMPLATE)
    assert dg < -10.0  # a perfect 20-mer duplex is strongly stable


def test_mismatch_weakens_duplex():
    mm = PRIMER[:10] + reverse_complement_base(PRIMER[10]) + PRIMER[11:]
    assert an.annealing_delta_g(mm, TEMPLATE) > an.annealing_delta_g(PRIMER, TEMPLATE)


def test_primer_longer_than_template_raises():
    with pytest.raises(ValueError):
        an.best_binding_site("ACGTACGT", "ACGT")


def reverse_complement_base(b: str) -> str:
    return {"A": "T", "T": "A", "C": "G", "G": "C"}[b]
