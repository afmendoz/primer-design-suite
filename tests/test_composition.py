"""Unit tests for primer_core.features.composition.

Hand-checked expected values per the project's rule that every feature gets a
unit test with a hand-checked expected value.
"""

from __future__ import annotations

import pytest

from primer_core.features.composition import (
    complement,
    gc_clamp,
    gc_content,
    length,
    reverse_complement,
)


def test_gc_content_all_gc() -> None:
    # "GCGC" -> 4/4 G/C bases -> 1.0
    assert gc_content("GCGC") == 1.0


def test_gc_content_all_at() -> None:
    # "ATAT" -> 0/4 G/C bases -> 0.0
    assert gc_content("ATAT") == 0.0


def test_gc_content_mixed() -> None:
    # "ATGC" -> G and C are 2 of 4 bases -> 0.5
    assert gc_content("ATGC") == 0.5


def test_gc_content_case_insensitive() -> None:
    # lowercase "atgc" -> same as "ATGC" -> 0.5
    assert gc_content("atgc") == 0.5


def test_gc_content_known_fraction() -> None:
    # "GATTACA" -> G,C bases are G and C: G,A,T,T,A,C,A -> G(1) + C(1) = 2/7
    assert gc_content("GATTACA") == pytest.approx(2 / 7)


def test_gc_clamp_default_window() -> None:
    # "AAAAAGGCCC" last 5 bases = "GGCCC" -> 5 G/C bases
    assert gc_clamp("AAAAAGGCCC") == 5


def test_gc_clamp_no_gc_in_window() -> None:
    # "GGCCCAAAAA" last 5 bases = "AAAAA" -> 0 G/C bases
    assert gc_clamp("GGCCCAAAAA") == 0


def test_gc_clamp_custom_window() -> None:
    # "ATGCGC", last 3 bases = "CGC" -> 3 G/C bases
    assert gc_clamp("ATGCGC", n=3) == 3


def test_gc_clamp_window_larger_than_sequence() -> None:
    # "GC" with n=5 -> whole sequence "GC" -> 2 G/C bases
    assert gc_clamp("GC", n=5) == 2


def test_gc_clamp_case_insensitive() -> None:
    assert gc_clamp("aaaaaggccc") == 5


def test_gc_clamp_invalid_n_raises() -> None:
    with pytest.raises(ValueError):
        gc_clamp("ATGC", n=0)


def test_length_basic() -> None:
    assert length("ATGCATGC") == 8


def test_length_single_base() -> None:
    assert length("A") == 1


def test_length_case_insensitive() -> None:
    assert length("atgcatgc") == 8


def test_complement_all_a() -> None:
    # A<->T, so complement("AAAA") == "TTTT"
    assert complement("AAAA") == "TTTT"


def test_complement_mixed() -> None:
    # A->T, T->A, G->C, C->G (same orientation, NOT reversed)
    assert complement("ATGC") == "TACG"


def test_complement_case_insensitive() -> None:
    assert complement("atgc") == "TACG"


def test_reverse_complement_basic() -> None:
    # ATGC -> complement TACG -> reversed -> GCAT
    assert reverse_complement("ATGC") == "GCAT"


def test_reverse_complement_palindrome() -> None:
    # GAATTC (EcoRI site) is its own reverse complement
    assert reverse_complement("GAATTC") == "GAATTC"


def test_reverse_complement_case_insensitive() -> None:
    assert reverse_complement("atgc") == "GCAT"


@pytest.mark.parametrize("bad_seq", ["", "ATGX", "atgn", "12345", "ATG C"])
def test_invalid_dna_raises_value_error(bad_seq: str) -> None:
    with pytest.raises(ValueError):
        gc_content(bad_seq)
    with pytest.raises(ValueError):
        gc_clamp(bad_seq)
    with pytest.raises(ValueError):
        length(bad_seq)
    with pytest.raises(ValueError):
        complement(bad_seq)
    with pytest.raises(ValueError):
        reverse_complement(bad_seq)
