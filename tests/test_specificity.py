"""Unit tests for primer_core.features.specificity (pure, synthetic hits).

Hand-checked expected values over synthetic BLAST hit dicts.
"""

from __future__ import annotations

from primer_core.features.specificity import (
    best_off_target_bitscore,
    off_target_hit_count,
    three_prime_offtarget_complementarity,
)


def _hit(subject_id: str, bitscore: float, pident: float, qstart: int, qend: int) -> dict:
    return {
        "subject_id": subject_id,
        "bitscore": bitscore,
        "pident": pident,
        "qstart": qstart,
        "qend": qend,
    }


def test_off_target_hit_count_excludes_target() -> None:
    hits = [
        _hit("target", 40.0, 100.0, 1, 20),
        _hit("off1", 30.0, 95.0, 1, 18),
        _hit("off2", 28.0, 90.0, 3, 20),
    ]
    # Excluding "target" leaves 2 off-target hits.
    assert off_target_hit_count(hits, exclude_target_id="target") == 2


def test_off_target_hit_count_none_counts_all() -> None:
    hits = [_hit("a", 40.0, 100.0, 1, 20), _hit("b", 30.0, 95.0, 1, 18)]
    assert off_target_hit_count(hits, exclude_target_id=None) == 2


def test_best_off_target_bitscore() -> None:
    hits = [
        _hit("target", 99.0, 100.0, 1, 20),
        _hit("off1", 30.0, 95.0, 1, 18),
        _hit("off2", 45.0, 90.0, 3, 20),
    ]
    # Best off-target (excluding target) bitscore is 45.0.
    assert best_off_target_bitscore(hits, exclude_target_id="target") == 45.0


def test_best_off_target_bitscore_empty() -> None:
    hits = [_hit("target", 99.0, 100.0, 1, 20)]
    assert best_off_target_bitscore(hits, exclude_target_id="target") == 0.0


def test_three_prime_complementarity_full_overlap() -> None:
    # primer length 20, window 5 -> 3' window is query coords [16, 20].
    # Most concerning hit (highest bitscore) fully covers [16,20] at 100% pident.
    primer = "A" * 20
    hits = [
        _hit("off_low", 10.0, 100.0, 1, 5),  # no overlap with [16,20]
        _hit("off_best", 50.0, 100.0, 1, 20),  # covers full window
    ]
    # coverage = 5/5 = 1.0; pident/100 = 1.0 -> 1.0
    assert three_prime_offtarget_complementarity(primer, hits) == 1.0


def test_three_prime_complementarity_partial_overlap_and_pident() -> None:
    # primer length 20, window 5 -> window [16,20].
    # hit covers query [18,20] -> overlap = 20-18+1 = 3 -> coverage 3/5 = 0.6
    # pident 90 -> 0.9 -> score = 0.6 * 0.9 = 0.54
    primer = "A" * 20
    hits = [_hit("off", 50.0, 90.0, 18, 20)]
    assert three_prime_offtarget_complementarity(primer, hits) == 0.54


def test_three_prime_complementarity_no_overlap() -> None:
    # hit is entirely in the 5' region, no overlap with 3' window -> 0.0
    primer = "A" * 20
    hits = [_hit("off", 50.0, 100.0, 1, 5)]
    assert three_prime_offtarget_complementarity(primer, hits) == 0.0


def test_three_prime_complementarity_empty() -> None:
    assert three_prime_offtarget_complementarity("A" * 20, []) == 0.0
