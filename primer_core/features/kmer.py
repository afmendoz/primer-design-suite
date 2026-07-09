"""K-mer frequency features (length-agnostic sequence composition).

Unlike the primer thermodynamic features, k-mer frequencies are defined for a
sequence of any length, so they are the sensible tabular representation for
Source B's 90-160 nt amplicon sequences (Primer3 duplex thermodynamics are
undefined above ~60 bp). Pure and deterministic, by project convention.
"""

from __future__ import annotations

from itertools import product

from primer_core.features.composition import _validate_dna


def all_kmers(k: int) -> list[str]:
    """All 4**k DNA k-mers in a fixed (lexicographic) order."""
    if k <= 0:
        raise ValueError("k must be positive")
    return ["".join(p) for p in product("ACGT", repeat=k)]


def kmer_frequencies(seq: str, k: int = 3) -> dict[str, float]:
    """Normalized frequencies of every DNA k-mer in ``seq``.

    Args:
        seq: DNA sequence (ACGT, any length >= k).
        k: k-mer length.

    Returns:
        Dict mapping each of the 4**k k-mers to its frequency (counts summing
        to 1 over the ``len(seq) - k + 1`` windows). k-mers absent from the
        sequence map to 0.0.

    Raises:
        ValueError: If ``seq`` is empty/non-ACGT, ``k`` <= 0, or ``len(seq)`` < k.
    """
    s = _validate_dna(seq)
    if len(s) < k:
        raise ValueError(f"sequence length {len(s)} shorter than k={k}")
    counts = dict.fromkeys(all_kmers(k), 0)
    total = len(s) - k + 1
    for i in range(total):
        counts[s[i : i + k]] += 1
    return {km: c / total for km, c in counts.items()}
