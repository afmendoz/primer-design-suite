"""Sequence composition features.

These are pure, deterministic, no-I/O functions over a DNA sequence string,
per the project's requirement that ``primer_core/features/`` functions be
unit-testable in isolation. Unlike the other modules in this package, these
are trivial enough to implement for real in the scaffold rather than stub
out, and are covered by ``tests/test_composition.py``.
"""

from __future__ import annotations

_VALID_BASES = frozenset("ACGT")
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _validate_dna(seq: str) -> str:
    """Uppercase and validate that ``seq`` is a non-empty DNA (ACGT) string.

    Args:
        seq: Candidate DNA sequence.

    Returns:
        The uppercased sequence.

    Raises:
        ValueError: If ``seq`` is empty or contains characters outside ACGT
            (case-insensitive).
    """
    if not seq:
        raise ValueError("sequence must be non-empty")
    upper = seq.upper()
    invalid = set(upper) - _VALID_BASES
    if invalid:
        raise ValueError(f"sequence contains non-DNA characters: {''.join(sorted(invalid))!r}")
    return upper


def gc_content(seq: str) -> float:
    """Compute GC content as a fraction in [0, 1].

    Case-insensitive; validates that ``seq`` uses only the DNA alphabet ACGT.

    Args:
        seq: DNA sequence.

    Returns:
        Fraction of bases that are G or C, in [0.0, 1.0].

    Raises:
        ValueError: If ``seq`` is empty or contains non-ACGT characters.
    """
    upper = _validate_dna(seq)
    gc_count = sum(1 for base in upper if base in "GC")
    return gc_count / len(upper)


def gc_clamp(seq: str, n: int = 5) -> int:
    """Count G/C bases in the last ``n`` bases (the 3' end) of the sequence.

    Case-insensitive; validates that ``seq`` uses only the DNA alphabet ACGT.

    Args:
        seq: DNA sequence, 5'->3'.
        n: Number of 3'-terminal bases to inspect. Clamped to ``len(seq)`` if
            the sequence is shorter than ``n``.

    Returns:
        Count of G/C bases within the last ``n`` bases.

    Raises:
        ValueError: If ``seq`` is empty, contains non-ACGT characters, or if
            ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    upper = _validate_dna(seq)
    tail = upper[-n:]
    return sum(1 for base in tail if base in "GC")


def length(seq: str) -> int:
    """Return the length of a DNA sequence.

    Case-insensitive; validates that ``seq`` uses only the DNA alphabet ACGT.

    Args:
        seq: DNA sequence.

    Returns:
        Number of bases in ``seq``.

    Raises:
        ValueError: If ``seq`` is empty or contains non-ACGT characters.
    """
    upper = _validate_dna(seq)
    return len(upper)


def complement(seq: str) -> str:
    """Return the base complement of a DNA sequence (A<->T, C<->G).

    Case-insensitive; validates that ``seq`` uses only the DNA alphabet ACGT.

    Args:
        seq: DNA sequence.

    Returns:
        The uppercased complement (same 5'->3' orientation as the input,
        i.e. NOT reversed).

    Raises:
        ValueError: If ``seq`` is empty or contains non-ACGT characters.
    """
    upper = _validate_dna(seq)
    return "".join(_COMPLEMENT[base] for base in upper)


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence.

    Case-insensitive; validates that ``seq`` uses only the DNA alphabet ACGT.

    Args:
        seq: DNA sequence (5'->3').

    Returns:
        The uppercased reverse complement (5'->3').

    Raises:
        ValueError: If ``seq`` is empty or contains non-ACGT characters.
    """
    upper = _validate_dna(seq)
    return "".join(_COMPLEMENT[base] for base in reversed(upper))
