"""Specificity features — derived from BLAST, not string matching.

By project convention: off-target hit count, best off-target bit score, and 3'-end
complementarity to the nearest off-target must come from a real BLAST search
(``blastn -task blastn-short``, appropriate for short primer queries) against
a reference/off-target database — never from naive substring or k-mer string
matching. String matching cannot account for mismatches, gaps, or genomic
context the way local alignment does.

These functions consume pre-computed BLAST hit records (parsed from
``blastn -outfmt 6`` output produced by
``primer_core.tools.specificity.check_specificity``) rather than running
BLAST themselves, keeping this module's functions pure and testable.

Hit dicts use keys: ``subject_id``, ``bitscore``, ``pident``, ``qstart``,
``qend`` (BLAST query coordinates are 1-based), among the standard outfmt-6
columns.
"""

from __future__ import annotations

from typing import Any


def _off_target_hits(
    blast_hits: list[dict[str, Any]],
    exclude_target_id: str | None,
) -> list[dict[str, Any]]:
    """Filter to off-target hits (subject_id != exclude_target_id).

    If ``exclude_target_id`` is None, every hit counts as off-target.
    """
    if exclude_target_id is None:
        return list(blast_hits)
    return [h for h in blast_hits if h["subject_id"] != exclude_target_id]


def off_target_hit_count(
    blast_hits: list[dict[str, Any]],
    exclude_target_id: str | None = None,
) -> int:
    """Count off-target BLAST hits for a primer.

    Args:
        blast_hits: Parsed ``blastn -task blastn-short`` hit records (each a
            dict with at minimum a ``subject_id`` key).
        exclude_target_id: Subject/accession ID of the intended on-target
            sequence, excluded from the off-target count. If None, all hits
            are counted.

    Returns:
        Number of BLAST hits not matching ``exclude_target_id``.
    """
    return len(_off_target_hits(blast_hits, exclude_target_id))


def best_off_target_bitscore(
    blast_hits: list[dict[str, Any]],
    exclude_target_id: str | None = None,
) -> float:
    """Return the highest bit score among off-target BLAST hits.

    Args:
        blast_hits: Parsed hit records, each with ``subject_id`` and
            ``bitscore`` keys.
        exclude_target_id: Subject/accession ID of the intended on-target
            sequence, excluded from consideration.

    Returns:
        Highest bit score among off-target hits, or 0.0 if there are none.
    """
    off_targets = _off_target_hits(blast_hits, exclude_target_id)
    if not off_targets:
        return 0.0
    return float(max(h["bitscore"] for h in off_targets))


def three_prime_offtarget_complementarity(
    primer_seq: str,
    blast_hits: list[dict[str, Any]],
    exclude_target_id: str | None = None,
    window: int = 5,
) -> float:
    """Estimate 3'-end complementarity to the most concerning off-target hit.

    High 3'-end complementarity to an off-target site is a strong mispriming
    risk signal even when the overall alignment score is moderate, since
    polymerase extension is most sensitive to 3'-end matches.

    Among off-target hits the "most concerning" hit is the one with the
    highest bit score. Let ``L = len(primer_seq)``. The primer's 3' window in
    1-based query coordinates is ``[L - window + 1, L]``. The score is the
    overlap length between the hit's ``[qstart, qend]`` and that window,
    divided by ``window`` (3'-window coverage in [0, 1]), multiplied by the
    hit's ``pident / 100``.

    Args:
        primer_seq: Primer sequence (5'->3').
        blast_hits: Parsed hit records including ``bitscore``, ``pident``,
            ``qstart``, ``qend`` keys (1-based BLAST query coordinates).
        exclude_target_id: Subject/accession ID of the intended on-target
            sequence, excluded from consideration.
        window: Number of 3'-terminal bases to assess complementarity over.

    Returns:
        Complementarity score in [0.0, 1.0] for the most concerning
        off-target hit's 3'-end alignment, or 0.0 if there are no off-target
        hits.
    """
    off_targets = _off_target_hits(blast_hits, exclude_target_id)
    if not off_targets:
        return 0.0

    hit = max(off_targets, key=lambda h: h["bitscore"])

    length = len(primer_seq)
    win_start = length - window + 1
    win_end = length

    overlap = min(hit["qend"], win_end) - max(hit["qstart"], win_start) + 1
    overlap = max(overlap, 0)
    coverage = overlap / window

    return coverage * (hit["pident"] / 100.0)
