"""Primer-template annealing features (Primer3-backed duplex ΔG).

`primer_core` originally had only *intrinsic* primer features; validation
against the openPrimeR set showed those do not capture primer-to-template
annealing energy or mismatch context, which are what actually drive whether a
primer amplifies a given template. This module fills that gap: it aligns a
primer to a template by ungapped minimum-mismatch search (both strands), then
computes the primer-template duplex free energy with Primer3 at that site,
plus mismatch-context features.

Per CLAUDE.md, primer duplex thermodynamics use primer3-py (not ViennaRNA).
These functions are deterministic given their inputs; the alignment search is
pure, the ΔG term calls into the compiled Primer3 bindings.
"""

from __future__ import annotations

from primer_core.features.composition import _validate_dna, reverse_complement


def _min_mismatch_offset(primer: str, template: str) -> tuple[int, int]:
    """Ungapped slide of ``primer`` over ``template``; return (offset, mismatches).

    Returns the offset with the fewest mismatches (first such offset on ties).
    """
    length = len(primer)
    best_off, best_mm = 0, length + 1
    for off in range(0, len(template) - length + 1):
        mm = sum(1 for i in range(length) if primer[i] != template[off + i])
        if mm < best_mm:
            best_off, best_mm = off, mm
            if mm == 0:
                break
    return best_off, best_mm


def best_binding_site(primer: str, template: str) -> dict[str, object]:
    """Locate the best ungapped primer binding site on either template strand.

    Args:
        primer: Primer sequence (5'->3', ACGT).
        template: Template sequence (ACGT).

    Returns:
        A dict with ``strand`` ("+" sense / "-" reverse-complement), ``start``
        (0-based offset on that strand), ``mismatches`` (count at the site),
        and ``site`` (the template subsequence the primer aligns to, same
        5'->3' orientation and length as the primer).

    Raises:
        ValueError: If either sequence is empty/non-ACGT, or the primer is
            longer than the template.
    """
    p = _validate_dna(primer)
    t = _validate_dna(template)
    if len(p) > len(t):
        raise ValueError("primer is longer than template")
    off_f, mm_f = _min_mismatch_offset(p, t)
    t_rc = reverse_complement(t)
    off_r, mm_r = _min_mismatch_offset(p, t_rc)
    if mm_r < mm_f:
        return {
            "strand": "-",
            "start": off_r,
            "mismatches": mm_r,
            "site": t_rc[off_r : off_r + len(p)],
        }
    return {"strand": "+", "start": off_f, "mismatches": mm_f, "site": t[off_f : off_f + len(p)]}


def mismatch_count(primer: str, template: str) -> int:
    """Number of mismatches between the primer and its best template site."""
    return int(best_binding_site(primer, template)["mismatches"])


def three_prime_mismatches(primer: str, template: str, n: int = 1) -> int:
    """Count mismatches within the primer's 3'-terminal ``n`` bases at its site.

    3'-end mismatches are the dominant driver of mispriming / failed extension,
    so this is tracked separately from the total mismatch count.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    p = _validate_dna(primer)
    site = str(best_binding_site(p, template)["site"])
    return sum(1 for a, b in zip(p[-n:], site[-n:]) if a != b)


def annealing_delta_g(primer: str, template: str) -> float:
    """Primer-template duplex free energy (ΔG, kcal/mol) at the best site.

    The primer hybridizes to the strand complementary to the site it matches,
    so ΔG is computed between the primer and the reverse complement of its
    aligned template subsequence (mismatches weaken the duplex). Backed by
    ``primer3.bindings.calc_heterodimer``.

    Returns:
        Duplex ΔG in kcal/mol (more negative = more stable annealing); 0.0 if
        Primer3 finds no stable structure.
    """
    import primer3.bindings as bindings

    p = _validate_dna(primer)
    site = str(best_binding_site(p, template)["site"])
    return bindings.calc_heterodimer(p, reverse_complement(site)).dg / 1000.0
