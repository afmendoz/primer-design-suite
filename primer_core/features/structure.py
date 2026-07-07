"""Template / amplicon secondary structure features — backed by ViennaRNA.

Per CLAUDE.md: ViennaRNA is used **only** for template / amplicon secondary
structure (binding-site accessibility). It must never be used for primer
duplex thermodynamics — that is ``primer3-py``'s job (see
``primer_core.features.thermo``).

These functions estimate how "open" / accessible a candidate primer binding
site is within the folded template or amplicon, which affects whether the
primer can actually anneal in practice.
"""

from __future__ import annotations


def fold_template(seq: str, **vienna_kwargs: object) -> str:
    """Compute the minimum free energy (MFE) secondary structure of a template.

    Backed by ViennaRNA's ``RNA.fold``. Used as an input to binding-site
    accessibility calculations, not as a primer duplex thermodynamics tool.

    Args:
        seq: Template or amplicon sequence.
        **vienna_kwargs: Unused; present for forward compatibility.

    Returns:
        Dot-bracket secondary structure notation for ``seq``.
    """
    import RNA

    structure, _mfe = RNA.fold(seq)
    return structure


def binding_site_accessibility(
    template_seq: str,
    site_start: int,
    site_end: int,
    **vienna_kwargs: object,
) -> float:
    """Estimate accessibility of a candidate primer binding site.

    Computed from ViennaRNA partition-function base-pairing probabilities
    over the template / amplicon fold. The unpaired probability at position
    ``i`` is ``1 - sum_j bpp[min(i, j)][max(i, j)]`` (ViennaRNA's ``bpp`` is
    1-indexed), and the result is the mean unpaired probability across the
    site. Higher values indicate a more single-stranded (accessible) region.

    Args:
        template_seq: Full template or amplicon sequence.
        site_start: 0-based start index of the candidate binding site.
        site_end: 0-based, exclusive end index of the candidate binding site.
        **vienna_kwargs: Unused; present for forward compatibility.

    Returns:
        Accessibility score in [0.0, 1.0], where 1.0 is fully unpaired
        (accessible) across the site.
    """
    import RNA

    n = len(template_seq)
    fc = RNA.fold_compound(template_seq)
    fc.pf()
    bpp = fc.bpp()

    total = 0.0
    count = 0
    for pos in range(site_start, site_end):
        i = pos + 1  # 0-based input -> 1-based ViennaRNA index
        paired = 0.0
        for j in range(1, n + 1):
            if j == i:
                continue
            paired += bpp[min(i, j)][max(i, j)]
        p_unpaired = 1.0 - paired
        total += p_unpaired
        count += 1

    mean = total / count if count else 0.0
    return max(0.0, min(1.0, mean))


def amplicon_secondary_structure_dg(seq: str, **vienna_kwargs: object) -> float:
    """Compute the MFE free energy of the amplicon fold.

    Backed by ViennaRNA's ``RNA.fold``. Used as a coarse robustness feature:
    amplicons with strong secondary structure can impede polymerase
    read-through.

    Args:
        seq: Amplicon sequence.
        **vienna_kwargs: Unused; present for forward compatibility.

    Returns:
        MFE free energy (delta-G, kcal/mol) of the amplicon fold.
    """
    import RNA

    _structure, mfe = RNA.fold(seq)
    return float(mfe)
