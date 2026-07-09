"""Primer duplex thermodynamics — backed by ``primer3-py``.

the project convention is explicit about the thermodynamics library split:

- ``primer3-py`` is the backend for *all* DNA primer thermodynamics here:
  nearest-neighbor Tm (``calc_tm``) and the hairpin / homodimer / heterodimer
  free-energy terms that drive qPCR robustness, plus a dedicated 3'-end delta-G
  term for mispriming propensity.
- ``Bio.SeqUtils.MeltingTemp.Tm_NN`` may be used elsewhere only as a
  cross-check, never as the primary Tm calculation.
- ViennaRNA must **not** be used in this module. ViennaRNA is reserved for
  template / amplicon secondary structure only (see
  ``primer_core.features.structure``).

Unit convention: primer3's ``ThermoResult.dg`` is in cal/mol; every delta-G
this repo returns is in kcal/mol, so we return ``.dg / 1000.0``. When
``structure_found`` is False primer3 reports ``dg == 0.0``, which passes
through unchanged. ``calc_tm`` already returns degrees Celsius and is
returned as-is.
"""

from __future__ import annotations

from primer_core.features.composition import reverse_complement


def calc_tm(seq: str, **primer3_kwargs: object) -> float:
    """Compute nearest-neighbor melting temperature (Tm) in Celsius.

    Backed by ``primer3.bindings.calc_tm`` (primer3-py). Never use
    ``Bio.SeqUtils.MeltingTemp.Tm_NN`` as the primary source here.

    Args:
        seq: Primer sequence (5'->3', DNA alphabet ACGT).
        **primer3_kwargs: Passed through to the primer3-py Tm calculation
            (e.g. salt concentrations, dNTP concentration).

    Returns:
        Melting temperature in degrees Celsius.
    """
    import primer3.bindings as bindings

    return float(bindings.calc_tm(seq, **primer3_kwargs))


def calc_hairpin(seq: str, **primer3_kwargs: object) -> float:
    """Compute hairpin free energy (delta-G, kcal/mol) via primer3-py.

    Args:
        seq: Primer sequence (5'->3', DNA alphabet ACGT).
        **primer3_kwargs: Passed through to ``primer3.bindings.calc_hairpin``.

    Returns:
        Hairpin delta-G in kcal/mol (more negative = more stable structure);
        0.0 if no structure is found.
    """
    import primer3.bindings as bindings

    return bindings.calc_hairpin(seq, **primer3_kwargs).dg / 1000.0


def calc_homodimer(seq: str, **primer3_kwargs: object) -> float:
    """Compute self-dimer (homodimer) free energy (delta-G, kcal/mol).

    Backed by ``primer3.bindings.calc_homodimer``.

    Args:
        seq: Primer sequence (5'->3', DNA alphabet ACGT).
        **primer3_kwargs: Passed through to primer3-py.

    Returns:
        Homodimer delta-G in kcal/mol; 0.0 if no structure is found.
    """
    import primer3.bindings as bindings

    return bindings.calc_homodimer(seq, **primer3_kwargs).dg / 1000.0


def calc_heterodimer(seq1: str, seq2: str, **primer3_kwargs: object) -> float:
    """Compute cross-dimer (heterodimer) free energy between two primers.

    Backed by ``primer3.bindings.calc_heterodimer``. Typically used for the
    forward/reverse primer pair.

    Args:
        seq1: First primer sequence (5'->3').
        seq2: Second primer sequence (5'->3').
        **primer3_kwargs: Passed through to primer3-py.

    Returns:
        Heterodimer delta-G in kcal/mol; 0.0 if no structure is found.
    """
    import primer3.bindings as bindings

    return bindings.calc_heterodimer(seq1, seq2, **primer3_kwargs).dg / 1000.0


def three_prime_end_dg(seq: str, window: int = 5, **primer3_kwargs: object) -> float:
    """Compute a 3'-end-focused delta-G term used as a mispriming risk proxy.

    Takes the last ``window`` bases of the primer's 3' end and computes the
    perfect-duplex binding delta-G of that segment against its reverse
    complement, since 3'-end stability is the dominant driver of extension /
    mispriming propensity in qPCR.

    Args:
        seq: Primer sequence (5'->3', DNA alphabet ACGT).
        window: Number of 3'-terminal bases to consider.
        **primer3_kwargs: Passed through to the underlying primer3-py call.

    Returns:
        3'-end delta-G in kcal/mol; 0.0 if no structure is found.
    """
    import primer3.bindings as bindings

    tail = seq[-window:]
    return bindings.calc_heterodimer(tail, reverse_complement(tail), **primer3_kwargs).dg / 1000.0
