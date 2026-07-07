"""Thermodynamics checking tool — Tm / delta-G report from primer3-py.

Summarizes ``primer_core.features.thermo`` into a single report dict for a
primer or primer pair. One of the four tools shared by the copilot agent and
plain scripts (see CLAUDE.md); the agent must call this rather than
reasoning about Tm/ΔG itself.
"""

from __future__ import annotations

from typing import Any

from primer_core.features import thermo


def thermo_check(
    primer_seq: str,
    partner_seq: str | None = None,
    primer3_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a Tm / delta-G thermodynamics report for a primer.

    Backed entirely by ``primer_core.features.thermo`` (primer3-py). If
    ``partner_seq`` is given (e.g. the paired forward/reverse primer),
    heterodimer delta-G is included in the report.

    Args:
        primer_seq: Primer sequence (5'->3') to evaluate.
        partner_seq: Optional paired primer sequence for heterodimer
            calculation.
        primer3_settings: Optional overrides passed through to the underlying
            primer3-py calls (salt/dNTP concentrations, etc).

    Returns:
        A report dict with keys ``tm``, ``hairpin_dg``, ``homodimer_dg``,
        ``three_prime_end_dg``, and (if ``partner_seq`` given)
        ``heterodimer_dg``. All delta-G values in kcal/mol, Tm in Celsius.
    """
    settings = primer3_settings or {}
    report: dict[str, Any] = {
        "tm": thermo.calc_tm(primer_seq, **settings),
        "hairpin_dg": thermo.calc_hairpin(primer_seq, **settings),
        "homodimer_dg": thermo.calc_homodimer(primer_seq, **settings),
        "three_prime_end_dg": thermo.three_prime_end_dg(primer_seq, **settings),
    }
    if partner_seq is not None:
        report["heterodimer_dg"] = thermo.calc_heterodimer(primer_seq, partner_seq, **settings)
    return report
