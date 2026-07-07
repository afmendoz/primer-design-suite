"""Tests for primer_core.features.thermo and primer_core.tools.thermo.

These exercise the real primer3-py backend, so they require primer3 to be
importable (it is, in the ``primer`` env).
"""

from __future__ import annotations

from primer_core.tools.thermo import thermo_check

# A plausible ~20-mer primer.
PRIMER = "ACGTACGTACGTACGTACGT"


def test_thermo_check_keys_and_tm_plausible() -> None:
    report = thermo_check(PRIMER)
    for key in ("tm", "hairpin_dg", "homodimer_dg", "three_prime_end_dg"):
        assert key in report
    assert isinstance(report["tm"], float)
    assert 0.0 < report["tm"] < 100.0
    # heterodimer only present when a partner is supplied
    assert "heterodimer_dg" not in report


def test_thermo_check_with_partner_adds_heterodimer() -> None:
    report = thermo_check(PRIMER, partner_seq="ACGTACGTACGTACGTTTTT")
    assert "heterodimer_dg" in report
    assert isinstance(report["heterodimer_dg"], float)
