"""Tests for primer_core.features.structure (ViennaRNA backend).

Values are library-version dependent, so we assert ranges and relative
orderings rather than exact floats.
"""

from __future__ import annotations

from primer_core.features.structure import (
    amplicon_secondary_structure_dg,
    binding_site_accessibility,
    fold_template,
)

OPEN_SEQ = "AAAAAAAAAAAAAAAAAAAA"
HAIRPIN_SEQ = "GGGGCCCCAAAAGGGGCCCC"


def test_fold_template_returns_dotbracket() -> None:
    structure = fold_template(HAIRPIN_SEQ)
    assert isinstance(structure, str)
    assert len(structure) == len(HAIRPIN_SEQ)
    assert set(structure) <= set(".()")


def test_amplicon_dg_is_float_and_nonpositive_for_hairpin() -> None:
    mfe = amplicon_secondary_structure_dg(HAIRPIN_SEQ)
    assert isinstance(mfe, float)
    # A structured sequence should have MFE <= 0.
    assert mfe <= 0.0


def test_binding_site_accessibility_in_range() -> None:
    acc = binding_site_accessibility(HAIRPIN_SEQ, 0, len(HAIRPIN_SEQ))
    assert 0.0 <= acc <= 1.0


def test_open_tract_more_accessible_than_hairpin() -> None:
    open_acc = binding_site_accessibility(OPEN_SEQ, 0, len(OPEN_SEQ))
    hairpin_acc = binding_site_accessibility(HAIRPIN_SEQ, 0, len(HAIRPIN_SEQ))
    assert open_acc > hairpin_acc
