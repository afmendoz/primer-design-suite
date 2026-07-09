"""Canonical primer featurizer — the single place feature vectors are built.

Both the predictor pipeline and the copilot's ``score_candidate`` assemble
features here, so the two halves of the project never drift (project convention: one
shared ``primer_core`` featurizer). Intrinsic features need only the primer;
primer-template annealing features are added when a template is supplied.
"""

from __future__ import annotations

from typing import Any

from primer_core.features import annealing as an
from primer_core.features import composition as comp
from primer_core.features import thermo as th

INTRINSIC_FEATURES = [
    "gc_content",
    "gc_clamp",
    "length",
    "tm",
    "hairpin_dg",
    "homodimer_dg",
    "three_prime_end_dg",
]
ANNEALING_FEATURES = ["annealing_dg", "mismatch_count", "three_prime_mismatch"]


def featurize_primer(
    primer_seq: str,
    template_seq: str | None = None,
    primer3_settings: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Build the feature dict for one primer (optionally against a template).

    Args:
        primer_seq: Primer sequence (5'->3', ACGT).
        template_seq: If given, primer-template annealing features are added.
        primer3_settings: Optional overrides for the primer3-py thermo calls.

    Returns:
        A dict of feature name -> float. Contains ``INTRINSIC_FEATURES`` always,
        plus ``ANNEALING_FEATURES`` when ``template_seq`` is provided.
    """
    kw = primer3_settings or {}
    feats: dict[str, float] = {
        "gc_content": comp.gc_content(primer_seq),
        "gc_clamp": float(comp.gc_clamp(primer_seq)),
        "length": float(comp.length(primer_seq)),
        "tm": th.calc_tm(primer_seq, **kw),
        "hairpin_dg": th.calc_hairpin(primer_seq, **kw),
        "homodimer_dg": th.calc_homodimer(primer_seq, **kw),
        "three_prime_end_dg": th.three_prime_end_dg(primer_seq, **kw),
    }
    if template_seq is not None:
        feats["annealing_dg"] = an.annealing_delta_g(primer_seq, template_seq)
        feats["mismatch_count"] = float(an.mismatch_count(primer_seq, template_seq))
        feats["three_prime_mismatch"] = float(an.three_prime_mismatches(primer_seq, template_seq))
    return feats
