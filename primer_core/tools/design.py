"""Primer design tool — wraps ``primer3-py``'s ``bindings.design_primers``.

The LLM copilot never writes a primer sequence itself; every candidate
sequence returned to the agent (and to any plain script) comes from this
function. This is the single point of contact with Primer3's design engine
and is imported by both ``copilot/agent`` and standalone pipeline code —
never reimplemented downstream (see CLAUDE.md).
"""

from __future__ import annotations

from typing import Any

_DEFAULT_GLOBAL_ARGS: dict[str, Any] = {
    "PRIMER_OPT_SIZE": 20,
    "PRIMER_MIN_SIZE": 18,
    "PRIMER_MAX_SIZE": 25,
    "PRIMER_OPT_TM": 60.0,
    "PRIMER_MIN_TM": 57.0,
    "PRIMER_MAX_TM": 63.0,
    "PRIMER_PRODUCT_SIZE_RANGE": [[75, 150]],
}


def design_primers(
    template_seq: str,
    target_region: tuple[int, int] | None = None,
    primer3_settings: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Design candidate primer pairs for a template using primer3-py.

    Wraps ``primer3.bindings.design_primers``. Every sequence in the returned
    candidates comes directly from Primer3's design engine — this function
    must not fabricate or post-hoc edit sequences.

    Args:
        template_seq: Full template/target sequence to design primers against.
        target_region: Optional ``(start, length)`` region that the amplicon
            must include, passed to Primer3's ``SEQUENCE_TARGET``.
        primer3_settings: Optional overrides merged onto the default primer3-py
            global settings (e.g. product size range, Tm bounds).

    Returns:
        A list of candidate primer pair records, one dict per returned pair,
        with keys ``left_sequence``, ``right_sequence``, ``left_tm``,
        ``right_tm``, ``product_size``, ``penalty``, ``left_start``,
        ``left_len``, ``right_start``, ``right_len``. Ordered by Primer3's
        internal ranking.
    """
    import primer3.bindings as bindings

    seq_args: dict[str, Any] = {"SEQUENCE_TEMPLATE": template_seq}
    if target_region is not None:
        start, length = target_region
        seq_args["SEQUENCE_TARGET"] = [start, length]

    global_args = dict(_DEFAULT_GLOBAL_ARGS)
    if primer3_settings:
        global_args.update(primer3_settings)

    result = bindings.design_primers(seq_args, global_args)

    num_returned = int(result.get("PRIMER_PAIR_NUM_RETURNED", 0))
    candidates: list[dict[str, Any]] = []
    for i in range(num_returned):
        left_pos, left_len = result[f"PRIMER_LEFT_{i}"]
        right_pos, right_len = result[f"PRIMER_RIGHT_{i}"]
        candidates.append(
            {
                "left_sequence": result[f"PRIMER_LEFT_{i}_SEQUENCE"],
                "right_sequence": result[f"PRIMER_RIGHT_{i}_SEQUENCE"],
                "left_tm": result[f"PRIMER_LEFT_{i}_TM"],
                "right_tm": result[f"PRIMER_RIGHT_{i}_TM"],
                "product_size": result[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"],
                "penalty": result[f"PRIMER_PAIR_{i}_PENALTY"],
                "left_start": left_pos,
                "left_len": left_len,
                "right_start": right_pos,
                "right_len": right_len,
            }
        )
    return candidates
