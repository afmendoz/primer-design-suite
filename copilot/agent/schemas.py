"""JSON tool schemas + the enforced ranked-output JSON contract.

Two kinds of schema live here:

1. Anthropic tool-use JSON schemas for the four ``primer_core.tools``
   functions, used to register them with the agent loop.
2. Pydantic models for the ranked-output JSON contract the system prompt
   (``copilot/agent/system_prompt.py``) forces the model to emit, so it can
   be parsed defensively rather than trusted blindly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# --- Anthropic tool-use JSON schemas -----------------------------------

DESIGN_PRIMERS_TOOL_SCHEMA: dict[str, Any] = {
    "name": "design_primers",
    "description": "Design candidate primer pairs for a template using primer3-py.",
    "input_schema": {
        "type": "object",
        "properties": {
            "template_seq": {"type": "string"},
            "target_region": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 2,
                "maxItems": 2,
            },
            "primer3_settings": {"type": "object"},
        },
        "required": ["template_seq"],
    },
}

THERMO_CHECK_TOOL_SCHEMA: dict[str, Any] = {
    "name": "thermo_check",
    "description": "Compute a Tm / delta-G thermodynamics report for a primer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "primer_seq": {"type": "string"},
            "partner_seq": {"type": "string"},
            "primer3_settings": {"type": "object"},
        },
        "required": ["primer_seq"],
    },
}

CHECK_SPECIFICITY_TOOL_SCHEMA: dict[str, Any] = {
    "name": "check_specificity",
    "description": "Run a BLAST off-target search for a primer and summarize the results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "primer_seq": {"type": "string"},
            "db_path": {"type": "string"},
            "exclude_target_id": {"type": "string"},
        },
        "required": ["primer_seq", "db_path"],
    },
}

SCORE_CANDIDATE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "score_candidate",
    "description": "Score a featurized primer candidate using the trained flagship model.",
    "input_schema": {
        "type": "object",
        "properties": {
            "candidate_features": {"type": "object"},
            "model_path": {"type": "string"},
        },
        "required": ["candidate_features", "model_path"],
    },
}

ALL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    DESIGN_PRIMERS_TOOL_SCHEMA,
    THERMO_CHECK_TOOL_SCHEMA,
    CHECK_SPECIFICITY_TOOL_SCHEMA,
    SCORE_CANDIDATE_TOOL_SCHEMA,
]


# --- Ranked-output JSON contract ----------------------------------------


class RankedCandidate(BaseModel):
    """One ranked primer candidate in the enforced JSON output.

    Every field here must trace back to a specific tool call result — the
    agent must not fabricate any of these values (see
    ``copilot/agent/system_prompt.py``).

    Attributes:
        rank: 1-based rank among returned candidates (1 = best).
        forward_primer: Forward primer sequence, verbatim from
            ``design_primers``.
        reverse_primer: Reverse primer sequence, verbatim from
            ``design_primers``.
        tm_forward: Forward primer Tm (C), from ``thermo_check``.
        tm_reverse: Reverse primer Tm (C), from ``thermo_check``.
        off_target_hit_count: From ``check_specificity``.
        predicted_efficiency: From ``score_candidate``.
        risk_flags: Short machine-readable risk tags (e.g. "dimer_risk",
            "offtarget_risk", "low_predicted_efficiency").
    """

    rank: int
    forward_primer: str
    reverse_primer: str
    tm_forward: float
    tm_reverse: float
    off_target_hit_count: int
    predicted_efficiency: float
    risk_flags: list[str] = Field(default_factory=list)


class RankedOutput(BaseModel):
    """Top-level enforced JSON contract emitted by the agent.

    Attributes:
        candidates: Ranked list of candidates, best first.
        design_goal: Echo of the parsed design goal for traceability.
    """

    candidates: list[RankedCandidate]
    design_goal: str


def parse_ranked_output(raw_json: str) -> RankedOutput:
    """Defensively parse the agent's ranked-output JSON block.

    Args:
        raw_json: Raw JSON text emitted by the agent (no markdown fences, no
            surrounding prose expected, but this function should not assume
            that holds and should fail with a clear error if it doesn't).

    Returns:
        A validated ``RankedOutput``.

    Raises:
        ValueError: If ``raw_json`` is not valid JSON or fails schema
            validation.
    """
    raise NotImplementedError
