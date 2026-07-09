"""JSON tool schemas + the enforced ranked-output JSON contract.

Two kinds of schema live here:

1. Anthropic tool-use JSON schemas for the four ``primer_core.tools``
   functions, used to register them with the agent loop.
2. Pydantic models for the ranked-output JSON contract the system prompt
   (``copilot/agent/system_prompt.py``) forces the model to emit, so it can
   be parsed defensively rather than trusted blindly.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

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
            "db_path": {
                "type": "string",
                "description": "(injected by the agent runtime; do not supply)",
            },
            "exclude_target_id": {"type": "string"},
        },
        "required": ["primer_seq"],
    },
}

SCORE_CANDIDATE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "score_candidate",
    "description": (
        "Score a primer with the trained head-A models: returns P(amplify) and "
        "predicted efficiency (with in-domain caveats). Provide just the primer "
        "sequence; the tool featurizes it (against the design template) itself."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "primer_seq": {"type": "string", "description": "Primer sequence, 5'->3'."},
            "model_path": {
                "type": "string",
                "description": "(injected by the agent runtime; do not supply)",
            },
        },
        "required": ["primer_seq"],
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
        predicted_efficiency: Head-A regression, from ``score_candidate``.
        amplify_probability: Head-A P(amplify), from ``score_candidate`` (may be
            absent if only the regressor head was available).
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
    amplify_probability: float | None = None
    risk_flags: list[str] = Field(default_factory=list)


class RankedOutput(BaseModel):
    """Top-level enforced JSON contract emitted by the agent.

    Attributes:
        candidates: Ranked list of candidates, best first.
        design_goal: Echo of the parsed design goal for traceability.
    """

    candidates: list[RankedCandidate]
    design_goal: str


def strip_code_fences(text: str) -> str:
    """Remove a wrapping markdown code fence (```` ``` ```` or ```` ```json ````)."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()[1:]  # drop the opening fence line
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def find_json_span(text: str) -> tuple[int, int] | None:
    """Return ``(start, end)`` of the first balanced ``{...}`` object in ``text``.

    Brace matching is string-aware: braces inside JSON string literals (and
    escaped quotes) are ignored, so prose or values containing braces do not
    confuse the scan. Returns ``None`` if no balanced object is found.
    """
    depth = 0
    start: int | None = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                return (start, i + 1)
    return None


def parse_ranked_output(raw_json: str) -> RankedOutput:
    """Defensively parse the agent's ranked-output JSON block.

    Tolerant of accidental markdown fences and of surrounding prose (the model
    is told to emit bare JSON, but we do not trust that): fences are stripped,
    then the whole string is tried as JSON, and failing that the first balanced
    ``{...}`` object is extracted and parsed.

    Args:
        raw_json: Raw text emitted by the agent.

    Returns:
        A validated ``RankedOutput``.

    Raises:
        ValueError: If no valid JSON object is found or it fails schema
            validation.
    """
    s = strip_code_fences(raw_json)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        span = find_json_span(s)
        if span is None:
            raise ValueError("no JSON object found in ranked output")
        try:
            data = json.loads(s[span[0] : span[1]])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid ranked-output JSON: {exc}") from exc
    try:
        return RankedOutput(**data)
    except ValidationError as exc:
        raise ValueError(f"ranked output failed schema validation: {exc}") from exc
