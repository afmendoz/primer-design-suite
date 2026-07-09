"""Provider-agnostic tool-use agent loop.

The LLM orchestrates and interprets tool results; it never fabricates a
sequence or a number (see ``copilot/agent/system_prompt.py``). This module
drives the loop against any :class:`~copilot.agent.providers.LLMProvider`:
send the design goal, execute whichever of the four ``primer_core.tools``
functions the model calls, feed results back, and repeat until the model emits
the final ranked-output JSON + prose memo.

Nothing here is tied to a specific LLM vendor — the provider adapter owns all
wire-format details, and the safety model (tools are the only source of
sequences/numbers, runtime-injected filesystem paths, defensive JSON parsing)
lives here, provider-independently. The default provider is OpenAI-style; a
provider (or a fake, for tests) can be injected to avoid any SDK or API key.
"""

from __future__ import annotations

from typing import Any

from copilot.agent.providers import LLMProvider, ToolResult, get_provider
from copilot.agent.schemas import (
    ALL_TOOL_SCHEMAS,
    RankedOutput,
    find_json_span,
    parse_ranked_output,
    strip_code_fences,
)
from copilot.agent.system_prompt import SYSTEM_PROMPT

# Default backend. claude-fable-5 is not assumed — the copilot is provider
# neutral and defaults to an OpenAI-compatible endpoint (which also covers
# local models via a custom base_url).
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"

_TOOL_NAMES = {"design_primers", "thermo_check", "check_specificity", "score_candidate"}

# Accepted argument names per tool (from the schemas) and which args are DNA
# sequences. Real models routinely emit spurious arguments and stray whitespace;
# we drop unknown kwargs and normalize sequence whitespace rather than crash.
_TOOL_PARAMS = {s["name"]: set(s["input_schema"].get("properties", {})) for s in ALL_TOOL_SCHEMAS}
_SEQ_ARGS = {"primer_seq", "template_seq", "partner_seq", "seq1", "seq2"}


def run_agent_loop(
    user_message: str,
    *,
    provider: LLMProvider | None = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 10,
    max_repairs: int = 2,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Drive the tool-use loop for one user design request.

    Args:
        user_message: Natural-language design goal + constraints.
        provider: An :class:`~copilot.agent.providers.LLMProvider`. If omitted,
            the default OpenAI-style provider is constructed lazily (this is the
            only path that needs a provider SDK / API key).
        model: Model id, used only when ``provider`` is not supplied.
        max_turns: Maximum number of tool-use round-trips before aborting.
        max_repairs: If the final message lacks valid ranked JSON, how many
            times to nudge the model to re-emit just the JSON before failing.
        context: Runtime config injected into tool calls — ``template_seq``,
            ``classifier_path``/``regressor_path`` (for ``score_candidate``) and
            ``blast_db`` (for ``check_specificity``).

    Returns:
        A dict with ``ranked_output`` (:class:`RankedOutput`), ``design_memo``
        (prose), ``raw_final_text``, and ``n_turns``.

    Raises:
        RuntimeError: If ``max_turns`` is exceeded without a final answer.
        ValueError: If no valid ranked-output JSON is produced within
            ``max_repairs`` repair attempts.
    """
    if provider is None:
        provider = get_provider(DEFAULT_PROVIDER, model=model)

    _REPAIR = (
        "Your previous message did not contain the required ranked-output JSON. "
        "If you have not yet designed and scored candidates with the tools, do that "
        'now. Then respond with ONLY the JSON object (top-level keys "design_goal" '
        'and "candidates"), no prose and no code fences.'
    )

    provider.start_conversation(SYSTEM_PROMPT, ALL_TOOL_SCHEMAS)
    turn = provider.send_user(user_message)
    n_turns = 1
    repairs = 0
    memo_source = None  # first prose answer, kept as the memo even if JSON is repaired

    while True:
        if turn.tool_calls:
            if n_turns >= max_turns:
                raise RuntimeError(f"agent exceeded max_turns={max_turns} without a final answer")
            results = [
                ToolResult(tc.id, dispatch_tool_call(tc.name, tc.input, context))
                for tc in turn.tool_calls
            ]
            turn = provider.send_tool_results(results)
            n_turns += 1
            continue

        if memo_source is None:
            memo_source = turn.text
        try:
            ranked = parse_ranked_output(turn.text)
            final_text = turn.text
            break
        except ValueError as exc:
            if repairs >= max_repairs:
                snippet = (turn.text or "").strip()[:400]
                raise ValueError(
                    f"{exc} | after {repairs} repair attempt(s); the model's final "
                    f"message was: {snippet!r}"
                ) from exc
            repairs += 1
            turn = provider.send_user(_REPAIR)
            n_turns += 1

    memo = _extract_memo(final_text)
    if not memo and memo_source and memo_source != final_text:
        memo = _extract_memo(memo_source)  # JSON was repaired; keep original prose
    return {
        "ranked_output": ranked,
        "design_memo": memo,
        "raw_final_text": final_text,
        "n_turns": n_turns,
    }


def _extract_memo(final_text: str) -> str:
    """Return the prose memo: the final text with the ranked JSON removed."""
    s = strip_code_fences(final_text.strip())
    span = find_json_span(s)
    if span is None:
        return s.strip()
    return (s[: span[0]] + s[span[1] :]).strip()


def dispatch_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one of the four ``primer_core.tools`` functions by name.

    Filesystem paths the model cannot know (``model_path`` for
    ``score_candidate``, ``db_path`` for ``check_specificity``) are injected
    from ``context`` and override anything the model supplied. A missing BLAST
    DB yields a structured error rather than an exception, so the loop
    continues and the model reports the gap (system-prompt rule 6). Real tool
    exceptions are captured as ``{"error": ...}`` for the same reason.

    Args:
        tool_name: One of the four known tool names.
        tool_input: Arguments from the model's tool-use request.
        context: Runtime config with ``model_path`` and ``blast_db``.

    Returns:
        A JSON-serializable result (or ``{"error": ...}``).

    Raises:
        ValueError: If ``tool_name`` is not one of the four known tools.
    """
    if tool_name not in _TOOL_NAMES:
        raise ValueError(f"unknown tool {tool_name!r}; expected one of {sorted(_TOOL_NAMES)}")

    ctx = context or {}
    # Keep only arguments the tool actually accepts (models hallucinate extras),
    # and strip stray whitespace models sometimes inject into sequences.
    allowed = _TOOL_PARAMS.get(tool_name, set())
    args = {k: v for k, v in tool_input.items() if k in allowed}
    for k in list(args):
        if k in _SEQ_ARGS and isinstance(args[k], str):
            args[k] = "".join(args[k].split())

    # Import lazily so this module stays importable without the primer3/vienna
    # backends present at collection time.
    from primer_core.tools import (
        check_specificity,
        design_primers,
        score_dual_head,
        thermo_check,
    )

    try:
        if tool_name == "design_primers":
            # A runtime-pinned design target overrides the model's copy, so it
            # never has to reproduce a long template verbatim.
            if ctx.get("template_seq"):
                args["template_seq"] = ctx["template_seq"]
            return {"candidates": design_primers(**args)}
        if tool_name == "thermo_check":
            return thermo_check(**args)
        if tool_name == "score_candidate":
            # Runtime-injected: the design template (for annealing features) and
            # both head-A artifacts. The model supplies only the primer sequence;
            # score_dual_head returns P(amplify) + predicted efficiency + caveats.
            classifier = ctx.get("classifier_path")
            regressor = ctx.get("regressor_path") or ctx.get("model_path")
            if not (classifier or regressor):
                return {"error": "scoring unavailable: no model artifacts configured"}
            return score_dual_head(
                primer_seq=args["primer_seq"],
                template_seq=ctx.get("template_seq"),
                classifier_path=classifier,
                regressor_path=regressor,
            )
        # check_specificity: local DB by default; NCBI remote API if configured
        # (remote is slow/rate-limited — opt in via context).
        args.pop("db_path", None)  # never trust a model-supplied path
        if ctx.get("blast_exclude"):  # the on-target id, so it isn't self-counted
            args["exclude_target_id"] = ctx["blast_exclude"]
        if ctx.get("blast_remote"):
            return check_specificity(remote=True, database=ctx.get("blast_database", "nt"), **args)
        db = ctx.get("blast_db")
        if not db:
            return {"error": "specificity check unavailable: no BLAST DB configured"}
        return check_specificity(db_path=db, **args)
    except Exception as exc:  # surface, never fabricate around it
        return {"error": f"{tool_name} failed: {exc}"}


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "RankedOutput",
    "dispatch_tool_call",
    "run_agent_loop",
]
