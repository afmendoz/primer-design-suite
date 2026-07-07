"""Tool-use agent loop skeleton.

The LLM orchestrates and interprets tool results; it never fabricates a
sequence or a number (see ``copilot/agent/system_prompt.py``). This module
drives the Anthropic tool-use loop: send the conversation + tool schemas,
execute whichever of the four ``primer_core.tools`` functions the model
calls, feed results back, and repeat until the model emits the final
ranked-output JSON + prose memo.

Uses the ``anthropic`` SDK, an optional dependency (see the ``copilot``
extra in ``pyproject.toml``), so the import is deferred into functions
rather than performed at module scope — this module must remain importable
without ``anthropic`` installed.
"""

from __future__ import annotations

from typing import Any

# Default model id for the copilot agent. claude-fable-5 is a reasonable
# default; override via config/env as needed.
DEFAULT_MODEL = "claude-fable-5"


def run_agent_loop(
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_turns: int = 10,
) -> dict[str, Any]:
    """Drive the tool-use loop for one user design request.

    Requires the ``anthropic`` package, imported lazily inside this function
    so the module remains importable without the optional ``copilot`` extra
    installed.

    Args:
        user_message: Natural-language design goal + constraints from the
            user.
        model: Anthropic model id to use.
        max_turns: Maximum number of tool-use round-trips before aborting.

    Returns:
        A dict with keys ``ranked_output`` (see
        ``copilot.agent.schemas.RankedOutput``) and ``design_memo`` (prose).

    Raises:
        ImportError: If ``anthropic`` is not installed.
        RuntimeError: If ``max_turns`` is exceeded without a final answer.
    """
    raise NotImplementedError


def dispatch_tool_call(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Execute one of the four ``primer_core.tools`` functions by name.

    Args:
        tool_name: One of ``"design_primers"``, ``"thermo_check"``,
            ``"check_specificity"``, ``"score_candidate"``.
        tool_input: Keyword arguments for the tool call, as produced by the
            model's tool-use request.

    Returns:
        The tool's return value, JSON-serializable.

    Raises:
        ValueError: If ``tool_name`` is not one of the four known tools.
    """
    raise NotImplementedError
