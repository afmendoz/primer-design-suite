"""The four tools shared by the copilot agent and plain scripts.

By project convention, the LLM never writes a sequence or asserts a number itself —
every sequence and every quantitative claim comes back from one of these
four tool functions. Both ``copilot/agent`` and standalone scripts import
from here; tool logic is never reimplemented inside the agent.
"""

from __future__ import annotations

from primer_core.tools.design import design_primers
from primer_core.tools.score import score_candidate, score_dual_head
from primer_core.tools.specificity import check_specificity
from primer_core.tools.thermo import thermo_check

__all__ = [
    "design_primers",
    "check_specificity",
    "thermo_check",
    "score_candidate",
    "score_dual_head",
]
