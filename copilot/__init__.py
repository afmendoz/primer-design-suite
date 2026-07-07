"""copilot — the LLM agent differentiator.

Orchestrates the four ``primer_core.tools`` (design, thermo, specificity,
score) to design and rank primers, then writes a prose design memo alongside
an enforced JSON ranked-output contract. The agent never fabricates a
sequence or a number itself — see ``copilot/agent/system_prompt.py``.
"""
