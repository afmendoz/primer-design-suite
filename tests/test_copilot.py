"""Copilot agent tests — real orchestration with a fake provider, no API.

The :class:`ScriptedProvider` implements the ``LLMProvider`` contract and walks
a real multi-turn conversation whose tool calls are dispatched to the *actual*
``primer_core`` tools (Primer3 design, thermodynamics, and the trained-model
scorer). No LLM SDK and no API key are involved — the provider seam is exactly
what makes the agent testable offline.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from copilot.agent.loop import dispatch_tool_call, run_agent_loop
from copilot.agent.providers import AssistantTurn, ToolCall, get_provider, to_openai_tools
from copilot.agent.schemas import RankedOutput, parse_ranked_output

TEMPLATE = (
    "CGATGCTAGCTAGCTAGGCTAGCATCGATCGTAGCTAGCTAGCATCGATCGATCGTAGCTAGCTAGCATCGATCG"
    "ATCGTACGTAGCATCGATCGATCGATCGTAGCTAGCATCGGCTAGCATCGATCGATCGTAGCTAGCATCGATCGA"
    "TCGTAGCTAGCTAGCATCGATCGATGCTAGCTAGCATCGATCGATCGTAGCTAGCATCGATCGATCGTAGCTAGC"
    "ATCGATCGATCGTAGCTAGCATCGATCG"
)

FEATURES = ["gc_content", "length", "tm"]


@pytest.fixture
def model_path(tmp_path):
    """A tiny real joblib scorer matching score_candidate's artifact contract."""
    x = np.array([[0.4, 18, 58.0], [0.5, 20, 60.0], [0.6, 22, 62.0]])
    y = np.array([0.3, 0.6, 0.8])
    lr = LinearRegression().fit(x, y)
    path = tmp_path / "model.joblib"
    joblib.dump({"model": lr, "feature_names": FEATURES}, path)
    return str(path)


class ScriptedProvider:
    """Fake LLMProvider: design -> thermo_check -> score -> final JSON.

    Later turns use values captured from earlier *real* tool results, so the
    final ranked JSON is built from genuine Primer3 sequences and a genuine
    predicted efficiency.
    """

    def __init__(self):
        self._sent = 0
        self.fwd = self.rev = ""
        self.tm = 60.0
        self.eff = 0.5

    def start_conversation(self, system, tools):
        self.system, self.tools = system, tools

    def send_user(self, text):
        return AssistantTurn(
            text="",
            tool_calls=[ToolCall("c1", "design_primers", {"template_seq": TEMPLATE})],
        )

    def send_tool_results(self, results):
        self._sent += 1
        content = results[0].content
        if self._sent == 1:  # design done
            cand = content["candidates"][0]
            self.fwd, self.rev = cand["left_sequence"], cand["right_sequence"]
            return AssistantTurn(
                text="",
                tool_calls=[ToolCall("c2", "thermo_check", {"primer_seq": self.fwd})],
            )
        if self._sent == 2:  # thermo done
            self.tm = content.get("tm", 60.0)
            return AssistantTurn(
                text="",
                tool_calls=[ToolCall("c3", "score_candidate", {"primer_seq": self.fwd})],
            )
        if self._sent == 3:  # score done
            self.eff = content.get("predicted_efficiency", 0.5)
            payload = {
                "design_goal": "test qPCR primers",
                "candidates": [
                    {
                        "rank": 1,
                        "forward_primer": self.fwd,
                        "reverse_primer": self.rev,
                        "tm_forward": self.tm,
                        "tm_reverse": self.tm,
                        "off_target_hit_count": 0,
                        "predicted_efficiency": self.eff,
                        "risk_flags": [],
                    }
                ],
            }
            text = (
                "Ranked design follows.\n\n"
                + json.dumps(payload)
                + "\n\nMemo: candidate 1 is best."
            )
            return AssistantTurn(text=text, is_final=True)
        raise AssertionError("unexpected extra turn")


class NeverFinalProvider:
    """Always requests a (cheap, no-op) tool — used to trip max_turns."""

    def start_conversation(self, system, tools):
        pass

    def _turn(self):
        return AssistantTurn(
            text="",
            tool_calls=[ToolCall("x", "check_specificity", {"primer_seq": "ACGTACGTACGTACGTACGT"})],
        )

    def send_user(self, text):
        return self._turn()

    def send_tool_results(self, results):
        return self._turn()


def test_agent_loop_end_to_end(model_path):
    result = run_agent_loop(
        "design qPCR primers",
        provider=ScriptedProvider(),
        context={"model_path": model_path, "blast_db": None},
    )
    ranked = result["ranked_output"]
    assert isinstance(ranked, RankedOutput)
    assert len(ranked.candidates) >= 1
    c = ranked.candidates[0]
    assert set(c.forward_primer) <= set("ACGT") and c.forward_primer
    assert result["design_memo"]
    assert "Memo: candidate 1 is best." in result["design_memo"]
    assert result["n_turns"] == 4


def test_dispatch_design_primers_is_real():
    out = dispatch_tool_call("design_primers", {"template_seq": TEMPLATE})
    assert out["candidates"], "primer3 should return candidates"
    seq = out["candidates"][0]["left_sequence"]
    assert seq and set(seq) <= set("ACGT")


def test_dispatch_design_injects_context_template():
    # A runtime-pinned template overrides the model's arg (here too short to design).
    out = dispatch_tool_call(
        "design_primers", {"template_seq": "ACGT"}, context={"template_seq": TEMPLATE}
    )
    assert out["candidates"]


def test_dispatch_specificity_without_db_is_graceful():
    out = dispatch_tool_call(
        "check_specificity", {"primer_seq": "ACGTACGTACGTACGTACGT"}, context={"blast_db": None}
    )
    assert "error" in out and "BLAST DB" in out["error"]


def test_dispatch_unknown_tool_raises():
    with pytest.raises(ValueError):
        dispatch_tool_call("bogus_tool", {})


def test_dispatch_drops_unknown_kwargs_and_strips_whitespace():
    # A hallucinated kwarg must be dropped, and whitespace inside a sequence
    # normalized, rather than crashing the tool.
    out = dispatch_tool_call(
        "thermo_check",
        {"primer_seq": "ACGTACGT ACGTACGT ACGT", "bogus_arg": 123},
    )
    assert "tm" in out and "error" not in out


def test_max_turns_raises():
    with pytest.raises(RuntimeError):
        run_agent_loop("x", provider=NeverFinalProvider(), max_turns=3, context={"blast_db": None})


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError):
        get_provider("gemini", model="whatever")


def test_get_provider_forwards_api_key():
    # A UI-provided key reaches both adapters without any SDK/network call.
    assert get_provider("anthropic", model="claude-x", api_key="sk-a")._api_key == "sk-a"
    assert get_provider("openai", model="gpt-x", api_key="sk-o")._api_key == "sk-o"


def test_app_clean_dna_strips_fasta_and_whitespace():
    from copilot.app.main import _clean_dna  # module-level, no streamlit needed

    raw = ">KC713938|IGHV1-2*02 header\nATGG ACTG\nacgt\n"
    assert _clean_dna(raw) == "ATGGACTGACGT"


class _ProseThenJSON:
    """Final turn is prose-only first; the repair nudge then yields valid JSON."""

    def __init__(self, ever_json=True):
        self.calls = 0
        self.ever_json = ever_json

    def start_conversation(self, system, tools):
        pass

    def send_user(self, text):
        self.calls += 1
        if self.calls == 1:
            return AssistantTurn(text="Here is my prose answer with no JSON.", is_final=True)
        if self.ever_json:
            return AssistantTurn(text=json.dumps(_VALID), is_final=True)
        return AssistantTurn(text="still just prose", is_final=True)

    def send_tool_results(self, results):
        raise AssertionError("no tool calls expected")


def test_json_repair_recovers_and_keeps_prose_memo():
    result = run_agent_loop("goal", provider=_ProseThenJSON(), context={"blast_db": None})
    assert result["ranked_output"].candidates[0].rank == 1
    assert "prose answer" in result["design_memo"]  # original prose preserved


def test_json_repair_gives_up_after_max_repairs():
    with pytest.raises(ValueError):
        run_agent_loop("goal", provider=_ProseThenJSON(ever_json=False), max_repairs=1)


# --- parse_ranked_output ------------------------------------------------

_VALID = {
    "design_goal": "g",
    "candidates": [
        {
            "rank": 1,
            "forward_primer": "ACGT",
            "reverse_primer": "TGCA",
            "tm_forward": 60.0,
            "tm_reverse": 60.0,
            "off_target_hit_count": 0,
            "predicted_efficiency": 0.7,
            "risk_flags": [],
        }
    ],
}


def test_parse_clean_json():
    out = parse_ranked_output(json.dumps(_VALID))
    assert out.candidates[0].rank == 1


def test_parse_fenced_json():
    out = parse_ranked_output("```json\n" + json.dumps(_VALID) + "\n```")
    assert out.design_goal == "g"


def test_parse_prose_wrapped_json():
    out = parse_ranked_output("Here you go:\n" + json.dumps(_VALID) + "\nThanks!")
    assert out.candidates[0].predicted_efficiency == 0.7


def test_parse_malformed_raises():
    with pytest.raises(ValueError):
        parse_ranked_output("not json at all")


# --- pure translation ---------------------------------------------------


def test_to_openai_tools_translation():
    schemas = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    out = to_openai_tools(schemas)
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "t"
    assert out[0]["function"]["parameters"] == {"type": "object"}
