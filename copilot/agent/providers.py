"""Provider-agnostic LLM adapter layer.

The agent loop (``copilot.agent.loop``) is deliberately provider-neutral: it
speaks only in the normalized types defined here (:class:`AssistantTurn`,
:class:`ToolCall`, :class:`ToolResult`) and drives an :class:`LLMProvider`.
Each concrete adapter owns *all* of its provider's wire format — how tools are
declared, how a completion is requested, how tool-use requests are read out of
a response, and how tool results are fed back — so swapping the underlying LLM
is a matter of choosing a different adapter, not touching the loop or the
safety rules.

The default is :class:`OpenAIProvider`, which talks the OpenAI chat-completions
"function calling" dialect and accepts a ``base_url``, so it also drives any
OpenAI-compatible endpoint (Ollama, vLLM, together.ai, ...). SDK imports are
lazy: this module imports cleanly with no provider SDK installed, which is what
lets the test suite drive a fake provider with no dependency and no API key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """A single tool-use request emitted by the model, normalized."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """The result of executing one :class:`ToolCall`, to feed back."""

    tool_call_id: str
    content: dict[str, Any]


@dataclass
class AssistantTurn:
    """One normalized assistant response.

    Attributes:
        text: Any assistant prose in this turn (may be empty).
        tool_calls: Tool-use requests the model made this turn.
        is_final: True when the model stopped without requesting tools, i.e.
            this turn carries the final answer.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    is_final: bool = False


class LLMProvider(Protocol):
    """Stateful conversation adapter for one LLM backend.

    The adapter owns its native message history; the loop only ever hands it a
    user message or a batch of tool results and reads back an
    :class:`AssistantTurn`.
    """

    def start_conversation(self, system: str, tools: list[dict[str, Any]]) -> None:
        """Begin a fresh conversation with a system prompt and tool schemas."""
        ...

    def send_user(self, text: str) -> AssistantTurn:
        """Send a user message and return the model's next turn."""
        ...

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        """Send executed tool results and return the model's next turn."""
        ...


# --- pure tool-schema translation (no SDK import needed) -----------------


def to_openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the neutral tool schemas to OpenAI function-tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


def to_anthropic_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic already uses ``{name, description, input_schema}`` natively."""
    return list(schemas)


# --- concrete adapters ---------------------------------------------------


class OpenAIProvider:
    """Default adapter: OpenAI chat-completions function-calling dialect.

    Works against api.openai.com or any OpenAI-compatible endpoint via
    ``base_url`` (e.g. a local Ollama/vLLM server), which is how this project
    supports "any LLM". The ``openai`` SDK is imported lazily and only when no
    client is injected.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = client
        self.messages: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # lazy: only needed for the real path

            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def start_conversation(self, system: str, tools: list[dict[str, Any]]) -> None:
        self.messages = [{"role": "system", "content": system}]
        self.tools = to_openai_tools(tools)

    def _create(self) -> AssistantTurn:
        resp = self.client.chat.completions.create(
            model=self.model, messages=self.messages, tools=self.tools
        )
        msg = resp.choices[0].message
        tool_calls = list(msg.tool_calls or [])
        if tool_calls:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            return AssistantTurn(
                text=msg.content or "",
                tool_calls=[
                    ToolCall(tc.id, tc.function.name, json.loads(tc.function.arguments or "{}"))
                    for tc in tool_calls
                ],
                is_final=False,
            )
        self.messages.append({"role": "assistant", "content": msg.content or ""})
        return AssistantTurn(text=msg.content or "", tool_calls=[], is_final=True)

    def send_user(self, text: str) -> AssistantTurn:
        self.messages.append({"role": "user", "content": text})
        return self._create()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        for r in results:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": r.tool_call_id,
                    "content": json.dumps(r.content),
                }
            )
        return self._create()


class AnthropicProvider:
    """Adapter for the Anthropic Messages API (tool_use / tool_result blocks).

    Selectable via config; not the default. The SDK import is lazy and is never
    exercised by the test suite.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = client
        self.system = ""
        self.tools: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # lazy: only needed for the real path

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def start_conversation(self, system: str, tools: list[dict[str, Any]]) -> None:
        self.system = system
        self.tools = to_anthropic_tools(tools)
        self.messages = []

    def _create(self) -> AssistantTurn:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=self.tools,
            messages=self.messages,
        )
        self.messages.append({"role": "assistant", "content": resp.content})
        tool_calls = [
            ToolCall(b.id, b.name, dict(b.input))
            for b in resp.content
            if getattr(b, "type", None) == "tool_use"
        ]
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return AssistantTurn(
            text=text, tool_calls=tool_calls, is_final=resp.stop_reason != "tool_use"
        )

    def send_user(self, text: str) -> AssistantTurn:
        self.messages.append({"role": "user", "content": text})
        return self._create()

    def send_tool_results(self, results: list[ToolResult]) -> AssistantTurn:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.tool_call_id,
                        "content": json.dumps(r.content),
                    }
                    for r in results
                ],
            }
        )
        return self._create()


def get_provider(name: str = "openai", *, model: str, **kwargs: Any) -> LLMProvider:
    """Construct a provider adapter by name.

    Args:
        name: ``"openai"`` (default) or ``"anthropic"``.
        model: Model id for the backend.
        **kwargs: Adapter-specific options (e.g. ``base_url``, ``api_key`` for
            OpenAI, or an injected ``client`` for either).

    Raises:
        ValueError: If ``name`` is not a known provider.
    """
    name = name.lower()
    if name == "openai":
        return OpenAIProvider(model=model, **kwargs)
    if name == "anthropic":
        return AnthropicProvider(model=model, **kwargs)
    raise ValueError(f"unknown provider {name!r}; expected 'openai' or 'anthropic'")
