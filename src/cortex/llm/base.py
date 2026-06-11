"""Provider-agnostic LLM interface.

Why this file exists
--------------------
Every LLM vendor (DeepSeek, OpenAI, Anthropic, ...) has a slightly
different SDK and response shape. If application code used a vendor SDK
directly, switching providers would mean rewriting every call site.

Instead, the whole project programs against two things defined here:

1. `LLMClient` — the *protocol* (interface) any provider must implement.
2. `LLMResponse` — the *normalized* response every provider maps into.

Requests use the OpenAI chat-message format (`[{"role": ..., "content":
...}]`) because it is the de-facto industry standard that DeepSeek and
most providers accept natively; only responses need normalizing.

NOTE (learning): this is the "ports and adapters" (hexagonal) pattern —
`base.py` is the port, `deepseek.py` is one adapter.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one API call — the raw material for cost tracking."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the model (used from M4 onwards)."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """Normalized result of a chat completion, whatever the provider."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # "stop" = model finished its answer; "tool_calls" = model wants us to
    # run a tool and report back. The agent loop (M4) branches on this.
    finish_reason: str = "stop"
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))

    @property
    def wants_tool_call(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient(Protocol):
    """What every LLM provider adapter must look like.

    NOTE (learning): `Protocol` means structural typing — any class with a
    matching `chat` method satisfies this interface, no inheritance needed.
    That also makes faking it in tests trivial (see tests/test_llm_base.py).
    """

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Send a conversation, get one normalized response back.

        Args:
            messages: OpenAI-format chat messages, e.g.
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."}]
            tools: optional JSON-schema tool definitions (OpenAI format).
            temperature: sampling randomness; low = more deterministic.
        """
        ...

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
    ) -> Iterator[str | LLMResponse]:
        """Stream a chat completion: yield text deltas, then one LLMResponse.

        Contract: every yielded item is a `str` delta except the LAST one,
        which is the complete normalized `LLMResponse` (full content + usage).
        Callers display the deltas and read usage from the final item.

        NOTE (learning): no `tools` parameter on purpose — streaming
        tool-call fragments is a protocol rabbit hole. The agent loop uses
        non-streaming chat() for its iterations; streaming is for final
        answer generation only.
        """
        ...
