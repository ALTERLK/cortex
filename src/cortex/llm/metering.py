"""Usage metering: count the tokens spent through an LLM client.

A request can involve several LLM calls that the caller never sees directly —
a routing call, a query rewrite, each step of an agent loop. Wrapping the
client once and reading the meter afterwards gives the true total, without
threading a counter through every component.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from cortex.llm.base import LLMClient, LLMResponse, TokenUsage


class MeteredLLM:
    """An LLMClient that forwards every call and accumulates its token usage.

    It satisfies the LLMClient protocol, so it can be passed anywhere a
    client is expected.
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def reset(self) -> None:
        self.input_tokens = self.output_tokens = self.calls = 0

    @property
    def usage(self) -> TokenUsage:
        return TokenUsage(self.input_tokens, self.output_tokens)

    def _record(self, usage: TokenUsage) -> None:
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.calls += 1

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        response = self._inner.chat(messages, tools=tools, temperature=temperature)
        self._record(response.usage)
        return response

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
    ) -> Iterator[str | LLMResponse]:
        for item in self._inner.chat_stream(messages, temperature=temperature):
            if isinstance(item, LLMResponse):
                self._record(item.usage)
            yield item
