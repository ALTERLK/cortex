"""Adapter for any OpenAI-compatible LLM endpoint.

Works with DeepSeek, OpenRouter, or any other provider or gateway that
exposes an OpenAI-compatible chat completions API. This adapter's whole
job is:

1. authenticate and send the request,
2. map the provider's response object into our `LLMResponse`.

Nothing outside this file imports the `openai` SDK.

"OpenAI-compatible" means the provider implements OpenAI's HTTP contract —
same endpoints, same JSON shapes — so the `openai` SDK can talk to it with
only `base_url` changed. That is why the provider is selected by
LLM_BASE_URL and LLM_MODEL in .env rather than by code.
"""

import json
from collections.abc import Iterator
from typing import Any

from openai import OpenAI, OpenAIError

from cortex.llm.base import LLMResponse, LLMUnavailableError, TokenUsage, ToolCall


class MissingAPIKeyError(RuntimeError):
    """Raised when a live LLM call is attempted without an API key."""


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise MissingAPIKeyError(
                "LLM_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        # An explicit timeout and bounded retries: without a timeout, one
        # hung provider connection pins a worker thread indefinitely. The
        # SDK retries transient failures (connection errors, 5xx) with backoff.
        self._client = OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=timeout_s, max_retries=max_retries,
        )
        self._model = model

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools or None,  # the API rejects an empty list
                temperature=temperature,
            )
        except OpenAIError as exc:
            raise LLMUnavailableError(f"LLM provider error: {exc}") from exc

        choice = response.choices[0]

        # The provider returns tool arguments as a JSON *string*; it is
        # parsed once here so the rest of the codebase only ever sees a dict.
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in (choice.message.tool_calls or [])
        ]

        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return LLMResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
    ) -> Iterator[str | LLMResponse]:
        # With stream=True the SDK yields chunks, each carrying a small
        # "delta" of the answer. stream_options asks the provider to append
        # one final chunk with token usage; some OpenAI-compatible gateways
        # omit it, in which case usage stays at zero.
        parts: list[str] = []
        usage = TokenUsage(0, 0)
        finish = "stop"

        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            for chunk in stream:
                if chunk.usage:
                    usage = TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                if not chunk.choices:
                    continue  # the usage-only final chunk has no choices
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish = choice.finish_reason
                delta = choice.delta.content if choice.delta else None
                if delta:
                    parts.append(delta)
                    yield delta
        except OpenAIError as exc:
            raise LLMUnavailableError(f"LLM provider error: {exc}") from exc

        yield LLMResponse(content="".join(parts), finish_reason=finish, usage=usage)
