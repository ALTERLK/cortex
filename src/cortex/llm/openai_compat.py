"""Adapter for any OpenAI-compatible LLM endpoint.

Works with DeepSeek, Claude via 4SAPI, OpenRouter, or any provider
that exposes an OpenAI-compatible chat completions API. This adapter's
whole job is:

1. authenticate and send the request,
2. map the provider's response object into our `LLMResponse`.

Nothing outside this file should import the `openai` SDK.

NOTE (learning): "OpenAI-compatible" means the provider copied OpenAI's
HTTP API contract (same endpoints, same JSON shapes). The `openai` Python
SDK lets you point it at a different server with just `base_url=...` — so
the exact same code works for DeepSeek, Claude via 4SAPI, or GPT. That
is the whole reason we set LLM_BASE_URL in .env instead of hardcoding it.
"""

import json
from typing import Any

from openai import OpenAI

from cortex.llm.base import LLMResponse, TokenUsage, ToolCall


class MissingAPIKeyError(RuntimeError):
    """Raised when a live LLM call is attempted without an API key."""


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        if not api_key:
            raise MissingAPIKeyError(
                "LLM_API_KEY is not set. Copy .env.example to .env and fill in your key."
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools or None,  # the API rejects an empty list
            temperature=temperature,
        )

        choice = response.choices[0]

        # NOTE (learning): the provider returns tool arguments as a JSON
        # *string*; we parse it once here so the rest of the codebase only
        # ever sees a dict. Centralizing quirks like this is exactly what
        # an adapter is for.
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
