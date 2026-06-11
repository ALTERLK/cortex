"""DeepSeek adapter for the LLMClient protocol.

DeepSeek's API is OpenAI-compatible, so we reuse the official `openai`
SDK and only change `base_url`. This adapter's whole job is:

1. authenticate and send the request,
2. map the provider's response object into our `LLMResponse`.

Nothing outside this file should import the `openai` SDK.
"""

import json
from typing import Any

from openai import OpenAI

from cortex.llm.base import LLMResponse, TokenUsage, ToolCall


class MissingAPIKeyError(RuntimeError):
    """Raised when a live LLM call is attempted without an API key."""


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        if not api_key:
            raise MissingAPIKeyError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and "
                "fill in your key from https://platform.deepseek.com/api_keys"
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
