"""Tests for the LLM abstraction layer.

NOTE (learning): none of these tests touch the network. We test *our*
logic (the interface contract, config loading, error handling), not
the LLM provider's servers. Live connectivity is covered by scripts/smoke_test.py.
"""

from typing import Any

import pytest

from cortex.config import Settings
from cortex.llm.base import LLMClient, LLMResponse, TokenUsage, ToolCall
from cortex.llm.openai_compat import MissingAPIKeyError, OpenAICompatibleClient


class FakeLLMClient:
    """A canned-response stand-in that satisfies the LLMClient protocol.

    This is the payoff of using `Protocol`: tests (and later, eval runs)
    can swap in a fake with zero changes to production code.
    """

    def __init__(self, canned: LLMResponse) -> None:
        self.canned = canned
        self.seen_messages: list[dict[str, Any]] | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        self.seen_messages = messages
        return self.canned


def test_fake_client_satisfies_protocol() -> None:
    client: LLMClient = FakeLLMClient(LLMResponse(content="hi"))
    response = client.chat([{"role": "user", "content": "hello"}])
    assert response.content == "hi"
    assert not response.wants_tool_call


def test_response_with_tool_calls() -> None:
    response = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="1", name="search", arguments={"query": "x"})],
        finish_reason="tool_calls",
    )
    assert response.wants_tool_call
    assert response.tool_calls[0].arguments == {"query": "x"}


def test_token_usage_total() -> None:
    assert TokenUsage(input_tokens=10, output_tokens=5).total_tokens == 15


def test_client_requires_api_key() -> None:
    with pytest.raises(MissingAPIKeyError):
        OpenAICompatibleClient(api_key="", base_url="https://api.deepseek.com", model="deepseek-chat")


def test_settings_have_sane_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ignore any local .env so the test is reproducible on any machine.
    settings = Settings(_env_file=None)
    assert settings.llm_base_url == "https://api.deepseek.com"
    assert settings.llm_model == "deepseek-chat"
    assert settings.llm_api_key == ""
