"""Tests for the LLM abstraction layer.

None of these tests touch the network: they cover our own logic (the
interface contract, config loading, error handling), not the provider's
servers. Live connectivity is checked by scripts/check_llm.py.
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

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.3,
    ):
        self.seen_messages = messages
        if self.canned.content:
            yield self.canned.content
        yield self.canned


def test_fake_client_satisfies_protocol() -> None:
    client: LLMClient = FakeLLMClient(LLMResponse(content="hi"))
    response = client.chat([{"role": "user", "content": "hello"}])
    assert response.content == "hi"
    assert not response.wants_tool_call


def test_chat_stream_contract() -> None:
    # Contract: str deltas, then exactly one final LLMResponse.
    client: LLMClient = FakeLLMClient(LLMResponse(content="hi"))
    items = list(client.chat_stream([{"role": "user", "content": "x"}]))
    assert items[:-1] == ["hi"]
    assert isinstance(items[-1], LLMResponse)


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


# ---------------------------------------------------------------------------
# Adapter: models that reject sampling parameters
# ---------------------------------------------------------------------------


class _FakeCompletions:
    """Stands in for client.chat.completions; optionally rejects `temperature`."""

    def __init__(self, *, rejects_temperature: bool, other_error: bool = False) -> None:
        self._rejects = rejects_temperature
        self._other_error = other_error
        self.calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Any:
        import httpx
        import openai

        self.calls.append(params)
        response = httpx.Response(400, request=httpx.Request("POST", "http://test/v1/chat/completions"))
        if self._other_error:
            raise openai.BadRequestError("messages: field required", response=response, body=None)
        if self._rejects and "temperature" in params:
            raise openai.BadRequestError(
                "`temperature` is deprecated for this model.", response=response, body=None)

        message = type("Message", (), {"content": "ok", "tool_calls": None})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        usage = type("Usage", (), {"prompt_tokens": 7, "completion_tokens": 3})()
        return type("Completion", (), {"choices": [choice], "usage": usage})()


def _client_with(completions: _FakeCompletions) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(api_key="k", base_url="http://test/v1", model="m")
    client._client = type("SDK", (), {"chat": type("Chat", (), {"completions": completions})()})()
    return client


def test_adapter_sends_temperature_when_the_model_accepts_it() -> None:
    completions = _FakeCompletions(rejects_temperature=False)
    response = _client_with(completions).chat([{"role": "user", "content": "hi"}], temperature=0.1)
    assert response.content == "ok"
    assert completions.calls == [
        {"model": "m", "temperature": 0.1, "messages": [{"role": "user", "content": "hi"}]}
    ]


def test_adapter_drops_temperature_once_the_model_rejects_it() -> None:
    completions = _FakeCompletions(rejects_temperature=True)
    client = _client_with(completions)

    first = client.chat([{"role": "user", "content": "hi"}], temperature=0.1)
    assert first.usage == TokenUsage(7, 3)
    # One rejected attempt, then the same request without the parameter.
    assert ["temperature" in call for call in completions.calls] == [True, False]

    client.chat([{"role": "user", "content": "again"}], temperature=0.1)
    # The adapter remembers: later calls do not pay for the discovery again.
    assert ["temperature" in call for call in completions.calls] == [True, False, False]


def test_adapter_does_not_swallow_unrelated_bad_requests() -> None:
    from cortex.llm.base import LLMUnavailableError

    completions = _FakeCompletions(rejects_temperature=False, other_error=True)
    with pytest.raises(LLMUnavailableError, match="field required"):
        _client_with(completions).chat([{"role": "user", "content": "hi"}])
    assert len(completions.calls) == 1  # no blind retry
