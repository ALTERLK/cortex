"""Tests for the query router and the usage meter."""

from __future__ import annotations

from typing import Any

from cortex.agent.router import QueryRouter
from cortex.llm.base import LLMResponse, LLMUnavailableError, TokenUsage
from cortex.llm.metering import MeteredLLM


class CannedLLM:
    """Returns one fixed reply and remembers the messages it was sent."""

    def __init__(self, reply: str, *, fail: bool = False) -> None:
        self._reply = reply
        self._fail = fail
        self.seen: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], *, tools: Any = None,
             temperature: float = 0.3) -> LLMResponse:
        if self._fail:
            raise LLMUnavailableError("down")
        self.seen = messages
        return LLMResponse(content=self._reply, usage=TokenUsage(30, 8))

    def chat_stream(self, messages: list[dict[str, Any]], *, temperature: float = 0.3):
        yield "par"
        yield "tial"
        yield LLMResponse(content="partial", usage=TokenUsage(11, 4))


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


def test_routes_to_agent() -> None:
    llm = CannedLLM('{"route": "agent", "reason": "compares two features"}')
    route = QueryRouter(llm).route("How do X and Y differ?")
    assert route.mode == "agent"
    assert route.reason == "compares two features"
    assert route.usage == TokenUsage(30, 8)


def test_routes_to_rag() -> None:
    route = QueryRouter(CannedLLM('{"route": "rag", "reason": "single fact"}')).route("What is X?")
    assert route.mode == "rag"


def test_json_wrapped_in_prose_and_thinking_is_parsed() -> None:
    reply = '<thinking>Two documents needed.</thinking>Sure:\n```json\n{"route": "agent", "reason": "x"}\n```'
    assert QueryRouter(CannedLLM(reply)).route("Q?").mode == "agent"


def test_unparseable_output_falls_back_to_rag() -> None:
    route = QueryRouter(CannedLLM("I think the agent would be best here.")).route("Q?")
    assert route.mode == "rag"
    assert route.usage == TokenUsage(30, 8)  # the call was still paid for


def test_unknown_route_value_falls_back_to_rag() -> None:
    assert QueryRouter(CannedLLM('{"route": "search-engine"}')).route("Q?").mode == "rag"


def test_provider_failure_falls_back_to_rag() -> None:
    route = QueryRouter(CannedLLM("", fail=True)).route("Q?")
    assert route.mode == "rag"
    assert route.usage == TokenUsage(0, 0)


def test_history_is_sent_to_the_router() -> None:
    llm = CannedLLM('{"route": "rag", "reason": "follow-up"}')
    QueryRouter(llm).route("And after?", [{"role": "user", "content": "Background tasks?"}])
    assert [m["role"] for m in llm.seen] == ["system", "user", "user"]
    assert llm.seen[-1]["content"] == "Question: And after?"


# ---------------------------------------------------------------------------
# MeteredLLM
# ---------------------------------------------------------------------------


def test_meter_accumulates_chat_usage() -> None:
    meter = MeteredLLM(CannedLLM("ok"))
    meter.chat([{"role": "user", "content": "a"}])
    meter.chat([{"role": "user", "content": "b"}])
    assert (meter.calls, meter.input_tokens, meter.output_tokens) == (2, 60, 16)
    assert meter.usage == TokenUsage(60, 16)


def test_meter_counts_streamed_usage_and_passes_deltas_through() -> None:
    meter = MeteredLLM(CannedLLM("ok"))
    items = list(meter.chat_stream([{"role": "user", "content": "a"}]))
    assert items[:2] == ["par", "tial"]
    assert (meter.calls, meter.input_tokens, meter.output_tokens) == (1, 11, 4)


def test_meter_reset() -> None:
    meter = MeteredLLM(CannedLLM("ok"))
    meter.chat([])
    meter.reset()
    assert (meter.calls, meter.input_tokens, meter.output_tokens) == (0, 0, 0)
