"""Tests for the agent loop.

All offline — the FakeLLM is configured to return scripted responses in order,
simulating single-step, multi-step, and max-iteration scenarios.
"""

from __future__ import annotations

from typing import Any

from cortex.agent.loop import AgentLoop, _assistant_message, _tool_result_message
from cortex.agent.tools import _format_search_results
from cortex.ingest.store import SearchResult
from cortex.llm.base import LLMResponse, TokenUsage, ToolCall

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ScriptedLLM:
    """Returns responses from a pre-set script, one per chat() call."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.call_count = 0
        self.all_messages: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        self.all_messages.append(messages)
        self.call_count += 1
        if not self._responses:
            return LLMResponse(content="fallback answer")
        return self._responses.pop(0)


class FakeExecutor:
    """Always returns a fixed result string for any tool call."""

    def __init__(self, result: str = "Fake search result.") -> None:
        self._result = result
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return self._result


def make_tool_response(query: str = "test query") -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="tc1", name="search_knowledge_base", arguments={"query": query})],
        finish_reason="tool_calls",
        usage=TokenUsage(50, 10),
    )


def make_final_response(answer: str = "Final answer [1].") -> LLMResponse:
    return LLMResponse(content=answer, finish_reason="stop", usage=TokenUsage(80, 20))


# ---------------------------------------------------------------------------
# AgentLoop tests
# ---------------------------------------------------------------------------


def test_single_iteration_no_tool_call() -> None:
    llm = ScriptedLLM([make_final_response("Direct answer.")])
    result = AgentLoop(llm, FakeExecutor()).run("Simple question?")
    assert result.answer == "Direct answer."
    assert result.iterations == 1
    assert result.tool_calls == []


def test_tool_call_then_answer() -> None:
    executor = FakeExecutor("Search result text.")
    llm = ScriptedLLM([make_tool_response("rag definition"), make_final_response("Answer [1].")])
    result = AgentLoop(llm, executor).run("What is RAG?")

    assert result.answer == "Answer [1]."
    assert result.iterations == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_knowledge_base"
    assert result.tool_calls[0].arguments == {"query": "rag definition"}


def test_multiple_tool_calls_accumulate() -> None:
    executor = FakeExecutor()
    llm = ScriptedLLM([
        make_tool_response("first query"),
        make_tool_response("second query"),
        make_final_response("Comprehensive answer [1][2]."),
    ])
    result = AgentLoop(llm, executor).run("Complex multi-part question?")

    assert result.iterations == 3
    assert len(result.tool_calls) == 2
    queries = [tc.arguments["query"] for tc in result.tool_calls]
    assert queries == ["first query", "second query"]


def test_max_iterations_guard() -> None:
    # LLM keeps asking for tool calls; loop must stop at max_iterations.
    llm = ScriptedLLM([make_tool_response() for _ in range(20)])
    result = AgentLoop(llm, FakeExecutor(), max_iterations=3).run("Question?")
    assert result.iterations == 3
    assert "maximum" in result.answer.lower()


def test_tool_result_appended_to_messages() -> None:
    executor = FakeExecutor("My result")
    llm = ScriptedLLM([make_tool_response(), make_final_response()])
    AgentLoop(llm, executor).run("Question?")

    # Second LLM call should have received the tool result in messages.
    second_call_messages = llm.all_messages[1]
    roles = [m["role"] for m in second_call_messages]
    assert "tool" in roles


def test_token_usage_accumulates() -> None:
    llm = ScriptedLLM([make_tool_response(), make_final_response()])
    result = AgentLoop(llm, FakeExecutor()).run("Q?")
    # First call: 50 in / 10 out. Second call: 80 in / 20 out.
    assert result.usage.input_tokens == 130
    assert result.usage.output_tokens == 30


# ---------------------------------------------------------------------------
# Message helper tests
# ---------------------------------------------------------------------------


def test_assistant_message_shape() -> None:
    tc = ToolCall(id="id1", name="search_knowledge_base", arguments={"query": "x"})
    msg = _assistant_message("thinking…", [tc])
    assert msg["role"] == "assistant"
    assert msg["tool_calls"][0]["id"] == "id1"
    assert msg["tool_calls"][0]["type"] == "function"
    assert '"query": "x"' in msg["tool_calls"][0]["function"]["arguments"]


def test_tool_result_message_shape() -> None:
    msg = _tool_result_message("id1", "result text")
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "id1"
    assert msg["content"] == "result text"


# ---------------------------------------------------------------------------
# _format_search_results
# ---------------------------------------------------------------------------


def test_format_search_results_numbering() -> None:
    results = [
        SearchResult(score=0.9, text="Alpha", source="a.md", chunk_index=0),
        SearchResult(score=0.8, text="Beta", source="b.md", chunk_index=1),
    ]
    formatted = _format_search_results(results)
    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "Alpha" in formatted
    assert "a.md" in formatted
