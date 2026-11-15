"""Tests for the agent loop and its tools.

All offline — the scripted LLM returns pre-set responses in order, so
single-step, multi-step, limit, and compaction scenarios are deterministic.
"""

from __future__ import annotations

from typing import Any

from cortex.agent.loop import (
    AgentLoop,
    AgentResponse,
    ToolCallRecord,
    _assistant_message,
    _tool_result_message,
)
from cortex.agent.tools import RunContext, ToolExecutor, format_passages
from cortex.agent.trace import Span
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
        # Snapshot: the loop mutates message dicts in place when compacting.
        self.all_messages.append([dict(m) for m in messages])
        self.call_count += 1
        if not self._responses:
            return LLMResponse(content="fallback answer")
        return self._responses.pop(0)


class FakeExecutor:
    """Always returns a fixed result string for any tool call."""

    schemas: list[dict] = []

    def __init__(self, result: str = "Fake search result.") -> None:
        self._result = result
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name: str, arguments: dict, ctx: RunContext | None = None) -> str:
        self.calls.append((name, arguments))
        return self._result


class FakeRetriever:
    """Returns a different passage per query, so numbering can be checked."""

    def __init__(self, by_query: dict[str, list[SearchResult]]) -> None:
        self._by_query = by_query

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        return self._by_query.get(query, [])


class FakeStore:
    def __init__(self, chunks: list[SearchResult]) -> None:
        self._chunks = chunks

    def get_chunks(self, source: str, first: int, last: int) -> list[SearchResult]:
        return [c for c in self._chunks if c.source == source and first <= c.chunk_index <= last]

    def list_sources(self) -> list[str]:
        return sorted({c.source for c in self._chunks})


def passage(source: str, index: int, text: str = "text") -> SearchResult:
    return SearchResult(score=0.9, text=text, source=source, chunk_index=index)


def make_tool_response(query: str = "test query", *, call_id: str = "tc1",
                       input_tokens: int = 50) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id=call_id, name="search_knowledge_base", arguments={"query": query})],
        finish_reason="tool_calls",
        usage=TokenUsage(input_tokens, 10),
    )


def make_final_response(answer: str = "Final answer [1].") -> LLMResponse:
    return LLMResponse(content=answer, finish_reason="stop", usage=TokenUsage(80, 20))


# ---------------------------------------------------------------------------
# AgentLoop: basic control flow
# ---------------------------------------------------------------------------


def test_single_iteration_no_tool_call() -> None:
    llm = ScriptedLLM([make_final_response("Direct answer.")])
    result = AgentLoop(llm, FakeExecutor()).run("Simple question?")
    assert result.answer == "Direct answer."
    assert result.iterations == 1
    assert result.tool_calls == []
    assert result.stop_reason == "answered"


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
    llm = ScriptedLLM([
        make_tool_response("first query"),
        make_tool_response("second query"),
        make_final_response("Comprehensive answer [1][2]."),
    ])
    result = AgentLoop(llm, FakeExecutor()).run("Complex multi-part question?")

    assert result.iterations == 3
    assert [tc.arguments["query"] for tc in result.tool_calls] == ["first query", "second query"]


def test_tool_result_appended_to_messages() -> None:
    llm = ScriptedLLM([make_tool_response(), make_final_response()])
    AgentLoop(llm, FakeExecutor("My result")).run("Question?")

    # The second LLM call must see the assistant tool_calls turn and its result.
    roles = [m["role"] for m in llm.all_messages[1]]
    assert roles[-2:] == ["assistant", "tool"]
    assert llm.all_messages[1][-1]["content"] == "My result"


def test_history_is_placed_between_system_prompt_and_question() -> None:
    llm = ScriptedLLM([make_final_response()])
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    AgentLoop(llm, FakeExecutor()).run("Now?", history)

    roles = [m["role"] for m in llm.all_messages[0]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.all_messages[0][-1]["content"] == "Now?"


def test_token_usage_accumulates() -> None:
    llm = ScriptedLLM([make_tool_response(), make_final_response()])
    result = AgentLoop(llm, FakeExecutor()).run("Q?")
    # First call: 50 in / 10 out. Second call: 80 in / 20 out.
    assert result.usage.input_tokens == 130
    assert result.usage.output_tokens == 30


# ---------------------------------------------------------------------------
# AgentLoop: limits
# ---------------------------------------------------------------------------


def test_max_iterations_guard() -> None:
    # The LLM keeps asking for tools; the loop must stop at max_iterations.
    llm = ScriptedLLM([make_tool_response() for _ in range(20)])
    result = AgentLoop(llm, FakeExecutor(), max_iterations=3).run("Question?")
    assert result.iterations == 3
    assert llm.call_count == 3
    assert result.stop_reason == "max_iterations"
    assert "maximum" in result.answer.lower()


def test_last_turn_asks_for_a_final_answer() -> None:
    llm = ScriptedLLM([make_tool_response(), make_final_response("Best effort [1].")])
    result = AgentLoop(llm, FakeExecutor(), max_iterations=2).run("Question?")

    # On the final allowed turn the model is told to stop calling tools…
    assert "Do not call any more tools" in llm.all_messages[1][-1]["content"]
    # …and an answer it gives there is a normal answer, not a failure.
    assert result.answer == "Best effort [1]."
    assert result.stop_reason == "answered"


def test_token_budget_ends_the_run() -> None:
    # Each tool turn spends 60 tokens; a 100-token budget is gone after two.
    llm = ScriptedLLM([make_tool_response() for _ in range(20)])
    result = AgentLoop(llm, FakeExecutor(), max_iterations=10, token_budget=100).run("Q?")

    assert result.iterations == 3  # two tool turns, then the forced final turn
    assert result.stop_reason == "token_budget"
    assert "token budget" in result.answer


# ---------------------------------------------------------------------------
# AgentLoop: context compaction
# ---------------------------------------------------------------------------


def test_old_tool_results_are_compacted_when_prompt_grows() -> None:
    retriever = FakeRetriever({
        "q1": [passage("a.md", 0, "long text one")],
        "q2": [passage("b.md", 0, "long text two")],
        "q3": [passage("c.md", 0, "long text three")],
    })
    llm = ScriptedLLM([
        make_tool_response("q1", call_id="t1"),
        make_tool_response("q2", call_id="t2"),
        # This turn reports a prompt above the threshold → compaction runs.
        make_tool_response("q3", call_id="t3", input_tokens=5000),
        make_final_response("Answer [1][3]."),
    ])
    result = AgentLoop(llm, ToolExecutor(retriever), compact_at=1000).run("Q?")

    tool_contents = [m["content"] for m in llm.all_messages[3] if m["role"] == "tool"]
    # The oldest result is replaced by a stub that keeps its citation…
    assert "long text one" not in tool_contents[0]
    assert "[1] a.md (chunk 0)" in tool_contents[0]
    # …while the two most recent results are untouched.
    assert "long text two" in tool_contents[1]
    assert "long text three" in tool_contents[2]
    assert result.compactions == 1
    # Compaction never drops a source: citations in the answer still resolve.
    assert [s.source for s in result.sources] == ["a.md", "b.md", "c.md"]


def test_no_compaction_below_threshold() -> None:
    retriever = FakeRetriever({
        "q1": [passage("a.md", 0, "kept text")],
        "q2": [passage("b.md", 0, "kept text")],
        "q3": [passage("c.md", 0, "kept text")],
    })
    llm = ScriptedLLM([
        make_tool_response("q1", call_id="t1"),
        make_tool_response("q2", call_id="t2"),
        make_tool_response("q3", call_id="t3"),
        make_final_response(),
    ])
    result = AgentLoop(llm, ToolExecutor(retriever), compact_at=10_000).run("Q?")
    assert result.compactions == 0
    assert all("kept text" in m["content"] for m in llm.all_messages[3] if m["role"] == "tool")


# ---------------------------------------------------------------------------
# AgentLoop: events, trace, and citation numbering
# ---------------------------------------------------------------------------


def test_run_events_yields_spans_tool_calls_then_response() -> None:
    llm = ScriptedLLM([make_tool_response(), make_final_response()])
    events = list(AgentLoop(llm, FakeExecutor()).run_events("Q?"))

    kinds = [type(e).__name__ for e in events]
    assert kinds == ["Span", "ToolCallRecord", "Span", "AgentResponse"]
    assert isinstance(events[0], Span) and events[0].kind == "llm"
    assert events[0].input_tokens == 50
    assert isinstance(events[-1], AgentResponse)
    # The final response carries the whole trace: 2 LLM calls + 1 tool call.
    assert [s.kind for s in events[-1].spans] == ["llm", "tool", "llm"]


def test_citation_numbers_are_stable_across_searches() -> None:
    shared = passage("shared.md", 3)
    retriever = FakeRetriever({
        "q1": [passage("a.md", 0), shared],
        "q2": [shared, passage("b.md", 1)],  # `shared` comes back a second time
    })
    llm = ScriptedLLM([
        make_tool_response("q1", call_id="t1"),
        make_tool_response("q2", call_id="t2"),
        make_final_response("Answer [1][2][3]."),
    ])
    result = AgentLoop(llm, ToolExecutor(retriever)).run("Q?")

    # Three distinct passages, numbered once each in order of first sight.
    assert [(s.source, s.chunk_index) for s in result.sources] == [
        ("a.md", 0), ("shared.md", 3), ("b.md", 1),
    ]
    # The second search prints only the new passage; `shared` is referred to
    # by its ORIGINAL number, 2, instead of being repeated.
    second = result.tool_calls[1].result
    assert "[3] source: b.md" in second
    assert "source: shared.md" not in second
    assert second.endswith("Also matched, already retrieved: [2].")
    # Each call reports only the passages it added.
    assert [s.source for s in result.tool_calls[0].new_sources] == ["a.md", "shared.md"]
    assert [s.source for s in result.tool_calls[1].new_sources] == ["b.md"]


def test_runs_do_not_share_citation_state() -> None:
    retriever = FakeRetriever({"q": [passage("a.md", 0)]})
    executor = ToolExecutor(retriever)  # one executor, shared like in the server
    for _ in range(2):
        llm = ScriptedLLM([make_tool_response("q"), make_final_response()])
        result = AgentLoop(llm, executor).run("Q?")
        # Every run starts numbering at [1] with only its own sources.
        assert len(result.sources) == 1
        assert "[1] source: a.md" in result.tool_calls[0].result


# ---------------------------------------------------------------------------
# ToolExecutor: registry and error handling
# ---------------------------------------------------------------------------


def test_schemas_list_every_registered_tool() -> None:
    names = [s["function"]["name"] for s in ToolExecutor(FakeRetriever({})).schemas]
    assert names == ["search_knowledge_base", "read_neighbors", "list_documents"]


def test_schema_declares_required_arguments() -> None:
    schema = ToolExecutor(FakeRetriever({})).schemas[0]["function"]
    assert schema["parameters"]["required"] == ["query"]
    assert schema["parameters"]["properties"]["query"]["type"] == "string"


def test_unknown_tool_is_reported_to_the_model() -> None:
    result = ToolExecutor(FakeRetriever({})).execute("delete_everything", {})
    assert result.startswith("Error: unknown tool 'delete_everything'")
    assert "search_knowledge_base" in result  # tells the model what does exist


def test_missing_argument_is_reported_to_the_model() -> None:
    result = ToolExecutor(FakeRetriever({})).execute("search_knowledge_base", {})
    assert result == "Error: missing required argument 'query'."


def test_wrong_argument_type_is_reported_to_the_model() -> None:
    executor = ToolExecutor(FakeRetriever({}), FakeStore([]))
    result = executor.execute("read_neighbors", {"source": "a.md", "chunk_index": "three"})
    assert result == "Error: argument 'chunk_index' must be of type integer."


def test_unexpected_argument_is_reported_to_the_model() -> None:
    result = ToolExecutor(FakeRetriever({})).execute(
        "search_knowledge_base", {"query": "x", "limit": 3}
    )
    assert "unexpected argument 'limit'" in result


def test_tool_crash_becomes_an_error_observation() -> None:
    class BrokenRetriever:
        def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
            raise RuntimeError("index offline")

    result = ToolExecutor(BrokenRetriever()).execute("search_knowledge_base", {"query": "x"})
    assert result == "Error: tool 'search_knowledge_base' failed (index offline)."


def test_repeated_search_reports_no_new_passages() -> None:
    executor = ToolExecutor(FakeRetriever({"q": [passage("a.md", 0, "body"), passage("b.md", 1, "body")]}))
    ctx = RunContext()
    first = executor.execute("search_knowledge_base", {"query": "q"}, ctx)
    again = executor.execute("search_knowledge_base", {"query": "q"}, ctx)

    assert "body" in first
    # The same passages are not printed twice; the model is told to move on.
    assert again.startswith("No new passages")
    assert "[1], [2]" in again
    assert "body" not in again
    assert len(ctx.sources) == 2


def test_empty_search_suggests_another_query() -> None:
    result = ToolExecutor(FakeRetriever({})).execute("search_knowledge_base", {"query": "nothing"})
    assert "No relevant passages" in result


def test_read_neighbors_returns_surrounding_chunks() -> None:
    chunks = [passage("doc.md", i, f"part {i}") for i in range(6)]
    executor = ToolExecutor(FakeRetriever({}), FakeStore(chunks))
    ctx = RunContext()
    result = executor.execute("read_neighbors", {"source": "doc.md", "chunk_index": 3}, ctx)

    assert "part 2" in result and "part 3" in result and "part 4" in result
    assert "part 1" not in result and "part 5" not in result
    assert len(ctx.sources) == 3  # neighbours are citable like any passage


def test_read_neighbors_window_is_clamped() -> None:
    chunks = [passage("doc.md", i, f"part {i}") for i in range(20)]
    executor = ToolExecutor(FakeRetriever({}), FakeStore(chunks))
    ctx = RunContext()
    executor.execute("read_neighbors", {"source": "doc.md", "chunk_index": 10, "window": 99}, ctx)
    assert len(ctx.sources) == 7  # window capped at 3 → chunks 7..13


def test_read_neighbors_unknown_document() -> None:
    executor = ToolExecutor(FakeRetriever({}), FakeStore([]))
    result = executor.execute("read_neighbors", {"source": "nope.md", "chunk_index": 0})
    assert result.startswith("Error: no document 'nope.md'")


def test_list_documents() -> None:
    executor = ToolExecutor(FakeRetriever({}), FakeStore([passage("b.md", 0), passage("a.md", 0)]))
    assert executor.execute("list_documents", {}) == "Available documents:\n- a.md\n- b.md"


# ---------------------------------------------------------------------------
# Formatting and message helpers
# ---------------------------------------------------------------------------


def test_format_passages_numbers_and_delimits() -> None:
    ctx = RunContext()
    formatted = format_passages([passage("a.md", 0, "Alpha"), passage("b.md", 1, "Beta")], ctx)
    assert formatted.startswith("<search_results>")
    assert formatted.endswith("</search_results>")
    assert "[1] source: a.md · chunk 0\nAlpha" in formatted
    assert "[2] source: b.md · chunk 1\nBeta" in formatted


def test_assistant_message_shape() -> None:
    tc = ToolCall(id="id1", name="search_knowledge_base", arguments={"query": "x"})
    msg = _assistant_message("thinking…", [tc])
    assert msg["role"] == "assistant"
    assert msg["tool_calls"][0]["id"] == "id1"
    assert msg["tool_calls"][0]["type"] == "function"
    assert '"query": "x"' in msg["tool_calls"][0]["function"]["arguments"]


def test_tool_result_message_shape() -> None:
    msg = _tool_result_message("id1", "result text")
    assert msg == {"role": "tool", "tool_call_id": "id1", "content": "result text"}


def test_tool_call_record_defaults() -> None:
    record = ToolCallRecord(name="n", arguments={}, result="r")
    assert record.ms == 0.0 and record.new_sources == []
