"""Tests for the evaluation pipelines and the eval loop.

All offline: retrieval, generation, the agent, and the judge are scripted.
"""

from __future__ import annotations

from typing import Any

import pytest

from cortex.agent.loop import AgentLoop
from cortex.agent.router import QueryRouter
from cortex.agent.tools import ToolExecutor
from cortex.eval.dataset import QAPair
from cortex.eval.pipelines import AgentPipeline, AutoPipeline, RagPipeline
from cortex.eval.runner import EvalReport, EvalResult, run_eval
from cortex.ingest.store import SearchResult
from cortex.llm.base import LLMResponse, TokenUsage, ToolCall
from cortex.llm.metering import MeteredLLM
from cortex.rag.generator import Generator

REFUSAL = "I don't have information about that in the provided documents."


def passage(source: str, index: int = 0) -> SearchResult:
    return SearchResult(score=0.9, text=f"text of {source}", source=source, chunk_index=index)


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.queries.append(query)
        return self._results


class FakeRewriter:
    def rewrite(self, question: str, history: list[dict] | None) -> str:
        return "standalone query"


class ScriptedLLM:
    """Replies from a script; the same object serves as system LLM or judge."""

    def __init__(self, replies: list[LLMResponse | str]) -> None:
        self._replies = list(replies)

    def chat(self, messages: list[dict[str, Any]], *, tools: Any = None,
             temperature: float = 0.3) -> LLMResponse:
        reply = self._replies.pop(0)
        if isinstance(reply, str):
            return LLMResponse(content=reply, usage=TokenUsage(100, 20))
        return reply


def search_call(query: str) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="t1", name="search_knowledge_base", arguments={"query": query})],
        finish_reason="tool_calls",
        usage=TokenUsage(200, 15),
    )


JUDGE_5 = '{"correctness": 5, "faithfulness": 5}'


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def test_rag_pipeline_retrieves_and_generates() -> None:
    retriever = FakeRetriever([passage("a.md")])
    pipeline = RagPipeline(retriever, Generator(ScriptedLLM(["<thinking>hm</thinking>Answer [1]."])))
    out = pipeline.run(QAPair("What is A?", "A.", ["a.md"]))

    assert out.answer == "Answer [1]."  # thinking stripped
    assert out.mode == "rag" and out.tool_calls == 0
    assert [p.source for p in out.passages] == ["a.md"]
    assert retriever.queries == ["What is A?"]


def test_rag_pipeline_rewrites_only_when_there_is_history() -> None:
    retriever = FakeRetriever([passage("a.md")])
    pipeline = RagPipeline(retriever, Generator(ScriptedLLM(["A [1].", "B [1]."])), FakeRewriter())

    pipeline.run(QAPair("First?", "x", ["a.md"]))
    pipeline.run(QAPair("And it?", "x", ["a.md"], history=[{"role": "user", "content": "First?"}]))
    assert retriever.queries == ["First?", "standalone query"]


def test_agent_pipeline_reports_sources_and_tool_calls() -> None:
    llm = ScriptedLLM([search_call("q"), "Agent answer [1]."])
    agent = AgentLoop(llm, ToolExecutor(FakeRetriever([passage("a.md"), passage("b.md")])))
    out = AgentPipeline(agent).run(QAPair("Q?", "x", ["b.md"]))

    assert out.mode == "agent"
    assert out.tool_calls == 1
    assert [p.source for p in out.passages] == ["a.md", "b.md"]


def build_auto(rag_reply: str, router_reply: str | None = None) -> AutoPipeline:
    retriever = FakeRetriever([passage("a.md")])
    rag = RagPipeline(retriever, Generator(ScriptedLLM([rag_reply])))
    agent = AgentPipeline(AgentLoop(
        ScriptedLLM([search_call("q"), "Agent answer [1]."]), ToolExecutor(retriever)))
    router = QueryRouter(ScriptedLLM([router_reply])) if router_reply else None
    return AutoPipeline(rag, agent, router)


def test_auto_classify_follows_the_router() -> None:
    qa = QAPair("Q?", "x", ["a.md"])
    assert build_auto("RAG answer [1].", '{"route": "rag", "reason": "r"}').run(qa).mode == "rag"
    assert build_auto("RAG answer [1].", '{"route": "agent", "reason": "r"}').run(qa).mode == "agent"


def test_auto_escalate_keeps_a_grounded_rag_answer() -> None:
    out = build_auto("RAG answer [1].").run(QAPair("Q?", "x", ["a.md"]))
    assert out.mode == "rag"
    assert out.answer == "RAG answer [1]."


def test_auto_escalate_hands_a_refusal_to_the_agent() -> None:
    out = build_auto(REFUSAL).run(QAPair("Q?", "x", ["a.md"]))
    assert out.mode == "agent"
    assert out.answer == "Agent answer [1]."
    assert out.tool_calls == 1


def test_auto_escalate_hands_an_uncited_answer_to_the_agent() -> None:
    out = build_auto("A confident claim with no citation.").run(QAPair("Q?", "x", ["a.md"]))
    assert out.mode == "agent"


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------


def test_run_eval_scores_quality_and_records_cost() -> None:
    meter = MeteredLLM(ScriptedLLM(["Answer [1].", "Other [2]."]))
    pipeline = RagPipeline(FakeRetriever([passage("a.md"), passage("b.md")]), Generator(meter))
    dataset = [
        QAPair("Q1?", "A1", ["b.md"]),            # expected source at rank 2
        QAPair("Q2?", "A2", ["missing.md"]),      # expected source never retrieved
    ]
    report = run_eval(dataset, pipeline, ScriptedLLM([JUDGE_5, JUDGE_5]),
                      meter=meter, verbose=False)

    first, second = report.results
    assert (first.hit, first.reciprocal_rank) == (True, 0.5)
    assert (second.hit, second.reciprocal_rank) == (False, 0.0)
    assert report.hit_rate == pytest.approx(0.5)
    # Cost is metered per question and excludes the judge's own calls.
    assert (first.llm_calls, first.input_tokens, first.output_tokens) == (1, 100, 20)
    assert first.grounded is True
    assert first.latency_ms >= 0


def test_run_eval_scores_refusals_without_the_judge() -> None:
    pipeline = RagPipeline(FakeRetriever([passage("a.md")]),
                           Generator(ScriptedLLM([REFUSAL, "Made-up answer [1]."])))
    dataset = [
        QAPair("Trap 1?", "", [], category="unanswerable", answerable=False),
        QAPair("Trap 2?", "", [], category="unanswerable", answerable=False),
    ]
    # An empty judge script: any judge call would raise IndexError.
    report = run_eval(dataset, pipeline, ScriptedLLM([]), verbose=False)

    assert [r.correctness for r in report.results] == [5.0, 1.0]
    assert report.refusal_accuracy == pytest.approx(0.5)


def test_run_eval_scores_agent_on_everything_it_retrieved() -> None:
    # Six passages: the expected one is 6th, outside any top-5 cut-off.
    passages = [passage(f"d{i}.md") for i in range(5)] + [passage("target.md")]
    llm = ScriptedLLM([search_call("q"), "Answer [6]."])
    agent = AgentLoop(llm, ToolExecutor(FakeRetriever(passages), top_k=6))
    report = run_eval([QAPair("Q?", "A", ["target.md"])], AgentPipeline(agent),
                      ScriptedLLM([JUDGE_5]), top_k=5, verbose=False)

    result = report.results[0]
    assert result.mode == "agent" and result.tool_calls == 1
    assert result.hit is True  # counted although it is beyond rank 5


# ---------------------------------------------------------------------------
# EvalReport: cost aggregates and persistence
# ---------------------------------------------------------------------------


def make_row(**kw: Any) -> EvalResult:
    base: dict[str, Any] = dict(question="Q?", hit=True, reciprocal_rank=1.0,
                                generated_answer="A [1].", correctness=4.0, faithfulness=4.0)
    return EvalResult(**{**base, **kw})


def test_report_cost_aggregates() -> None:
    report = EvalReport(
        [
            make_row(mode="rag", llm_calls=1, input_tokens=1000, output_tokens=100, latency_ms=2000),
            make_row(mode="agent", llm_calls=3, tool_calls=2, input_tokens=5000, output_tokens=300,
                     latency_ms=8000),
        ],
        {"price_input_per_m": 1.0, "price_output_per_m": 5.0},
    )
    summary = report.summary()
    assert summary["agent_share"] == pytest.approx(0.5)
    assert summary["avg_llm_calls"] == pytest.approx(2.0)
    assert summary["avg_tool_calls"] == pytest.approx(1.0)
    assert summary["avg_latency_ms"] == pytest.approx(5000)
    # (1000 + 5000) / 2 input tokens at $1/M + (100 + 300) / 2 output tokens at $5/M
    assert summary["avg_cost_usd"] == pytest.approx(3000 / 1e6 + 200 * 5 / 1e6)


def test_report_without_token_data_reports_no_cost() -> None:
    summary = EvalReport([make_row()]).summary()
    assert summary["avg_cost_usd"] is None
    assert summary["avg_input_tokens"] is None


def test_report_grounded_rate_ignores_unchecked_rows() -> None:
    report = EvalReport([make_row(grounded=True), make_row(grounded=False), make_row(grounded=None)])
    assert report.grounded_rate == pytest.approx(0.5)
    assert EvalReport([make_row(grounded=None)]).grounded_rate is None


def test_summary_is_json_safe_for_categories_without_retrieval() -> None:
    report = EvalReport([make_row(category="unanswerable", answerable=False)])
    assert report.summary()["by_category"]["unanswerable"]["hit_rate"] is None


def test_report_round_trips_through_json(tmp_path) -> None:
    original = EvalReport(
        [make_row(mode="agent", tool_calls=2, input_tokens=900, category="multihop", grounded=True)],
        {"label": "Agent", "group": "mode"},
    )
    path = tmp_path / "run.json"
    original.save(path)
    loaded = EvalReport.load(path)

    assert loaded.meta == {"label": "Agent", "group": "mode"}
    assert loaded.results[0].mode == "agent"
    assert loaded.results[0].tool_calls == 2
    assert loaded.results[0].category == "multihop"
    assert loaded.summary() == original.summary()


def test_report_loads_the_original_bare_list_layout(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        '[{"question": "Q?", "category": "factual", "answerable": true, "hit": true,'
        ' "reciprocal_rank": 0.5, "correctness": 4.0, "faithfulness": 5.0,'
        ' "retrieved_sources": ["a.md"], "answer_preview": "Answer."}]',
        encoding="utf-8",
    )
    report = EvalReport.load(path)
    assert report.meta == {}
    assert report.results[0].generated_answer == "Answer."
    assert report.results[0].mode == "rag"      # default for runs that predate modes
    assert report.has_cost is False


# ---------------------------------------------------------------------------
# run_eval: a dropped connection costs a retry, not the run
# ---------------------------------------------------------------------------


class FlakyPipeline:
    """Fails with a provider error a set number of times, then answers."""

    def __init__(self, failures: int, meter: MeteredLLM | None = None) -> None:
        self._failures = failures
        self._meter = meter
        self.attempts = 0

    def run(self, qa: QAPair):
        from cortex.eval.pipelines import PipelineOutput
        from cortex.llm.base import LLMUnavailableError

        self.attempts += 1
        if self._meter is not None:
            self._meter.chat([{"role": "user", "content": qa.question}])  # 100 in / 20 out
        if self.attempts <= self._failures:
            raise LLMUnavailableError("connection reset")
        return PipelineOutput("Answer [1].", [passage("a.md")])


def test_run_eval_retries_a_question_when_the_provider_drops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cortex.eval.runner._RETRY_DELAYS_S", (0, 0, 0))
    meter = MeteredLLM(ScriptedLLM(["x", "x", "x"]))
    pipeline = FlakyPipeline(failures=2, meter=meter)
    report = run_eval([QAPair("Q?", "A", ["a.md"])], pipeline, ScriptedLLM([JUDGE_5]),
                      meter=meter, verbose=False)

    assert pipeline.attempts == 3
    assert report.results[0].correctness == 5.0
    # Only the successful attempt is charged to the answer.
    assert (report.results[0].llm_calls, report.results[0].input_tokens) == (1, 100)


def test_run_eval_gives_up_after_the_last_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from cortex.llm.base import LLMUnavailableError

    monkeypatch.setattr("cortex.eval.runner._RETRY_DELAYS_S", (0, 0, 0))
    pipeline = FlakyPipeline(failures=99)
    with pytest.raises(LLMUnavailableError):
        run_eval([QAPair("Q?", "A", ["a.md"])], pipeline, ScriptedLLM([]), verbose=False)
    assert pipeline.attempts == 4  # the first try plus three retries
