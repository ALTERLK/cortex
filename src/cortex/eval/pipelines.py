"""Answer pipelines under evaluation.

The eval loop should not care HOW an answer was produced — single-shot RAG,
the agent loop, or the router choosing between them. Each pipeline exposes
the same `run(qa)` method and returns the same `PipelineOutput`, so the
three can be measured on the same questions with the same metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cortex.agent.loop import AgentLoop
from cortex.agent.router import QueryRouter
from cortex.eval.dataset import QAPair
from cortex.ingest.store import SearchResult
from cortex.llm.postprocess import strip_thinking
from cortex.rag.citations import check_citations, is_refusal
from cortex.rag.generator import Generator
from cortex.rag.rewriter import QueryRewriter


@dataclass
class PipelineOutput:
    """What one pipeline produced for one question."""

    answer: str
    passages: list[SearchResult] = field(default_factory=list)  # [N] refers to passages[N - 1]
    mode: str = "rag"      # the pipeline that actually answered: "rag" or "agent"
    tool_calls: int = 0


class Pipeline(Protocol):
    def run(self, qa: QAPair) -> PipelineOutput: ...


class RagPipeline:
    """Retrieve once, then generate. Optionally rewrites follow-up questions."""

    def __init__(
        self, retriever: object, generator: Generator, rewriter: QueryRewriter | None = None
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._rewriter = rewriter

    def run(self, qa: QAPair) -> PipelineOutput:
        query = qa.question
        if self._rewriter is not None and qa.history:
            query = self._rewriter.rewrite(qa.question, qa.history)
        passages = self._retriever.retrieve(query)
        response = self._generator.generate(qa.question, passages, qa.history or None)
        return PipelineOutput(strip_thinking(response.answer), passages, mode="rag")


class AgentPipeline:
    """The tool-use loop: the model decides what and how often to search."""

    def __init__(self, agent: AgentLoop) -> None:
        self._agent = agent

    def run(self, qa: QAPair) -> PipelineOutput:
        result = self._agent.run(qa.question, qa.history or None)
        return PipelineOutput(
            strip_thinking(result.answer),
            result.sources,
            mode="agent",
            tool_calls=len(result.tool_calls),
        )


class AutoPipeline:
    """Use the agent only when needed, as the API's auto mode does.

    With a router, one LLM call classifies the question up front
    ("classify"). Without one, RAG answers first and the agent takes over
    only if that answer is a refusal or cites nothing ("escalate").
    """

    def __init__(self, rag: RagPipeline, agent: AgentPipeline, router: QueryRouter | None = None) -> None:
        self._rag = rag
        self._agent = agent
        self._router = router

    def run(self, qa: QAPair) -> PipelineOutput:
        if self._router is not None:
            route = self._router.route(qa.question, qa.history or None)
            return (self._agent if route.mode == "agent" else self._rag).run(qa)

        first = self._rag.run(qa)
        failed = is_refusal(first.answer) or not check_citations(
            first.answer, len(first.passages)
        ).grounded
        return self._agent.run(qa) if failed else first
