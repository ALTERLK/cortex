"""Tests for the standalone-query rewriter — offline, fake LLM."""

from __future__ import annotations

from typing import Any

from cortex.llm.base import LLMResponse, TokenUsage
from cortex.rag.rewriter import QueryRewriter

HISTORY = [
    {"role": "user", "content": "FastAPI 的后台任务怎么用？"},
    {"role": "assistant", "content": "用 BackgroundTasks 参数和 add_task()。"},
]


class _FakeLLM:
    def __init__(self, answer: str = "FastAPI BackgroundTasks 在响应前还是后执行") -> None:
        self._answer = answer
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], *, tools: Any = None,
             temperature: float = 0.3) -> LLMResponse:
        self.calls += 1
        self.seen_messages = messages
        return LLMResponse(content=self._answer, usage=TokenUsage(10, 10))


class _ExplodingLLM:
    def chat(self, messages: Any, *, tools: Any = None, temperature: float = 0.3) -> LLMResponse:
        raise RuntimeError("provider down")


def test_no_history_returns_question_without_llm_call() -> None:
    llm = _FakeLLM()
    out = QueryRewriter(llm).rewrite("它什么时候执行？", None)
    assert out == "它什么时候执行？"
    assert llm.calls == 0


def test_rewrites_with_history() -> None:
    llm = _FakeLLM()
    out = QueryRewriter(llm).rewrite("它什么时候执行？", HISTORY)
    assert out == "FastAPI BackgroundTasks 在响应前还是后执行"
    assert llm.calls == 1
    # History must be part of the rewrite prompt.
    assert any("后台任务" in m.get("content", "") for m in llm.seen_messages)


def test_strips_thinking_from_rewrite() -> None:
    llm = _FakeLLM("<thinking>resolve 它</thinking>BackgroundTasks 执行时机")
    out = QueryRewriter(llm).rewrite("它呢？", HISTORY)
    assert out == "BackgroundTasks 执行时机"


def test_llm_failure_falls_back_to_question() -> None:
    out = QueryRewriter(_ExplodingLLM()).rewrite("它呢？", HISTORY)
    assert out == "它呢？"


def test_overlong_output_falls_back() -> None:
    llm = _FakeLLM("explanation " * 60)  # model rambled instead of rewriting
    out = QueryRewriter(llm).rewrite("它呢？", HISTORY)
    assert out == "它呢？"


def test_empty_output_falls_back() -> None:
    llm = _FakeLLM("")
    assert QueryRewriter(llm).rewrite("它呢？", HISTORY) == "它呢？"
