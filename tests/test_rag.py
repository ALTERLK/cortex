"""Tests for the RAG retriever and generator.

All offline — no network, no embedding model, no Qdrant on disk.
"""

from __future__ import annotations

from typing import Any

from cortex.ingest.store import SearchResult
from cortex.llm.base import LLMResponse, TokenUsage
from cortex.rag.generator import Generator, _build_user_message
from cortex.rag.retriever import Retriever


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Returns a fixed vector regardless of input."""

    dimension = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class FakeStore:
    """Returns pre-canned SearchResults."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_vector: list[float] | None = None
        self.last_top_k: int | None = None

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        self.last_vector = query_vector
        self.last_top_k = top_k
        return self._results[:top_k]


class FakeLLM:
    """Returns a canned answer."""

    def __init__(self, answer: str = "Canned answer [1].") -> None:
        self._answer = answer
        self.seen_messages: list[dict[str, Any]] | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        self.seen_messages = messages
        return LLMResponse(
            content=self._answer,
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )


def make_result(source: str = "doc.md", idx: int = 0, text: str = "Some text.") -> SearchResult:
    return SearchResult(score=0.9, text=text, source=source, chunk_index=idx)


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------


def test_retriever_passes_vector_to_store() -> None:
    store = FakeStore([make_result()])
    retriever = Retriever(store, FakeEmbedder(), top_k=3)
    retriever.retrieve("my query")
    assert store.last_vector == [0.1, 0.2, 0.3, 0.4]
    assert store.last_top_k == 3


def test_retriever_returns_store_results() -> None:
    expected = [make_result(source="a.md", idx=0), make_result(source="b.md", idx=1)]
    store = FakeStore(expected)
    results = Retriever(store, FakeEmbedder()).retrieve("query")
    assert results == expected


# ---------------------------------------------------------------------------
# Generator / prompt-builder tests
# ---------------------------------------------------------------------------


def test_user_message_contains_question() -> None:
    msg = _build_user_message("What is RAG?", [])
    assert "What is RAG?" in msg


def test_user_message_numbers_passages() -> None:
    passages = [make_result(text="Passage A"), make_result(text="Passage B")]
    msg = _build_user_message("Q?", passages)
    assert "[1]" in msg
    assert "[2]" in msg
    assert "Passage A" in msg
    assert "Passage B" in msg


def test_user_message_includes_source_name() -> None:
    passages = [make_result(source="notes.md", text="Content")]
    msg = _build_user_message("Q?", passages)
    assert "notes.md" in msg


def test_generator_calls_llm_with_system_and_user() -> None:
    llm = FakeLLM()
    gen = Generator(llm)
    gen.generate("What is RAG?", [make_result()])
    messages = llm.seen_messages
    assert messages is not None
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]


def test_generator_returns_llm_answer() -> None:
    gen = Generator(FakeLLM("My answer [1]."))
    result = gen.generate("Q?", [make_result()])
    assert result.answer == "My answer [1]."


def test_generator_attaches_passages_to_response() -> None:
    passages = [make_result(source="x.md")]
    result = Generator(FakeLLM()).generate("Q?", passages)
    assert result.passages == passages


def test_generator_records_token_usage() -> None:
    result = Generator(FakeLLM()).generate("Q?", [make_result()])
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20


def test_generator_handles_empty_passages() -> None:
    result = Generator(FakeLLM("No context answer.")).generate("Q?", [])
    assert result.answer == "No context answer."
