"""Tests for the MCP server tool functions.

All offline — the lazy singletons are monkeypatched with fakes, so no
embedding model, no Qdrant, no LLM.
"""

from __future__ import annotations

import pytest

import cortex.mcp_server as mcp_mod
from cortex.ingest.store import SearchResult
from cortex.llm.base import TokenUsage
from cortex.rag.generator import GeneratorResponse


class _FakeRetriever:
    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        return [
            SearchResult(score=0.91, text="Passage one.", source="a.md", chunk_index=0),
            SearchResult(score=0.84, text="Passage two.", source="b.md", chunk_index=3),
        ][: (top_k or 5)]


class _FakeGenerator:
    def generate(self, question: str, passages, history=None) -> GeneratorResponse:
        return GeneratorResponse(
            answer="<thinking>checking</thinking>Cited answer [1].",
            passages=passages,
            usage=TokenUsage(10, 5),
        )


@pytest.fixture(autouse=True)
def fake_components(monkeypatch: pytest.MonkeyPatch) -> _FakeRetriever:
    retriever = _FakeRetriever()
    monkeypatch.setattr(mcp_mod, "_retriever", retriever)
    monkeypatch.setattr(mcp_mod, "_generator", _FakeGenerator())
    return retriever


def test_search_returns_numbered_passages(fake_components: _FakeRetriever) -> None:
    out = mcp_mod.search_knowledge_base("chunking")
    assert "[1] (source: a.md, score: 0.910)" in out
    assert "Passage one." in out
    assert "[2] (source: b.md" in out


def test_search_respects_top_k(fake_components: _FakeRetriever) -> None:
    out = mcp_mod.search_knowledge_base("chunking", top_k=1)
    assert fake_components.last_top_k == 1
    assert "[2]" not in out


def test_ask_returns_answer_with_sources(fake_components: _FakeRetriever) -> None:
    out = mcp_mod.ask_knowledge_base("What is chunking?")
    assert out.startswith("Cited answer [1].")          # thinking stripped
    assert "Sources:" in out
    assert "[1] a.md (chunk 0)" in out
    assert "[2] b.md (chunk 3)" in out


def test_tools_are_registered() -> None:
    # FastMCP keeps a registry of @mcp.tool() functions; both must be there.
    import anyio

    tools = anyio.run(mcp_mod.mcp.list_tools)
    names = {t.name for t in tools}
    assert {"search_knowledge_base", "ask_knowledge_base"} <= names
