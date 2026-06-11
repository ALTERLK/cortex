"""Unit tests for the FastAPI endpoints.

All offline — no LLM calls, no Qdrant, no embedding model.
The real lifespan is replaced with a test fixture that injects fake components
into app.state, so tests run in milliseconds.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cortex.api.app import create_app
from cortex.ingest.store import SearchResult
from cortex.llm.base import TokenUsage
from cortex.rag.generator import GeneratorResponse


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRetriever:
    def retrieve(self, query: str) -> list[SearchResult]:
        return [SearchResult(score=0.9, text="Test passage.", source="test.md", chunk_index=0)]


class _FakeGenerator:
    def generate(self, question: str, passages: list[SearchResult]) -> GeneratorResponse:
        return GeneratorResponse(
            answer="Test answer [1].",
            passages=passages,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )


class _FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeStore:
    def ensure_collection(self, dimension: int) -> None:
        pass

    def upsert(self, chunks: Any, embeddings: Any) -> None:
        pass

    def count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Fixture: replace lifespan so no real models load
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    app = create_app()

    @asynccontextmanager
    async def test_lifespan(a):
        a.state.retriever = _FakeRetriever()
        a.state.generator = _FakeGenerator()
        a.state.embedder = _FakeEmbedder()
        a.state.store = _FakeStore()
        yield

    # NOTE (learning): app.router.lifespan_context is the Starlette hook that
    # TestClient calls on __enter__/__exit__. Replacing it lets us inject fake
    # state without loading any real models.
    app.router.lifespan_context = test_lifespan

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------


def test_ask_returns_answer(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "What is Cortex?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer [1]."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "test.md"
    assert data["sources"][0]["chunk_index"] == 0


def test_ask_sources_include_passage_text(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Q?"})
    assert resp.json()["sources"][0]["text"] == "Test passage."


def test_ask_token_counts(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Q?"})
    data = resp.json()
    assert data["input_tokens"] == 100
    assert data["output_tokens"] == 50


def test_ask_cost_is_non_negative(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Q?"})
    assert resp.json()["cost_usd_est"] >= 0.0


def test_ask_latency_is_non_negative(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Q?"})
    assert resp.json()["latency_ms"] >= 0.0


def test_ask_missing_question_is_422(client: TestClient) -> None:
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_invalid_top_k_is_422(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Q?", "top_k": 0})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


def test_ingest_nonexistent_directory_is_400(client: TestClient) -> None:
    resp = client.post("/ingest", json={"directory": "does_not_exist_xyz"})
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


def test_ingest_existing_directory(client: TestClient, tmp_path) -> None:
    (tmp_path / "hello.txt").write_text("Hello world, this is a test document.")
    resp = client.post("/ingest", json={"directory": str(tmp_path)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_stored"] >= 0
    assert data["latency_ms"] >= 0.0


def test_ingest_response_has_required_fields(client: TestClient, tmp_path) -> None:
    resp = client.post("/ingest", json={"directory": str(tmp_path)})
    assert resp.status_code == 200
    assert set(resp.json().keys()) == {"chunks_stored", "latency_ms"}


# ---------------------------------------------------------------------------
# Frontend (static files)
# ---------------------------------------------------------------------------


def test_index_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Cortex" in resp.text


def test_static_css_served(client: TestClient) -> None:
    resp = client.get("/static/styles.css")
    assert resp.status_code == 200
    assert "backdrop-filter" in resp.text


def test_static_js_served(client: TestClient) -> None:
    resp = client.get("/static/app.js")
    assert resp.status_code == 200
    assert "fetch" in resp.text
