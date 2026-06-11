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

from cortex.agent.loop import AgentResponse, ToolCallRecord
from cortex.api.app import create_app
from cortex.ingest.store import SearchResult
from cortex.llm.base import TokenUsage
from cortex.rag.generator import GeneratorResponse


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRetriever:
    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        return [SearchResult(score=0.9, text="Test passage.", source="test.md", chunk_index=0)]


class _FakeGenerator:
    _ANSWER = "<thinking>Let me check the passages.</thinking>Test answer [1]."

    def __init__(self) -> None:
        self.last_history: list[dict] | None = None

    def generate(
        self, question: str, passages: list[SearchResult], history: list[dict] | None = None
    ) -> GeneratorResponse:
        self.last_history = history
        return GeneratorResponse(
            # The <thinking> block must be stripped by the API layer.
            answer=self._ANSWER,
            passages=passages,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )

    def generate_stream(
        self, question: str, passages: list[SearchResult], history: list[dict] | None = None
    ):
        self.last_history = history
        # Deltas torn mid-tag to exercise the streaming thinking filter.
        yield "<thinking>Let me check"
        yield " the passages.</thi"
        yield "nking>Test answer"
        yield " [1]."
        yield self.generate(question, passages, history)


class _FakeAgent:
    _RESPONSE = AgentResponse(
        answer="<thinking>Searching first.</thinking>Agent answer [1].",
        tool_calls=[
            ToolCallRecord(
                name="search_knowledge_base",
                arguments={"query": "test query"},
                result="[1] (source: test.md, score: 0.900)\nTest passage.",
            )
        ],
        iterations=2,
        usage=TokenUsage(input_tokens=300, output_tokens=80),
    )

    def __init__(self) -> None:
        self.last_history: list[dict] | None = None

    def run(self, question: str, history: list[dict] | None = None) -> AgentResponse:
        self.last_history = history
        return self._RESPONSE

    def run_events(self, question: str, history: list[dict] | None = None):
        self.last_history = history
        yield from self._RESPONSE.tool_calls
        yield self._RESPONSE


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
        a.state.agent = _FakeAgent()
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
# /ask — agent mode
# ---------------------------------------------------------------------------


def test_ask_default_mode_is_rag(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Q?"})
    data = resp.json()
    assert data["mode"] == "rag"
    assert data["iterations"] is None
    assert data["tool_calls"] == []


def test_ask_agent_mode_returns_tool_calls(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Q?", "mode": "agent"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "agent"
    assert data["answer"] == "Agent answer [1]."
    assert data["iterations"] == 2
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["name"] == "search_knowledge_base"
    assert data["tool_calls"][0]["arguments"] == {"query": "test query"}
    assert data["sources"] == []


def test_ask_agent_mode_token_usage(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Q?", "mode": "agent"})
    data = resp.json()
    assert data["input_tokens"] == 300
    assert data["output_tokens"] == 80


def test_ask_invalid_mode_is_422(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Q?", "mode": "chaos"})
    assert resp.status_code == 422


def test_ask_strips_thinking_blocks(client: TestClient) -> None:
    # _FakeGenerator embeds a <thinking> block; the API must remove it.
    resp = client.post("/ask", json={"question": "Q?"})
    assert resp.json()["answer"] == "Test answer [1]."
    assert "<thinking>" not in resp.json()["answer"]


# ---------------------------------------------------------------------------
# /ask — multi-turn history
# ---------------------------------------------------------------------------


def test_ask_passes_history_to_generator(client: TestClient) -> None:
    turns = [
        {"role": "user", "content": "First question?"},
        {"role": "assistant", "content": "First answer."},
    ]
    client.post("/ask", json={"question": "Follow-up?", "history": turns})
    assert client.app.state.generator.last_history == turns


def test_ask_history_truncated_to_cap(client: TestClient) -> None:
    # 20 turns sent; the server forwards only the most recent 12.
    turns = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(20)
    ]
    client.post("/ask", json={"question": "Q?", "history": turns})
    forwarded = client.app.state.generator.last_history
    assert len(forwarded) == 12
    assert forwarded[-1]["content"] == "turn 19"


def test_ask_agent_receives_history(client: TestClient) -> None:
    turns = [{"role": "user", "content": "Earlier."}]
    client.post("/ask", json={"question": "Q?", "mode": "agent", "history": turns})
    assert client.app.state.agent.last_history == turns


def test_ask_invalid_history_role_is_422(client: TestClient) -> None:
    resp = client.post("/ask", json={
        "question": "Q?",
        "history": [{"role": "system", "content": "evil prompt injection"}],
    })
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /ask/stream (SSE)
# ---------------------------------------------------------------------------


def parse_sse(text: str) -> list[tuple[str, Any]]:
    """Parse an SSE body into (event, decoded-json-data) pairs."""
    events: list[tuple[str, Any]] = []
    for frame in text.strip().split("\n\n"):
        event, data = "message", ""
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data += line[len("data: "):]
        events.append((event, json.loads(data)))
    return events


def test_stream_rag_event_sequence(client: TestClient) -> None:
    resp = client.post("/ask/stream", json={"question": "Q?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(resp.text)
    types = [e for e, _ in events]
    assert types[0] == "sources"
    assert types[-1] == "done"
    assert "delta" in types


def test_stream_rag_deltas_filter_thinking(client: TestClient) -> None:
    # The fake generator tears the <thinking> tags across chunks; the
    # reassembled visible text must contain none of it.
    events = parse_sse(client.post("/ask/stream", json={"question": "Q?"}).text)
    answer = "".join(d for e, d in events if e == "delta")
    assert answer == "Test answer [1]."


def test_stream_rag_sources_payload(client: TestClient) -> None:
    events = parse_sse(client.post("/ask/stream", json={"question": "Q?"}).text)
    sources = next(d for e, d in events if e == "sources")
    assert sources[0]["source"] == "test.md"
    assert sources[0]["text"] == "Test passage."


def test_stream_agent_event_sequence(client: TestClient) -> None:
    resp = client.post("/ask/stream", json={"question": "Q?", "mode": "agent"})
    events = parse_sse(resp.text)
    assert [e for e, _ in events] == ["tool", "answer", "done"]

    tool = events[0][1]
    assert tool["name"] == "search_knowledge_base"
    assert tool["arguments"] == {"query": "test query"}

    answer = events[1][1]
    assert answer["text"] == "Agent answer [1]."  # thinking stripped

    done = events[2][1]
    assert done["mode"] == "agent"
    assert done["iterations"] == 2


def test_stream_done_carries_stats(client: TestClient) -> None:
    events = parse_sse(client.post("/ask/stream", json={"question": "Q?"}).text)
    done = events[-1][1]
    assert done["input_tokens"] == 100
    assert done["output_tokens"] == 50
    assert done["latency_ms"] >= 0
    assert done["cost_usd_est"] >= 0


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
