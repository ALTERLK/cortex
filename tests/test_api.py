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
from cortex.agent.router import Route
from cortex.agent.trace import Span
from cortex.api.app import create_app
from cortex.ingest.store import RetrievalTrace, SearchResult
from cortex.llm.base import TokenUsage
from cortex.rag.generator import GeneratorResponse

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRetriever:
    def __init__(self) -> None:
        self.last_query: str | None = None
        self.refreshed = 0

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.last_query = query
        return [SearchResult(
            score=0.9, text="Test passage.", source="test.md", chunk_index=0,
            trace=RetrievalTrace(dense_rank=2, sparse_rank=1, rrf_score=0.0325),
        )]

    def refresh(self) -> None:
        self.refreshed += 1


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
    _SOURCE = SearchResult(score=0.9, text="Agent passage.", source="agent.md", chunk_index=4)
    _SPANS = [
        Span(kind="llm", name="plan", ms=12.0, input_tokens=200, output_tokens=30),
        Span(kind="tool", name="search_knowledge_base", ms=3.0),
        Span(kind="llm", name="answer", ms=20.0, input_tokens=100, output_tokens=50),
    ]
    _RESPONSE = AgentResponse(
        answer="<thinking>Searching first.</thinking>Agent answer [1].",
        tool_calls=[
            ToolCallRecord(
                name="search_knowledge_base",
                arguments={"query": "test query"},
                result="<search_results>\n[1] source: agent.md · chunk 4\nAgent passage.\n</search_results>",
                ms=3.0,
                new_sources=[_SOURCE],
            )
        ],
        iterations=2,
        usage=TokenUsage(input_tokens=300, output_tokens=80),
        sources=[_SOURCE],
        spans=_SPANS,
    )

    def __init__(self) -> None:
        self.last_history: list[dict] | None = None

    def run(self, question: str, history: list[dict] | None = None) -> AgentResponse:
        self.last_history = history
        return self._RESPONSE

    def run_events(self, question: str, history: list[dict] | None = None):
        self.last_history = history
        yield self._SPANS[0]
        yield from self._RESPONSE.tool_calls
        yield self._SPANS[2]
        yield self._RESPONSE


class _FakeRouter:
    """Routes to the agent when the question mentions "compare"."""

    def route(self, question: str, history: list[dict] | None = None) -> Route:
        if "compare" in question.lower():
            return Route("agent", "needs two documents", TokenUsage(40, 5))
        return Route("rag", "single fact", TokenUsage(40, 5))


class _FakeRewriter:
    def rewrite(self, question: str, history: list[dict] | None) -> str:
        return f"standalone: {question}" if history else question


class _FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeStore:
    """In-memory stand-in for VectorStore: remembers which chunks it holds."""

    def __init__(self) -> None:
        self.points: dict[tuple[str, int], str] = {}

    def ensure_collection(self, dimension: int) -> None:
        pass

    def upsert(self, chunks: Any, embeddings: Any) -> None:
        for c in chunks:
            self.points[(c.metadata["source"], c.metadata["chunk_index"])] = c.text

    def count(self) -> int:
        return len(self.points)

    def get_file_hash(self, source: str) -> str | None:
        return None  # nothing ingested yet — every file looks new

    def delete_source(self, source: str) -> None:
        self.points = {k: v for k, v in self.points.items() if k[0] != source}

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for source, _ in self.points:
            counts[source] = counts.get(source, 0) + 1
        return counts

    def list_sources(self) -> list[str]:
        return sorted(self.source_counts())


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
        a.state.router = _FakeRouter()
        a.state.rewriter = _FakeRewriter()
        a.state.embedder = _FakeEmbedder()
        a.state.store = _FakeStore()
        yield

    # lifespan_context is the Starlette hook TestClient calls on enter/exit;
    # replacing it installs fake components without loading any real model.
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
    # The passages the agent found come back, so [1] in the answer resolves.
    assert [s["source"] for s in data["sources"]] == ["agent.md"]
    assert data["citations"] == {"cited": [1], "invalid": [], "grounded": True}
    assert [s["kind"] for s in data["spans"]] == ["llm", "tool", "llm"]


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
    assert [e for e, _ in events] == ["step", "tool", "sources", "step", "answer", "done"]
    by_type = dict(events)

    assert by_type["tool"]["name"] == "search_knowledge_base"
    assert by_type["tool"]["arguments"] == {"query": "test query"}
    # Sources found by a tool call are sent right after it.
    assert by_type["sources"][0]["source"] == "agent.md"
    assert by_type["answer"]["text"] == "Agent answer [1]."  # thinking stripped

    done = by_type["done"]
    assert done["mode"] == "agent"
    assert done["iterations"] == 2
    assert done["stop_reason"] == "answered"
    assert done["citations"]["grounded"] is True


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


def test_ingest_outside_allowlist_is_403(client: TestClient, tmp_path) -> None:
    # tmp_path is outside the configured ingest roots.
    (tmp_path / "hello.txt").write_text("Should never be indexed.")
    resp = client.post("/ingest", json={"directory": str(tmp_path)})
    assert resp.status_code == 403
    assert "allowlist" in resp.json()["detail"].lower()


def test_ingest_nonexistent_subdir_is_400(client: TestClient) -> None:
    # Inside the allowlist root but does not exist.
    resp = client.post("/ingest", json={"directory": "examples/does_not_exist_xyz"})
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


def test_ingest_returns_202_and_job_completes(client: TestClient) -> None:
    resp = client.post("/ingest", json={"directory": "examples"})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    assert resp.json()["status"] == "running"

    # TestClient runs BackgroundTasks synchronously on response exit, so by
    # the time we poll, the job has already finished.
    status = client.get(f"/ingest/{job_id}")
    assert status.status_code == 200
    data = status.json()
    assert data["status"] == "done"
    assert data["chunks_stored"] >= 1
    assert data["latency_ms"] >= 0.0
    assert data["files_done"] == data["files_total"] >= 1
    # The sparse index is rebuilt so new documents are reachable by BM25 too.
    assert client.app.state.retriever.refreshed == 1


def test_ingest_unknown_job_is_404(client: TestClient) -> None:
    resp = client.get("/ingest/nope")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Frontend (static files)
# ---------------------------------------------------------------------------


def test_index_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Cortex" in resp.text


def test_static_css_served(client: TestClient) -> None:
    for sheet in ("tokens.css", "base.css", "chat.css"):
        resp = client.get(f"/static/css/{sheet}")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/css")


def test_static_js_served(client: TestClient) -> None:
    for module in ("app.js", "api.js", "chat.js", "markdown.js"):
        resp = client.get(f"/static/js/{module}")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"]


def test_vendored_libraries_are_served_locally(client: TestClient) -> None:
    # The UI must work offline: no script may come from a CDN.
    assert client.get("/static/vendor/marked.min.js").status_code == 200
    assert client.get("/static/vendor/purify.min.js").status_code == 200
    for page in ("/", "/eval"):
        html = client.get(page).text
        assert "https://" not in html and "http://" not in html.replace("http://www.w3.org", "")


# ---------------------------------------------------------------------------
# Resilience: 502 mapping, SSE error event, auth, rate limit, request id
# ---------------------------------------------------------------------------

from cortex.llm.base import LLMUnavailableError  # noqa: E402


def _make_app(generator=None):
    """App with fake components; optionally override the generator."""
    app = create_app()

    @asynccontextmanager
    async def test_lifespan(a):
        a.state.retriever = _FakeRetriever()
        a.state.generator = generator or _FakeGenerator()
        a.state.agent = _FakeAgent()
        a.state.router = _FakeRouter()
        a.state.rewriter = _FakeRewriter()
        a.state.embedder = _FakeEmbedder()
        a.state.store = _FakeStore()
        yield

    app.router.lifespan_context = test_lifespan
    return app


class _DownGenerator:
    """Simulates a dead/timing-out LLM provider."""

    def generate(self, question, passages, history=None):
        raise LLMUnavailableError("connection timed out")

    def generate_stream(self, question, passages, history=None):
        raise LLMUnavailableError("connection timed out")
        yield  # pragma: no cover — makes this a generator function


def test_ask_llm_down_returns_502() -> None:
    with TestClient(_make_app(generator=_DownGenerator())) as c:
        resp = c.post("/ask", json={"question": "Q?"})
        assert resp.status_code == 502
        assert resp.json()["detail"] == "LLM provider unavailable"


def test_stream_llm_down_emits_error_event() -> None:
    with TestClient(_make_app(generator=_DownGenerator())) as c:
        events = parse_sse(c.post("/ask/stream", json={"question": "Q?"}).text)
        types = [e for e, _ in events]
        assert "error" in types
        error = next(d for e, d in events if e == "error")
        assert error["detail"] == "LLM provider unavailable"


def test_response_carries_request_id(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "Q?"})
    assert len(resp.headers["X-Request-ID"]) == 12


def test_api_key_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    from cortex.config import get_settings

    monkeypatch.setenv("API_KEY", "sekret")
    get_settings.cache_clear()
    try:
        with TestClient(_make_app()) as c:
            # No key -> 401; wrong key -> 401; right key -> 200.
            assert c.post("/ask", json={"question": "Q?"}).status_code == 401
            assert c.post("/ask", json={"question": "Q?"},
                          headers={"X-API-Key": "wrong"}).status_code == 401
            assert c.post("/ask", json={"question": "Q?"},
                          headers={"X-API-Key": "sekret"}).status_code == 200
            # Health and the UI stay open (load balancers, browsers).
            assert c.get("/health").status_code == 200
            assert c.get("/").status_code == 200
    finally:
        get_settings.cache_clear()


def test_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    from cortex.config import get_settings

    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    get_settings.cache_clear()
    try:
        with TestClient(_make_app()) as c:
            assert c.post("/ask", json={"question": "Q?"}).status_code == 200
            assert c.post("/ask", json={"question": "Q?"}).status_code == 200
            resp = c.post("/ask", json={"question": "Q?"})
            assert resp.status_code == 429
            assert "limit" in resp.json()["detail"].lower()
            # /health is not rate limited.
            assert c.get("/health").status_code == 200
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Retrieval trace, query rewriting, citation check
# ---------------------------------------------------------------------------


def test_sources_carry_retrieval_trace(client: TestClient) -> None:
    source = client.post("/ask", json={"question": "Q?"}).json()["sources"][0]
    assert source["trace"]["dense_rank"] == 2
    assert source["trace"]["sparse_rank"] == 1
    assert source["trace"]["rrf_score"] == pytest.approx(0.0325)
    assert source["trace"]["rerank_score"] is None  # the reranker did not run


def test_followup_is_rewritten_before_retrieval(client: TestClient) -> None:
    turns = [{"role": "user", "content": "How do background tasks work?"}]
    data = client.post("/ask", json={"question": "Before or after?", "history": turns}).json()
    assert data["rewritten_query"] == "standalone: Before or after?"
    # Retrieval used the standalone query; generation still got the original.
    assert client.app.state.retriever.last_query == "standalone: Before or after?"


def test_first_question_is_not_rewritten(client: TestClient) -> None:
    data = client.post("/ask", json={"question": "Standalone?"}).json()
    assert data["rewritten_query"] is None
    assert client.app.state.retriever.last_query == "Standalone?"


def test_stream_emits_rewrite_event(client: TestClient) -> None:
    turns = [{"role": "user", "content": "Earlier."}]
    events = parse_sse(client.post(
        "/ask/stream", json={"question": "And it?", "history": turns}).text)
    assert events[0] == ("rewrite", {"original": "And it?", "rewritten": "standalone: And it?"})
    assert events[1][0] == "sources"
    # The rewrite and the retrieval are both reported as timed steps.
    assert [d["name"] for e, d in events if e == "step"] == ["rewrite", "retrieve", "generate"]


def test_rag_answer_citations_are_checked(client: TestClient) -> None:
    data = client.post("/ask", json={"question": "Q?"}).json()
    # "Test answer [1]." with one retrieved passage: valid and grounded.
    assert data["citations"] == {"cited": [1], "invalid": [], "grounded": True}


def test_stream_rag_ends_with_answer_then_done(client: TestClient) -> None:
    events = parse_sse(client.post("/ask/stream", json={"question": "Q?"}).text)
    assert [e for e, _ in events][-2:] == ["answer", "done"]
    assert dict(events)["answer"]["text"] == "Test answer [1]."


# ---------------------------------------------------------------------------
# Auto mode (routing)
# ---------------------------------------------------------------------------


class _RefusingGenerator(_FakeGenerator):
    """A generator whose single-pass answer is always a refusal."""

    _ANSWER = "I don't have information about that in the provided documents."

    def generate_stream(self, question, passages, history=None):
        self.last_history = history
        yield self._ANSWER
        yield self.generate(question, passages, history)


def test_auto_escalate_keeps_a_grounded_single_pass_answer(client: TestClient) -> None:
    data = client.post("/ask", json={"question": "What is CORS?", "mode": "auto"}).json()
    assert data["mode"] == "rag"
    assert data["route"] == {"mode": "rag", "reason": "answered in a single pass"}
    assert data["tool_calls"] == []
    # The common case costs exactly what plain RAG costs.
    assert (data["input_tokens"], data["output_tokens"]) == (100, 50)


def test_auto_escalate_hands_a_refusal_to_the_agent() -> None:
    with TestClient(_make_app(generator=_RefusingGenerator())) as c:
        data = c.post("/ask", json={"question": "Hard one?", "mode": "auto"}).json()
    assert data["mode"] == "agent"
    assert data["route"]["mode"] == "agent"
    assert data["answer"] == "Agent answer [1]."
    # Only the agent's sources are reported; the failed first pass is dropped…
    assert [s["source"] for s in data["sources"]] == ["agent.md"]
    # …but its tokens are still charged: 100/50 (first pass) + 300/80 (agent).
    assert (data["input_tokens"], data["output_tokens"]) == (400, 130)


def test_stream_auto_escalate_tells_the_client_to_start_over() -> None:
    with TestClient(_make_app(generator=_RefusingGenerator())) as c:
        events = parse_sse(c.post("/ask/stream", json={"question": "Hard?", "mode": "auto"}).text)
    types = [e for e, _ in events]
    # First pass, then escalate + route, then the agent's own events.
    assert types == ["sources", "step", "delta", "step", "answer", "escalate", "route",
                     "step", "tool", "sources", "step", "answer", "done"]
    assert events[-1][1]["mode"] == "agent"


def test_explicit_rag_mode_never_escalates() -> None:
    with TestClient(_make_app(generator=_RefusingGenerator())) as c:
        data = c.post("/ask", json={"question": "Hard one?", "mode": "rag"}).json()
    assert data["mode"] == "rag"
    assert data["refusal"] is True
    assert data["route"] is None


@pytest.fixture
def classify_client(monkeypatch: pytest.MonkeyPatch):
    """Client with auto mode set to classify-first routing."""
    from cortex.config import get_settings

    monkeypatch.setenv("AUTO_STRATEGY", "classify")
    get_settings.cache_clear()
    try:
        with TestClient(_make_app()) as c:
            yield c
    finally:
        get_settings.cache_clear()


def test_auto_classify_routes_simple_question_to_rag(classify_client: TestClient) -> None:
    data = classify_client.post("/ask", json={"question": "What is CORS?", "mode": "auto"}).json()
    assert data["mode"] == "rag"
    assert data["route"] == {"mode": "rag", "reason": "single fact"}


def test_auto_classify_routes_complex_question_to_agent(classify_client: TestClient) -> None:
    data = classify_client.post("/ask", json={"question": "Compare A and B", "mode": "auto"}).json()
    assert data["mode"] == "agent"
    assert data["answer"] == "Agent answer [1]."


def test_auto_classify_charges_the_routing_call(classify_client: TestClient) -> None:
    data = classify_client.post("/ask", json={"question": "What is CORS?", "mode": "auto"}).json()
    # 100/50 from the generator plus 40/5 spent on the routing call.
    assert (data["input_tokens"], data["output_tokens"]) == (140, 55)


def test_stream_auto_classify_starts_with_route_event(classify_client: TestClient) -> None:
    events = parse_sse(classify_client.post(
        "/ask/stream", json={"question": "Compare A and B", "mode": "auto"}).text)
    assert events[0] == ("route", {"mode": "agent", "reason": "needs two documents"})
    assert events[-1][1]["mode"] == "agent"


# ---------------------------------------------------------------------------
# Knowledge base: stats, documents, upload, delete
# ---------------------------------------------------------------------------


@pytest.fixture
def kb_client(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Client whose uploads land in a temporary directory."""
    from cortex.config import get_settings

    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    try:
        with TestClient(_make_app()) as c:
            yield c
    finally:
        get_settings.cache_clear()


def test_stats_describe_the_system(kb_client: TestClient) -> None:
    data = kb_client.get("/stats").json()
    assert data["documents"] == 0 and data["chunks"] == 0
    assert data["retrieval_mode"] in {"dense", "hybrid", "hybrid_rerank"}
    assert data["read_only"] is False
    assert data["max_upload_mb"] == 1


def test_upload_indexes_document_and_refreshes_sparse_index(kb_client: TestClient, tmp_path) -> None:
    body = b"# Notes\n\nCortex stores uploaded documents under a separate prefix.\n"
    resp = kb_client.put("/documents/notes.md", content=body)
    assert resp.status_code == 202
    assert resp.json()["source"] == "uploads/notes.md"
    assert (tmp_path / "uploads" / "notes.md").read_bytes() == body

    job = kb_client.get(f"/ingest/{resp.json()['job_id']}").json()
    assert job["status"] == "done" and job["chunks_stored"] >= 1

    docs = kb_client.get("/documents").json()
    assert docs["documents"] == [
        {"source": "uploads/notes.md", "chunks": job["chunks_stored"], "uploaded": True}
    ]
    assert kb_client.get("/stats").json()["documents"] == 1
    assert kb_client.app.state.retriever.refreshed == 1


def test_upload_rejects_unsupported_type(kb_client: TestClient) -> None:
    resp = kb_client.put("/documents/script.exe", content=b"MZ")
    assert resp.status_code == 415


def test_upload_rejects_oversized_file(kb_client: TestClient) -> None:
    resp = kb_client.put("/documents/big.txt", content=b"x" * (1024 * 1024 + 1))
    assert resp.status_code == 413


def test_upload_rejects_empty_file(kb_client: TestClient) -> None:
    assert kb_client.put("/documents/empty.txt", content=b"").status_code == 400


def test_upload_name_cannot_escape_upload_dir(kb_client: TestClient, tmp_path) -> None:
    # "%2F" is an encoded "/": the name reaches the handler as "../evil.md".
    resp = kb_client.put("/documents/..%2Fevil.md", content=b"text")
    if resp.status_code == 202:
        assert resp.json()["source"] == "uploads/evil.md"
        assert not (tmp_path / "evil.md").exists()
        assert (tmp_path / "uploads" / "evil.md").exists()
    else:
        # The router refused the path outright: a name containing "/" does
        # not match the single-segment upload route.
        assert resp.status_code in (400, 404, 405)
        assert not (tmp_path / "evil.md").exists()


def test_safe_filename() -> None:
    from fastapi import HTTPException

    from cortex.api.knowledge import safe_filename

    assert safe_filename("notes.md") == "notes.md"
    assert safe_filename("../../etc/passwd.txt") == "passwd.txt"
    assert safe_filename("C:\\Users\\me\\report final.pdf") == "report_final.pdf"
    assert safe_filename("笔记.md") == "笔记.md"  # non-ASCII names survive
    for bad in ("", "...", "noextension", "shell.sh"):
        with pytest.raises(HTTPException):
            safe_filename(bad)


def test_delete_uploaded_document(kb_client: TestClient, tmp_path) -> None:
    kb_client.put("/documents/notes.md", content=b"Some text to index.")
    resp = kb_client.delete("/documents/uploads/notes.md")
    assert resp.status_code == 204
    assert kb_client.get("/documents").json()["documents"] == []
    assert not (tmp_path / "uploads" / "notes.md").exists()
    assert kb_client.app.state.retriever.refreshed == 2  # once for upload, once for delete


def test_delete_refuses_corpus_documents(kb_client: TestClient) -> None:
    assert kb_client.delete("/documents/fastapi/en/index.md").status_code == 403


def test_delete_unknown_upload_is_404(kb_client: TestClient) -> None:
    assert kb_client.delete("/documents/uploads/ghost.md").status_code == 404


# ---------------------------------------------------------------------------
# Public-demo hardening
# ---------------------------------------------------------------------------


def _client_with_env(monkeypatch: pytest.MonkeyPatch, **env: str):
    from cortex.config import get_settings

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return TestClient(_make_app())


def test_read_only_blocks_every_write(monkeypatch: pytest.MonkeyPatch) -> None:
    from cortex.config import get_settings

    try:
        with _client_with_env(monkeypatch, READ_ONLY="true") as c:
            assert c.put("/documents/a.md", content=b"x").status_code == 403
            assert c.post("/ingest", json={"directory": "examples"}).status_code == 403
            assert c.delete("/documents/uploads/a.md").status_code == 403
            # Reading and asking still work.
            assert c.get("/documents").status_code == 200
            assert c.get("/stats").json()["read_only"] is True
            assert c.post("/ask", json={"question": "Q?"}).status_code == 200
    finally:
        get_settings.cache_clear()


def test_daily_budget_caps_total_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    from cortex.config import get_settings

    try:
        with _client_with_env(monkeypatch, DAILY_REQUEST_BUDGET="2") as c:
            assert c.post("/ask", json={"question": "Q?"}).status_code == 200
            assert c.post("/ask", json={"question": "Q?"}).status_code == 200
            resp = c.post("/ask", json={"question": "Q?"})
            assert resp.status_code == 429
            assert "budget" in resp.json()["detail"].lower()
            assert c.get("/stats").status_code == 200  # only LLM-backed routes count
    finally:
        get_settings.cache_clear()


def test_rate_limit_keys_on_forwarded_ip_only_when_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
    from cortex.config import get_settings

    def ask(c: TestClient, ip: str) -> int:
        return c.post("/ask", json={"question": "Q?"}, headers={"X-Forwarded-For": ip}).status_code

    try:
        # Trusted proxy: each forwarded address gets its own bucket.
        with _client_with_env(monkeypatch, RATE_LIMIT_PER_MINUTE="1", TRUST_PROXY="true") as c:
            assert ask(c, "1.1.1.1") == 200
            assert ask(c, "1.1.1.1") == 429
            assert ask(c, "2.2.2.2") == 200
        # Untrusted: the header is ignored, so a client cannot dodge the
        # limit by inventing addresses.
        with _client_with_env(monkeypatch, RATE_LIMIT_PER_MINUTE="1", TRUST_PROXY="false") as c:
            assert ask(c, "1.1.1.1") == 200
            assert ask(c, "9.9.9.9") == 429
    finally:
        get_settings.cache_clear()


def test_api_key_protects_document_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    from cortex.config import get_settings

    try:
        with _client_with_env(monkeypatch, API_KEY="sekret") as c:
            assert c.put("/documents/a.md", content=b"x").status_code == 401
            assert c.get("/documents").status_code == 200  # reads stay open
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# /eval/results
# ---------------------------------------------------------------------------


def test_eval_results_summarise_saved_runs(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cortex.config import get_settings
    from cortex.eval.runner import EvalReport, EvalResult

    def row(hit: bool, correctness: float, **kw) -> EvalResult:
        return EvalResult(question="Q?", hit=hit, reciprocal_rank=1.0 if hit else 0.0,
                          generated_answer="A [1].", correctness=correctness, faithfulness=4.0, **kw)

    EvalReport([row(True, 5.0), row(False, 3.0)],
               {"label": "Dense", "group": "retrieval", "order": 1}).save(tmp_path / "b-dense.json")
    EvalReport([row(True, 4.0, mode="agent", tool_calls=2, input_tokens=900, output_tokens=100)],
               {"label": "Agent", "group": "mode", "order": 2,
                "price_input_per_m": 1.0, "price_output_per_m": 5.0}).save(tmp_path / "a-agent.json")
    (tmp_path / "notes.json").write_text('{"not": "a run"}', encoding="utf-8")

    try:
        with _client_with_env(monkeypatch, EVAL_RESULTS_DIR=str(tmp_path)) as c:
            runs = c.get("/eval/results").json()["runs"]
    finally:
        get_settings.cache_clear()

    # Ordered by group then order; the unrelated JSON file is skipped.
    assert [r["meta"]["label"] for r in runs] == ["Agent", "Dense"]
    agent, dense = runs
    assert dense["summary"]["hit_rate"] == pytest.approx(0.5)
    assert dense["summary"]["correctness"] == pytest.approx(4.0)
    assert dense["summary"]["avg_cost_usd"] is None  # no token data recorded
    assert agent["summary"]["agent_share"] == pytest.approx(1.0)
    assert agent["summary"]["avg_cost_usd"] == pytest.approx(900 / 1e6 + 100 * 5 / 1e6)
    assert agent["results"][0]["tokens"] == 1000


def test_eval_page_is_served(client: TestClient) -> None:
    resp = client.get("/eval")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
