"""FastAPI application factory for the Cortex API.

NOTE (learning): the lifespan context manager is the modern FastAPI way to run
startup/shutdown logic — it replaced @app.on_event("startup"). Heavy objects
(the embedding model, the Qdrant store) are loaded ONCE here so every request
reuses them without paying the startup cost again.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cortex.agent.loop import AgentLoop
from cortex.agent.tools import ToolExecutor
from cortex.api.context import request_id_var
from cortex.api.ratelimit import SlidingWindowLimiter
from cortex.config import get_settings
from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.llm import get_llm_client
from cortex.rag.generator import Generator
from cortex.rag.hybrid import HybridRetriever
from cortex.rag.retriever import Retriever

logger = logging.getLogger("cortex.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    embedder = Embedder()
    store = VectorStore(path="data/qdrant")
    store.ensure_collection(embedder.dimension)

    # Retrieval strategy is a config switch so eval can A/B strategies fairly.
    mode = get_settings().retrieval_mode
    if mode == "dense":
        retriever = Retriever(store, embedder)
    else:
        retriever = HybridRetriever(store, embedder)  # builds BM25 at startup
        if mode == "hybrid_rerank":
            from cortex.rag.reranker import Reranker, RerankingRetriever

            retriever = RerankingRetriever(retriever, Reranker())
    llm = get_llm_client()

    app.state.embedder = embedder
    app.state.store = store
    app.state.retriever = retriever
    app.state.generator = Generator(llm)
    # The M4 hand-written tool-use loop, exposed via mode="agent" on /ask.
    app.state.agent = AgentLoop(llm, ToolExecutor(retriever))
    # Standalone-query rewriting for multi-turn rag requests.
    from cortex.rag.rewriter import QueryRewriter

    app.state.rewriter = QueryRewriter(llm) if get_settings().query_rewrite else None

    logger.info(json.dumps({"event": "startup", "status": "ready"}))
    yield
    logger.info(json.dumps({"event": "shutdown"}))


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Cortex",
        description="Agentic RAG knowledge assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    from cortex.api.routes import router
    app.include_router(router)

    # NOTE (learning): the frontend is plain HTML/CSS/JS served straight from
    # this package — no Node.js build step. `__file__`-relative paths work the
    # same whether the app runs from a checkout, a wheel, or a container.
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    # One limiter for the app's lifetime (per-process; Redis when multi-instance).
    app.state.limiter = SlidingWindowLimiter(max(get_settings().rate_limit_per_minute, 1))

    # NOTE (learning): with @app.middleware, the LAST registered middleware
    # runs FIRST. Registration order below: security (inner) then
    # logging/request-id (outer) — so even rejected requests get logged
    # with a request id.
    @app.middleware("http")
    async def security(request: Request, call_next: object) -> Response:
        settings = get_settings()
        path = request.url.path

        # API-key auth for the expensive/dangerous endpoints. /health stays
        # open for load balancers; the static UI stays viewable.
        if settings.api_key and path.startswith(("/ask", "/ingest")):
            provided = request.headers.get("x-api-key", "")
            # compare_digest: constant-time comparison, immune to timing attacks.
            if not secrets.compare_digest(provided, settings.api_key):
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

        # Per-IP rate limit on LLM-backed endpoints — every /ask costs money.
        if settings.rate_limit_per_minute > 0 and path.startswith("/ask"):
            client_ip = request.client.host if request.client else "unknown"
            if not app.state.limiter.allow(client_ip):
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        return await call_next(request)  # type: ignore[operator]

    # NOTE (learning): middleware runs around every request, so it's the right
    # place for cross-cutting concerns like access logging. The actual business
    # metrics (tokens, cost) are logged inside the /ask handler where they're
    # available.
    @app.middleware("http")
    async def log_requests(request: Request, call_next: object) -> Response:
        rid = uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)  # type: ignore[operator]
        finally:
            request_id_var.reset(token)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Request-ID"] = rid
        logger.info(json.dumps({
            "event": "request",
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
        }))
        return response

    return app
