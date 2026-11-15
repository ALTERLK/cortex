"""FastAPI application factory for the Cortex API.

Heavy objects — the embedding model, the Qdrant store, the BM25 index — are
built once in the lifespan handler and kept on `app.state`, so requests
reuse them instead of paying the load cost again. Tests replace the
lifespan with one that installs fakes, which is why handlers read
components from `request.app.state` rather than constructing them.
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
from cortex.agent.router import QueryRouter
from cortex.agent.tools import ToolExecutor
from cortex.api import ask, evaluation, knowledge
from cortex.api.context import request_id_var
from cortex.api.ratelimit import DailyBudget, SlidingWindowLimiter
from cortex.config import get_settings
from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.llm import get_llm_client
from cortex.rag.factory import build_retriever
from cortex.rag.generator import Generator
from cortex.rag.rewriter import QueryRewriter

logger = logging.getLogger("cortex.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()

    embedder = Embedder()
    store = VectorStore(path=settings.store_path)
    store.ensure_collection(embedder.dimension)
    retriever = build_retriever(store, embedder, settings.retrieval_mode)
    llm = get_llm_client()

    app.state.embedder = embedder
    app.state.store = store
    app.state.retriever = retriever
    app.state.generator = Generator(llm)
    app.state.agent = AgentLoop(
        llm,
        ToolExecutor(retriever, store),
        max_iterations=settings.agent_max_iterations,
        token_budget=settings.agent_token_budget,
        compact_at=settings.agent_compact_at,
    )
    app.state.router = QueryRouter(llm)
    app.state.rewriter = QueryRewriter(llm) if settings.query_rewrite else None

    logger.info(json.dumps({
        "event": "startup", "status": "ready",
        "retrieval_mode": settings.retrieval_mode, "chunks": store.count(),
    }))
    yield
    logger.info(json.dumps({"event": "shutdown"}))


def client_ip(request: Request, trust_proxy: bool) -> str:
    """The address rate limits are keyed on.

    Behind a reverse proxy every connection comes from the proxy, so all
    users would share one bucket. With trust_proxy the client address is
    taken from X-Forwarded-For instead — only safe when a trusted proxy
    sets that header, because a client can send any value it likes.
    """
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="Cortex",
        description="Agentic RAG knowledge assistant",
        version=knowledge.package_version(),
        lifespan=lifespan,
    )
    app.include_router(ask.router)
    app.include_router(knowledge.router)
    app.include_router(evaluation.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe: no dependencies, safe for load balancers."""
        return {"status": "ok"}

    # The frontend is plain HTML/CSS/JS shipped inside the package and served
    # as static files: no Node build step, and the same paths work from a
    # checkout, an installed wheel, or a container.
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/eval", include_in_schema=False)
    def eval_page() -> FileResponse:
        return FileResponse(static_dir / "eval.html")

    # Limiters live for the app's lifetime (per process; Redis if multi-instance).
    app.state.limiter = SlidingWindowLimiter(max(settings.rate_limit_per_minute, 1))
    app.state.daily_budget = DailyBudget(max(settings.daily_request_budget, 1))

    # With @app.middleware the LAST one registered runs FIRST. `security` is
    # registered first (inner) and `log_requests` second (outer), so a
    # request rejected by `security` is still logged with a request id.
    @app.middleware("http")
    async def security(request: Request, call_next: object) -> Response:
        settings = get_settings()
        path = request.url.path
        llm_backed = path.startswith("/ask")
        changes_index = path.startswith("/ingest") or (
            path.startswith("/documents") and request.method != "GET"
        )

        # API-key auth for endpoints that cost money or change the index.
        # /health stays open for load balancers; the static UI stays viewable.
        if settings.api_key and (llm_backed or changes_index):
            provided = request.headers.get("x-api-key", "")
            # compare_digest runs in constant time, so response timing does
            # not leak how many leading characters of the key were right.
            if not secrets.compare_digest(provided, settings.api_key):
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

        if llm_backed:
            # Per-client limit first, so one abusive client is rejected
            # without drawing down the budget everyone shares.
            if settings.rate_limit_per_minute > 0:
                ip = client_ip(request, settings.trust_proxy)
                if not app.state.limiter.allow(ip):
                    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
            if settings.daily_request_budget > 0 and not app.state.daily_budget.allow():
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Daily request budget for this demo is used up"},
                )

        return await call_next(request)  # type: ignore[operator]

    # Transport-level observability lives here; business metrics (tokens,
    # cost) are logged by the handler that knows them.
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
        if request.url.path in ("/", "/eval") or request.url.path.startswith("/static"):
            # "no-cache" means "revalidate before use", not "do not store":
            # the browser asks each time and gets a cheap 304 unless the file
            # changed, so a new build is never masked by a stale cached page.
            response.headers["Cache-Control"] = "no-cache"
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
