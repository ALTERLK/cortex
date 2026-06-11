"""FastAPI application factory for the Cortex API.

NOTE (learning): the lifespan context manager is the modern FastAPI way to run
startup/shutdown logic — it replaced @app.on_event("startup"). Heavy objects
(the embedding model, the Qdrant store) are loaded ONCE here so every request
reuses them without paying the startup cost again.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cortex.agent.loop import AgentLoop
from cortex.agent.tools import ToolExecutor
from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.llm import get_llm_client
from cortex.rag.generator import Generator
from cortex.rag.retriever import Retriever

logger = logging.getLogger("cortex.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    embedder = Embedder()
    store = VectorStore(path="data/qdrant")
    store.ensure_collection(embedder.dimension)

    retriever = Retriever(store, embedder)
    llm = get_llm_client()

    app.state.embedder = embedder
    app.state.store = store
    app.state.retriever = retriever
    app.state.generator = Generator(llm)
    # The M4 hand-written tool-use loop, exposed via mode="agent" on /ask.
    app.state.agent = AgentLoop(llm, ToolExecutor(retriever))

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

    # NOTE (learning): middleware runs around every request, so it's the right
    # place for cross-cutting concerns like access logging. The actual business
    # metrics (tokens, cost) are logged inside the /ask handler where they're
    # available.
    @app.middleware("http")
    async def log_requests(request: Request, call_next: object) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)  # type: ignore[operator]
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(json.dumps({
            "event": "request",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
        }))
        return response

    return app
