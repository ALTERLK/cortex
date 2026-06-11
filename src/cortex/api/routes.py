"""API route handlers.

Three endpoints:
  GET  /health   — liveness probe; no dependencies
  POST /ask      — RAG query: retrieves passages, generates a cited answer
  POST /ingest   — ingest a directory of documents into the vector store
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from cortex.api.schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    SourceRef,
)
from cortex.ingest.pipeline import ingest_directory

router = APIRouter()
logger = logging.getLogger("cortex.api")

# NOTE (learning): these are approximate Claude Haiku 4.5 prices (USD/token).
# Actual 4SAPI rates may differ; treat cost_usd_est as an order-of-magnitude
# signal, not a billing figure.
_INPUT_COST_PER_TOKEN = 0.80 / 1_000_000   # $0.80 per million input tokens
_OUTPUT_COST_PER_TOKEN = 4.00 / 1_000_000  # $4.00 per million output tokens


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    t0 = time.perf_counter()

    passages = request.app.state.retriever.retrieve(body.question)
    result = request.app.state.generator.generate(body.question, passages)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    cost = (
        result.usage.input_tokens * _INPUT_COST_PER_TOKEN
        + result.usage.output_tokens * _OUTPUT_COST_PER_TOKEN
    )

    logger.info(json.dumps({
        "event": "ask",
        "latency_ms": latency_ms,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "cost_usd_est": round(cost, 6),
        "sources_returned": len(passages),
    }))

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceRef(source=p.source, chunk_index=p.chunk_index, score=round(p.score, 4), text=p.text)
            for p in passages
        ],
        latency_ms=latency_ms,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cost_usd_est=round(cost, 6),
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(body: IngestRequest, request: Request) -> IngestResponse:
    directory = Path(body.directory)
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    t0 = time.perf_counter()
    n = ingest_directory(
        directory,
        request.app.state.store,
        request.app.state.embedder,
        chunk_size=body.chunk_size,
        overlap=body.overlap,
        verbose=False,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(json.dumps({
        "event": "ingest",
        "directory": str(directory),
        "chunks_stored": n,
        "latency_ms": latency_ms,
    }))

    return IngestResponse(chunks_stored=n, latency_ms=latency_ms)
