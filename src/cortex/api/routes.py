"""API route handlers.

Endpoints:
  GET  /health            — liveness probe; no dependencies
  POST /ask               — RAG/agent query with a cited answer
  POST /ask/stream        — same, as Server-Sent Events
  POST /ingest            — start a background ingest job (202 + job_id)
  GET  /ingest/{job_id}   — poll ingest job status
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse

from cortex.agent.loop import ToolCallRecord
from cortex.api.schemas import (
    AskRequest,
    AskResponse,
    IngestAccepted,
    IngestRequest,
    IngestStatus,
    SourceRef,
    ToolCallView,
)
from cortex.config import get_settings
from cortex.ingest.pipeline import ingest_directory
from cortex.llm.postprocess import ThinkingStreamFilter, strip_thinking
from cortex.rag.generator import GeneratorResponse

router = APIRouter()
logger = logging.getLogger("cortex.api")

# NOTE (learning): these are approximate Claude Haiku 4.5 prices (USD/token).
# Actual 4SAPI rates may differ; treat cost_usd_est as an order-of-magnitude
# signal, not a billing figure.
_INPUT_COST_PER_TOKEN = 0.80 / 1_000_000   # $0.80 per million input tokens
_OUTPUT_COST_PER_TOKEN = 4.00 / 1_000_000  # $4.00 per million output tokens

# Cap on prior turns sent to the LLM: keeps token cost bounded no matter
# how long the browser-side conversation grows.
_MAX_HISTORY_TURNS = 12


def _history_messages(body: AskRequest) -> list[dict]:
    return [t.model_dump() for t in body.history[-_MAX_HISTORY_TURNS:]]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    t0 = time.perf_counter()

    history = _history_messages(body)

    if body.mode == "agent":
        # M4 loop: the LLM decides when/what/how many times to search.
        result = request.app.state.agent.run(body.question, history)
        sources: list[SourceRef] = []
        tool_calls = [
            ToolCallView(name=tc.name, arguments=tc.arguments, result=tc.result)
            for tc in result.tool_calls
        ]
        iterations = result.iterations
    else:
        passages = request.app.state.retriever.retrieve(body.question, top_k=body.top_k)
        result = request.app.state.generator.generate(body.question, passages, history)
        sources = [
            SourceRef(source=p.source, chunk_index=p.chunk_index, score=round(p.score, 4), text=p.text)
            for p in passages
        ]
        tool_calls = []
        iterations = None

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    cost = (
        result.usage.input_tokens * _INPUT_COST_PER_TOKEN
        + result.usage.output_tokens * _OUTPUT_COST_PER_TOKEN
    )

    logger.info(json.dumps({
        "event": "ask",
        "mode": body.mode,
        "latency_ms": latency_ms,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "cost_usd_est": round(cost, 6),
        "sources_returned": len(sources),
        "tool_calls": len(tool_calls),
    }))

    return AskResponse(
        # Extended-thinking models leak <thinking> blocks into content;
        # users never see them.
        answer=strip_thinking(result.answer),
        mode=body.mode,
        sources=sources,
        tool_calls=tool_calls,
        iterations=iterations,
        latency_ms=latency_ms,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cost_usd_est=round(cost, 6),
    )


def _sse(event: str, data: object) -> str:
    """Format one Server-Sent Event.

    NOTE (learning): the SSE wire format is just text — an `event:` line
    naming the event type, a `data:` line with the payload, and a blank
    line as terminator. JSON-encoding the payload keeps newlines inside
    the data from breaking the framing.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/ask/stream")
def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Streaming variant of /ask.

    Event sequence:
      rag   mode: sources -> delta xN -> done
      agent mode: tool xN -> answer -> done
    """
    t0 = time.perf_counter()
    state = request.app.state
    history = _history_messages(body)

    def rag_events() -> Iterator[str]:
        passages = state.retriever.retrieve(body.question, top_k=body.top_k)
        yield _sse("sources", [
            SourceRef(source=p.source, chunk_index=p.chunk_index,
                      score=round(p.score, 4), text=p.text).model_dump()
            for p in passages
        ])

        # Deltas pass through the thinking filter so reasoning never renders.
        filt = ThinkingStreamFilter()
        final: GeneratorResponse | None = None
        for item in state.generator.generate_stream(body.question, passages, history):
            if isinstance(item, GeneratorResponse):
                final = item
            else:
                visible = filt.feed(item)
                if visible:
                    yield _sse("delta", visible)
        tail = filt.flush()
        if tail:
            yield _sse("delta", tail)

        yield _done_event(final, mode="rag", iterations=None, t0=t0,
                          sources=len(passages), tool_calls=0)

    def agent_events() -> Iterator[str]:
        final = None
        n_tools = 0
        for event in state.agent.run_events(body.question, history):
            if isinstance(event, ToolCallRecord):
                n_tools += 1
                yield _sse("tool", {
                    "name": event.name,
                    "arguments": event.arguments,
                    "result": event.result,
                })
            else:
                final = event

        yield _sse("answer", {"text": strip_thinking(final.answer) if final else ""})
        yield _done_event(final, mode="agent",
                          iterations=final.iterations if final else 0,
                          t0=t0, sources=0, tool_calls=n_tools)

    def _done_event(result: object, *, mode: str, iterations: int | None,
                    t0: float, sources: int, tool_calls: int) -> str:
        usage = result.usage if result else None
        in_tok = usage.input_tokens if usage else 0
        out_tok = usage.output_tokens if usage else 0
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        cost = round(in_tok * _INPUT_COST_PER_TOKEN + out_tok * _OUTPUT_COST_PER_TOKEN, 6)

        logger.info(json.dumps({
            "event": "ask_stream", "mode": mode, "latency_ms": latency_ms,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "cost_usd_est": cost, "sources_returned": sources,
            "tool_calls": tool_calls,
        }))
        payload = {
            "mode": mode, "latency_ms": latency_ms,
            "input_tokens": in_tok, "output_tokens": out_tok,
            "cost_usd_est": cost,
        }
        if iterations is not None:
            payload["iterations"] = iterations
        return _sse("done", payload)

    events = agent_events() if body.mode == "agent" else rag_events()
    return StreamingResponse(events, media_type="text/event-stream")


# In-memory job table. Survives only the process lifetime — acceptable for a
# single-instance service; a multi-instance deployment would move this to Redis.
_INGEST_JOBS: dict[str, dict] = {}


def _allowed_ingest_roots() -> list[Path]:
    dirs = get_settings().ingest_dirs.split(",")
    return [Path(d.strip()).resolve() for d in dirs if d.strip()]


def _run_ingest_job(job_id: str, directory: Path, store: object, embedder: object,
                    chunk_size: int, overlap: int) -> None:
    t0 = time.perf_counter()
    try:
        n = ingest_directory(
            directory, store, embedder,
            chunk_size=chunk_size, overlap=overlap, verbose=False,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        _INGEST_JOBS[job_id].update(status="done", chunks_stored=n, latency_ms=latency_ms)
        logger.info(json.dumps({
            "event": "ingest", "job_id": job_id, "directory": str(directory),
            "chunks_stored": n, "latency_ms": latency_ms,
        }))
    except Exception as exc:  # noqa: BLE001 — job must record any failure
        _INGEST_JOBS[job_id].update(status="failed", error=str(exc))
        logger.error(json.dumps({"event": "ingest_failed", "job_id": job_id, "error": str(exc)}))


@router.post("/ingest", response_model=IngestAccepted, status_code=202)
def ingest(body: IngestRequest, request: Request, background: BackgroundTasks) -> IngestAccepted:
    """Start an ingest job in the background; poll GET /ingest/{job_id}.

    NOTE (learning): ingest can take minutes (embedding is the bottleneck).
    Holding an HTTP request open that long invites client timeouts — the
    202 Accepted + status-poll pattern is the standard answer.
    """
    directory = Path(body.directory).resolve()
    roots = _allowed_ingest_roots()
    # Security: only allowlisted roots may be indexed. Without this, anyone
    # who can reach the API could index arbitrary server files and then read
    # them back through /ask.
    if not any(directory == root or directory.is_relative_to(root) for root in roots):
        raise HTTPException(
            status_code=403,
            detail=f"Directory not in the ingest allowlist ({get_settings().ingest_dirs})",
        )
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    job_id = uuid.uuid4().hex
    _INGEST_JOBS[job_id] = {"status": "running", "directory": str(directory)}
    background.add_task(
        _run_ingest_job, job_id, directory,
        request.app.state.store, request.app.state.embedder,
        body.chunk_size, body.overlap,
    )
    return IngestAccepted(job_id=job_id, status="running")


@router.get("/ingest/{job_id}", response_model=IngestStatus)
def ingest_status(job_id: str) -> IngestStatus:
    job = _INGEST_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown ingest job: {job_id}")
    return IngestStatus(**job)
