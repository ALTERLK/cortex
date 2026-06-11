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
from typing import Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from cortex.agent.loop import ToolCallRecord
from cortex.api.schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    SourceRef,
    ToolCallView,
)
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


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    t0 = time.perf_counter()

    if body.mode == "agent":
        # M4 loop: the LLM decides when/what/how many times to search.
        result = request.app.state.agent.run(body.question)
        sources: list[SourceRef] = []
        tool_calls = [
            ToolCallView(name=tc.name, arguments=tc.arguments, result=tc.result)
            for tc in result.tool_calls
        ]
        iterations = result.iterations
    else:
        passages = request.app.state.retriever.retrieve(body.question, top_k=body.top_k)
        result = request.app.state.generator.generate(body.question, passages)
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
        for item in state.generator.generate_stream(body.question, passages):
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
        for event in state.agent.run_events(body.question):
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
