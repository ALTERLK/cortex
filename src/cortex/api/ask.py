"""Question answering endpoints.

  POST /ask         — answer as one JSON document
  POST /ask/stream  — the same answer as Server-Sent Events

Both are thin views over one generator, `_answer_events`, which runs the
request and yields (event, payload) pairs as things happen. The streaming
endpoint forwards each pair to the client; the JSON endpoint drains the
generator and folds the pairs into a single response. One implementation,
so the two endpoints cannot drift apart.

Event sequence:
  rag   pipeline:  [rewrite] sources  delta×N  answer  done
  agent pipeline:  step / tool / sources … (repeated)  answer  done
  auto mode adds a `route` event naming the pipeline that answered, and —
  when a first RAG pass fails and the agent takes over — an `escalate`
  event telling the client to discard what it has shown so far.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from cortex.agent.loop import AgentResponse, ToolCallRecord
from cortex.agent.trace import Span
from cortex.api.context import request_id_var
from cortex.api.schemas import AskRequest, AskResponse, SourceRef, TraceView
from cortex.config import get_settings
from cortex.ingest.store import SearchResult
from cortex.llm.base import LLMUnavailableError, TokenUsage
from cortex.llm.postprocess import ThinkingStreamFilter, strip_thinking
from cortex.rag.citations import check_citations, is_refusal, strip_citations
from cortex.rag.generator import GeneratorResponse

router = APIRouter()
logger = logging.getLogger("cortex.api")

# Cap on prior turns sent to the LLM: keeps token cost bounded no matter
# how long the browser-side conversation grows.
_MAX_HISTORY_TURNS = 12

Event = tuple[str, Any]


def _log(payload: dict) -> None:
    """Emit one structured log line, stamped with the request id."""
    payload["request_id"] = request_id_var.get()
    logger.info(json.dumps(payload))


def _source_ref(p: SearchResult) -> dict:
    trace = TraceView(**p.trace.__dict__) if p.trace else None
    return SourceRef(
        source=p.source, chunk_index=p.chunk_index,
        score=round(p.score, 4), text=p.text, trace=trace,
    ).model_dump(exclude_none=True)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _cost_usd(usage: TokenUsage) -> float:
    settings = get_settings()
    return round(
        usage.input_tokens * settings.llm_price_input_per_m / 1_000_000
        + usage.output_tokens * settings.llm_price_output_per_m / 1_000_000,
        6,
    )


@dataclass
class _Outcome:
    """What a pipeline produced — the input for the final `done` event."""

    answer: str = ""
    n_sources: int = 0
    n_tools: int = 0
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))
    extra: dict = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """True when the answer is a refusal or cites no retrieved passage.

        Both are detected without an LLM call, which makes this a free
        signal for "the cheap pipeline did not work for this question".
        """
        return is_refusal(self.answer) or not check_citations(self.answer, self.n_sources).grounded


def _answer_events(body: AskRequest, state: Any) -> Iterator[Event]:
    """Run one request, yielding (event, payload) pairs as progress happens."""
    t0 = time.perf_counter()
    # Earlier answers go back to the model without their [N] markers: each
    # request numbers its own passages, so old numbers would point at the
    # wrong ones.
    history = [
        {"role": t.role, "content": strip_citations(t.content) if t.role == "assistant" else t.content}
        for t in body.history[-_MAX_HISTORY_TURNS:]
    ]
    outcome: _Outcome | None = None
    spent = TokenUsage(0, 0)  # usage beyond the answering pipeline: routing, a discarded first pass

    def run(events: Iterator[Event]) -> Iterator[Event]:
        """Forward a pipeline's events, keeping its internal outcome back."""
        nonlocal outcome
        for event, payload in events:
            if event == "_outcome":
                outcome = payload
            else:
                yield event, payload

    mode = body.mode
    if mode == "auto":
        router_ = getattr(state, "router", None)
        if get_settings().auto_strategy == "classify" and router_ is not None:
            # Predict first: one LLM call decides which pipeline to run.
            started = time.perf_counter()
            route = router_.route(body.question, history)
            mode, spent = route.mode, route.usage
            yield "route", {"mode": route.mode, "reason": route.reason}
            yield "step", Span(
                kind="llm", name="route", ms=_elapsed_ms(started),
                input_tokens=route.usage.input_tokens, output_tokens=route.usage.output_tokens,
            ).as_dict()
        else:
            # Escalate on failure: try single-shot RAG, and hand over to the
            # agent only if that answer is a refusal or cites nothing. The
            # common case pays nothing extra; the hard case pays for both.
            yield from run(_rag_events(body, state, history))
            if outcome is not None and outcome.failed:
                spent, outcome, mode = outcome.usage, None, "agent"
                yield "escalate", {"reason": "the single-pass answer found nothing to cite"}
                yield "route", {"mode": "agent", "reason": "single pass came up empty; escalated"}
            else:
                mode = "rag"
                yield "route", {"mode": "rag", "reason": "answered in a single pass"}

    if outcome is None:
        pipeline = _agent_events if mode == "agent" else _rag_events
        yield from run(pipeline(body, state, history))
    assert outcome is not None

    total = TokenUsage(outcome.usage.input_tokens + spent.input_tokens,
                       outcome.usage.output_tokens + spent.output_tokens)
    citations = check_citations(outcome.answer, outcome.n_sources)
    done = {
        "mode": mode,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "input_tokens": total.input_tokens,
        "output_tokens": total.output_tokens,
        "cost_usd_est": _cost_usd(total),
        "citations": citations.as_dict(),
        "refusal": is_refusal(outcome.answer),
        **outcome.extra,
    }
    _log({
        "event": "ask", "requested_mode": body.mode, **done,
        "citations": citations.grounded,
        "sources_returned": outcome.n_sources, "tool_calls": outcome.n_tools,
    })
    yield "done", done


def _rag_events(body: AskRequest, state: Any, history: list[dict]) -> Iterator[Event]:
    """Single-shot RAG: (rewrite) → retrieve → generate, streaming the answer."""
    query = body.question
    rewrite_ms = None
    rewriter = getattr(state, "rewriter", None)
    if rewriter is not None and history:
        # A follow-up like "does it run before or after?" has nothing to
        # retrieve on its own; rewrite it into a standalone query first.
        started = time.perf_counter()
        query = rewriter.rewrite(body.question, history)
        rewrite_ms = _elapsed_ms(started)
        if query != body.question:
            yield "rewrite", {"original": body.question, "rewritten": query}

    started = time.perf_counter()
    passages = state.retriever.retrieve(query, top_k=body.top_k)
    retrieve_ms = _elapsed_ms(started)
    yield "sources", [_source_ref(p) for p in passages]
    # Timed steps, in the same shape the agent loop reports, so the client
    # draws one kind of trace for both pipelines.
    if rewrite_ms is not None:
        yield "step", Span(kind="llm", name="rewrite", ms=rewrite_ms).as_dict()
    yield "step", Span(kind="tool", name="retrieve", ms=retrieve_ms).as_dict()

    # Reasoning models emit <thinking> blocks inline. The filter holds back
    # any suffix that could be a partial tag, so reasoning never reaches the
    # client even when a tag is split across two deltas.
    filt = ThinkingStreamFilter()
    final: GeneratorResponse | None = None
    started = time.perf_counter()
    for item in state.generator.generate_stream(body.question, passages, history):
        if isinstance(item, GeneratorResponse):
            final = item
        else:
            visible = filt.feed(item)
            if visible:
                yield "delta", visible
    tail = filt.flush()
    if tail:
        yield "delta", tail

    answer = strip_thinking(final.answer) if final else ""
    usage = final.usage if final else TokenUsage(0, 0)
    yield "step", Span(
        kind="llm", name="generate", ms=_elapsed_ms(started),
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
    ).as_dict()
    yield "answer", {"text": answer}
    yield "_outcome", _Outcome(answer, n_sources=len(passages), usage=usage)


def _agent_events(body: AskRequest, state: Any, history: list[dict]) -> Iterator[Event]:
    """Agent loop: forward each step, tool call, and newly found source."""
    final: AgentResponse | None = None
    for event in state.agent.run_events(body.question, history):
        if isinstance(event, Span):
            yield "step", event.as_dict()
        elif isinstance(event, ToolCallRecord):
            yield "tool", {
                "name": event.name, "arguments": event.arguments,
                "result": event.result, "ms": event.ms,
            }
            if event.new_sources:
                # Sources arrive incrementally; the client appends them, so
                # card N always matches citation [N].
                yield "sources", [_source_ref(p) for p in event.new_sources]
        else:
            final = event

    answer = strip_thinking(final.answer) if final else ""
    yield "answer", {"text": answer}
    if final is None:
        yield "_outcome", _Outcome(answer, extra={"iterations": 0})
        return
    extra = {"iterations": final.iterations, "stop_reason": final.stop_reason}
    if final.compactions:
        extra["compactions"] = final.compactions
    yield "_outcome", _Outcome(answer, len(final.sources), len(final.tool_calls), final.usage, extra)


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer a question and return everything in one JSON response."""
    out: dict[str, Any] = {"sources": [], "tool_calls": [], "spans": []}
    try:
        for event, payload in _answer_events(body, request.app.state):
            if event == "escalate":
                # The first pass is discarded; only the agent's run is reported.
                out.update(sources=[], tool_calls=[], spans=[], rewritten_query=None)
            elif event == "sources":
                out["sources"].extend(payload)
            elif event == "tool":
                out["tool_calls"].append(payload)
                out["spans"].append({"kind": "tool", "name": payload["name"], "ms": payload["ms"]})
            elif event == "step":
                out["spans"].append(payload)
            elif event == "route":
                out["route"] = payload
            elif event == "rewrite":
                out["rewritten_query"] = payload["rewritten"]
            elif event == "answer":
                out["answer"] = payload["text"]
            elif event == "done":
                out.update({k: v for k, v in payload.items() if k in AskResponse.model_fields})
    except LLMUnavailableError as exc:
        # Provider down or timed out: a clean 502, never a stack trace.
        _log({"event": "ask_failed", "mode": body.mode, "error": str(exc)})
        raise HTTPException(status_code=502, detail="LLM provider unavailable") from exc
    return AskResponse(**out)


def _sse(event: str, data: object) -> str:
    """Format one Server-Sent Event.

    The wire format is plain text: an `event:` line, a `data:` line, and a
    blank line as terminator. JSON-encoding the payload keeps newlines in
    the data from breaking the framing.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/ask/stream")
def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Answer a question as a stream of Server-Sent Events."""
    state = request.app.state

    def stream() -> Iterator[str]:
        try:
            for event, payload in _answer_events(body, state):
                yield _sse(event, payload)
        except LLMUnavailableError as exc:
            # The 200 status is already on the wire once streaming starts,
            # so a failure can only be reported in-band, as an event.
            _log({"event": "ask_stream_failed", "mode": body.mode, "error": str(exc)})
            yield _sse("error", {"detail": "LLM provider unavailable"})
        except Exception as exc:  # noqa: BLE001 — never end a stream without telling the client
            logger.exception("ask_stream crashed")
            _log({"event": "ask_stream_failed", "mode": body.mode, "error": repr(exc)})
            yield _sse("error", {"detail": "Internal error"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        # Tell reverse proxies not to buffer, or tokens arrive in one lump.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
