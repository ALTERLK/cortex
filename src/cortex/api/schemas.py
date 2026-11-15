"""Pydantic request/response models for the Cortex API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """One prior turn of the conversation, supplied by the client.

    The server is stateless: the browser owns the conversation and sends it
    back with every request. No sessions, no sticky routing. `role` is
    restricted to user/assistant so a client cannot inject a system prompt
    through the history.
    """

    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    # "rag"   = retrieve once, then answer (fast, cheap).
    # "agent" = the LLM decides when and how many times to search.
    # "auto"  = use the agent only when the question needs it.
    mode: Literal["auto", "rag", "agent"] = "rag"
    history: list[ChatTurn] = []


class TraceView(BaseModel):
    """Why a passage was retrieved: its rank or score at each stage."""

    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class SourceRef(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str
    trace: TraceView | None = None


class ToolCallView(BaseModel):
    """One agent tool invocation, exposed for the frontend timeline."""

    name: str
    arguments: dict[str, Any]
    result: str
    ms: float = 0.0


class RouteView(BaseModel):
    """The router's decision in auto mode."""

    mode: Literal["rag", "agent"]
    reason: str


class CitationView(BaseModel):
    """Deterministic check of the [N] markers in the answer."""

    cited: list[int]
    invalid: list[int]
    grounded: bool


class SpanView(BaseModel):
    """One timed step (LLM call or tool call) of the request."""

    kind: Literal["llm", "tool"]
    name: str
    ms: float
    input_tokens: int = 0
    output_tokens: int = 0


class AskResponse(BaseModel):
    answer: str
    mode: Literal["rag", "agent"] = "rag"   # the pipeline that answered
    route: RouteView | None = None           # set when the request used mode="auto"
    rewritten_query: str | None = None       # set when a follow-up was rewritten
    sources: list[SourceRef] = []
    tool_calls: list[ToolCallView] = []
    iterations: int | None = None
    citations: CitationView | None = None
    refusal: bool = False                    # the answer says the documents do not cover it
    spans: list[SpanView] = []
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd_est: float


class IngestRequest(BaseModel):
    directory: str = "corpus"
    chunk_size: int = Field(default=400, ge=50, le=4000)
    overlap: int = Field(default=50, ge=0, le=400)


class IngestAccepted(BaseModel):
    """202 response: the ingest job was started, poll /ingest/{job_id}."""

    job_id: str
    status: Literal["running"]
    source: str | None = None  # set for uploads: the label the document will get


class IngestStatus(BaseModel):
    status: Literal["running", "done", "failed"]
    directory: str
    files_done: int = 0
    files_total: int = 0
    chunks_stored: int | None = None
    latency_ms: float | None = None
    error: str | None = None


class DocumentInfo(BaseModel):
    source: str
    chunks: int
    uploaded: bool  # True for documents added through the API (deletable)


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class Stats(BaseModel):
    """What the UI needs to describe the running system."""

    version: str
    documents: int
    chunks: int
    retrieval_mode: str
    query_rewrite: bool
    llm_model: str
    read_only: bool
    max_upload_mb: int
