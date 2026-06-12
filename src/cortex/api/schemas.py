"""Pydantic request/response models for the Cortex API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """One prior turn of the conversation, supplied by the client.

    The server is stateless: the browser owns the conversation and sends
    it back with every request. No sessions, no sticky routing, trivially
    scalable — the standard pattern for chat APIs.
    """

    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    # "rag" = always retrieve once, then answer (fast, cheap).
    # "agent" = the LLM decides when and how many times to search (M4 loop).
    mode: Literal["rag", "agent"] = "rag"
    history: list[ChatTurn] = []


class SourceRef(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str


class ToolCallView(BaseModel):
    """One agent tool invocation, exposed for the frontend timeline."""

    name: str
    arguments: dict[str, Any]
    result: str


class AskResponse(BaseModel):
    answer: str
    mode: Literal["rag", "agent"] = "rag"
    sources: list[SourceRef] = []
    tool_calls: list[ToolCallView] = []
    iterations: int | None = None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd_est: float


class IngestRequest(BaseModel):
    directory: str = "docs"
    chunk_size: int = Field(default=400, ge=50, le=4000)
    overlap: int = Field(default=50, ge=0, le=400)


class IngestAccepted(BaseModel):
    """202 response: the ingest job was started, poll /ingest/{job_id}."""

    job_id: str
    status: Literal["running"]


class IngestStatus(BaseModel):
    status: Literal["running", "done", "failed"]
    directory: str
    chunks_stored: int | None = None
    latency_ms: float | None = None
    error: str | None = None
