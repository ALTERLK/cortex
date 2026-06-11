"""Pydantic request/response models for the Cortex API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class SourceRef(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd_est: float


class IngestRequest(BaseModel):
    directory: str = "docs"
    chunk_size: int = Field(default=400, ge=50, le=4000)
    overlap: int = Field(default=50, ge=0, le=400)


class IngestResponse(BaseModel):
    chunks_stored: int
    latency_ms: float
