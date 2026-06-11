# Cortex Roadmap

Tracking document. One milestone = one self-contained, runnable increment.
Workflow per milestone: concept lesson → implementation → run it → interview
self-test → hands-on exercise → commit + decision log entry.

## Phase 1 — a deployed, evaluated, agentic RAG

| # | Milestone | Status | Acceptance |
|---|---|---|---|
| M0 | Skeleton + LLM abstraction (provider-agnostic client, config, smoke test) | 🚧 in progress | `uv run pytest` green; smoke test makes a live chat + tool-call round |
| M1 | Document loading + chunking (.md/.txt/.pdf, recursive chunking + overlap, metadata) | ⬜ | chunker unit-tested; CLI prints chunks for a sample doc |
| M2 | Embeddings + vector store (local BGE-M3, Qdrant, incremental ingest) | ⬜ | ingest a folder; nearest-neighbor search returns sensible hits |
| M3 | Retrieval + cited generation — **first usable RAG** | ⬜ | CLI Q&A answers with [1][2] citations grounded in sources |
| M4 | Agent layer (hand-written tool-use loop; LLM decides when/what to retrieve) | ⬜ | multi-step questions trigger ≥2 retrievals; max-iteration guard tested |
| M5 | Evaluation harness — **the differentiator** (30–50 Q eval set; hit-rate@k, MRR; LLM-as-judge) | ⬜ | `eval` command outputs a scored report; baseline numbers recorded here |
| M6 | FastAPI service + observability + deployment (SSE streaming; latency/token/cost logs; Docker) | ⬜ | public URL answers via `curl`; logs show per-request cost |
| M7 | Minimal frontend (Streamlit chat with expandable citations) | ⬜ | demo-able in browser |

## Phase 2 — from "works" to "advanced" (order matters)

| # | Direction | Status | Resume payoff |
|---|---|---|---|
| P2-A | Adaptive retrieval: query rewrite, hybrid (dense+sparse), reranker, LLM-made retrieval plan | ⬜ | measured hit-rate/answer-quality lift vs M5 baseline |
| P2-B | Long-term memory: conversation summarization + user profile store | ⬜ | cross-session continuity |
| P2-C | Multi-agent (router / retrieval / synthesis) with LangGraph | ⬜ | orchestration-framework keyword + scheduling story |
| opt | Expose Cortex as an MCP server; multimodal ingest | ⬜ | low effort, high keyword value |

Dropped: LoRA fine-tuning (lowest relevance to the AI-application-engineer target, highest time cost).

## Baseline numbers (filled at M5, updated by every Phase 2 change)

| Date | Change | Hit-rate@5 | MRR | Judge score | Notes |
|---|---|---|---|---|---|
| — | — | — | — | — | — |
