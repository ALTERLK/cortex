# Cortex Roadmap

Tracking document. One milestone = one self-contained, runnable increment.
Workflow per milestone: concept lesson → implementation → run it → interview
self-test → hands-on exercise → commit + decision log entry.

## Phase 1 — a deployed, evaluated, agentic RAG

| # | Milestone | Status | Acceptance |
|---|---|---|---|
| M0 | Skeleton + LLM abstraction (provider-agnostic client, config, smoke test) | ✅ done | `uv run pytest` green; smoke test makes a live chat + tool-call round |
| M1 | Document loading + chunking (.md/.txt/.pdf, recursive chunking + overlap, metadata) | ✅ done | chunker unit-tested; CLI prints chunks for a sample doc |
| M2 | Embeddings + vector store (local BGE-M3, Qdrant, incremental ingest) | ✅ done | ingest a folder; nearest-neighbor search returns sensible hits |
| M3 | Retrieval + cited generation — **first usable RAG** | ✅ done | CLI Q&A answers with [1][2] citations grounded in sources |
| M4 | Agent layer (hand-written tool-use loop; LLM decides when/what to retrieve) | ✅ done | multi-step questions trigger ≥2 retrievals; max-iteration guard tested |
| M5 | Evaluation harness — **the differentiator** (30–50 Q eval set; hit-rate@k, MRR; LLM-as-judge) | ✅ done | `eval` command outputs a scored report; baseline numbers recorded here |
| M6 | FastAPI service + observability + deployment (SSE streaming; latency/token/cost logs; Docker) | ✅ done | SSE streaming live on `/ask/stream`; logs show per-request cost; cloud deploy pending |
| M7 | Liquid-glass web frontend (hand-written HTML/CSS/JS chat UI with clickable citations, served by FastAPI) | ✅ done | demo-able at `/`; citations expand to source passages; 79 tests green |

## Phase 2 — from "works" to "advanced" (order matters)

| # | Direction | Status | Resume payoff |
|---|---|---|---|
| P2-A | Adaptive retrieval: query rewrite, hybrid (dense+sparse), reranker | ✅ done (2026-06-12) | hit@5 77.8%→88.9%, MRR 0.567→0.640 on eval v2; LLM retrieval plan deferred |
| P2-B | Long-term memory: conversation summarization + user profile store | ⬜ | cross-session continuity |
| P2-C | Multi-agent (router / retrieval / synthesis) with LangGraph | ⬜ | orchestration-framework keyword + scheduling story |
| opt | Expose Cortex as an MCP server ✅ (done early, 2026-06-11); multimodal ingest ⬜ | partial | low effort, high keyword value |

Dropped: LoRA fine-tuning (lowest relevance to the AI-application-engineer target, highest time cost).

## Baseline numbers (filled at M5, updated by every Phase 2 change)

| Date | Change | Hit-rate@5 | MRR | Correctness | Faithfulness | Notes |
|---|---|---|---|---|---|---|
| 2026-06-11 | M5 baseline (chunk_size=400, overlap=50, BGE-M3, top_k=5) | 85.0% | 0.758 | 3.45/5 | 4.25/5 | RETIRED: 20 questions over a 3-file toy corpus — kept for history, not comparable |
| 2026-06-12 | Eval v2 dense baseline (360-doc corpus, 8,342 chunks, 50 questions) | 77.8% | 0.567 | 3.54/5 | 4.16/5 | honest baseline; refusal accuracy 100% |
| 2026-06-12 | + hybrid retrieval (hand-written BM25 + RRF) | 82.2% | 0.589 | 3.44/5 | 4.06/5 | crosslingual 70%→90% |
| 2026-06-12 | + cross-encoder reranker (bge-reranker-v2-m3) | 84.4% | 0.611 | 3.68/5 | 4.24/5 | multihop 70%→90%, factual 95% |
| 2026-06-12 | + LLM query rewriting for follow-ups | **88.9%** | **0.640** | **3.92/5** | **4.34/5** | followup 40%→80%; current production config |
