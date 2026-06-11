# Cortex — Agentic RAG Knowledge Assistant

Ask questions over your own documents and get answers **with source citations**.
Built from scratch — no LangChain, no LlamaIndex: the retrieval pipeline, the
agentic tool-use loop, the evaluation harness, and the web UI are all hand-written
and explainable line by line.

## Highlights

- **Hand-written agent loop** — the LLM decides whether, what, and how many times
  to retrieve, via a ~60-line tool-use loop (`src/cortex/agent/loop.py`). No framework.
- **Provider-agnostic LLM layer** — a `Protocol`-based abstraction
  (`src/cortex/llm/`); switching DeepSeek ↔ Claude ↔ GPT is a config change,
  proven in practice during development.
- **Self-built evaluation harness** — retrieval metrics (hit-rate@k, MRR) +
  LLM-as-judge answer grading (correctness, faithfulness), runnable in one command.
- **Observability** — structured JSON logs with per-request latency, token counts,
  and cost estimate.
- **Liquid-glass web UI** — hand-written HTML/CSS/JS chat interface with clickable
  citations, SSE token streaming, and live agent tool timelines. No Node toolchain.
- **MCP server** — exposes the knowledge base as Model Context Protocol tools, so
  Claude Code / Claude Desktop can search your documents directly.

## Baseline numbers (eval set: 20 questions, 2026-06-11)

| Metric | Value |
|---|---|
| Retrieval hit-rate@5 | **85.0%** |
| MRR | **0.758** |
| Answer correctness (LLM-as-judge, 1–5) | 3.45 |
| Answer faithfulness (LLM-as-judge, 1–5) | 4.25 |

Reproduce with `uv run python scripts/run_eval.py`. Every Phase 2 retrieval
improvement (hybrid search, reranking) must beat these numbers — see
[docs/roadmap.md](docs/roadmap.md).

## Architecture

```
                    ┌─ ingest ───────────────────────────────────┐
 .md/.txt/.pdf ──►  loader ─► recursive chunker ─► BGE-M3 (local) ─► Qdrant
                    └────────────────────────────────────────────┘

                    ┌─ serve ────────────────────────────────────┐
 browser ──► FastAPI ─► agent loop (hand-written tool use)       │
   │            │            └─► search_knowledge_base ─► retriever ─► Qdrant
   │            │                         │
   │            └─► generator (numbered context, cited answer)
   │            └─► JSON logs: latency / tokens / cost
                    └────────────────────────────────────────────┘

                    ┌─ eval ─────────────────────────────────────┐
 eval_set.json ──►  hit-rate@k + MRR + LLM-as-judge ─► report    │
                    └────────────────────────────────────────────┘
```

## Quickstart

```sh
# 1. Install (Python 3.12+, uv)
uv sync

# 2. Configure: any OpenAI-compatible provider works
copy .env.example .env        # set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# 3. Index your documents (first run downloads BGE-M3, ~2.3 GB)
uv run python scripts/ingest.py

# 4. Serve
uv run python scripts/serve.py
# open http://localhost:8000  — chat UI with citations
# or:  curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
#           -d "{\"question\": \"What chunking strategy does Cortex use?\"}"
```

Run the test suite (offline, no API key needed): `uv run pytest`

## MCP server

Cortex doubles as an [MCP](https://modelcontextprotocol.io) server exposing
`search_knowledge_base(query, top_k)` and `ask_knowledge_base(question)`.
The repo ships a project-scope [`.mcp.json`](.mcp.json) — open this folder in
Claude Code and approve the server, or register it globally:

```sh
claude mcp add cortex -- uv run --directory <path-to-repo> python scripts/mcp_serve.py
```

Verify the stdio round trip without any client:

```sh
uv run python scripts/mcp_smoke.py
```

## Tech stack

**Python 3.12** · [uv](https://docs.astral.sh/uv/) · **FastAPI** ·
**Qdrant** (local file mode) · **BGE-M3** via sentence-transformers (local, multilingual) ·
any OpenAI-compatible LLM (DeepSeek, Claude via proxy, GPT…) · Docker

## Project layout

```
src/cortex/
├── llm/        provider-agnostic LLMClient protocol + OpenAI-compatible adapter
├── ingest/     loader, recursive chunker, embedder, Qdrant store, pipeline
├── rag/        retriever + cited-answer generator
├── agent/      tool schemas + hand-written tool-use loop
├── eval/       dataset, metrics (hit-rate/MRR/LLM-as-judge), runner
└── api/        FastAPI app, routes, schemas, static web UI
```

## Design decisions

Every milestone logs its key trade-offs in [docs/decisions.md](docs/decisions.md) —
chunking strategy, why no framework, eval design, glass-UI performance budget, and more.
The milestone plan lives in [docs/roadmap.md](docs/roadmap.md).
