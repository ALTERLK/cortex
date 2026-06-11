# Cortex — Agentic RAG Knowledge Assistant

Ask questions over your own documents and get answers **with source citations**.
Built from scratch (no agent/RAG framework) as a production-grade portfolio project:
hand-written agentic retrieval loop, self-built evaluation harness, observability,
and cloud deployment.

## Status

🚧 In active development — currently at **M0** (project skeleton + provider-agnostic
LLM layer). See [docs/roadmap.md](docs/roadmap.md) for the full milestone plan and
[docs/decisions.md](docs/decisions.md) for the design-decision log.

## Architecture (planned)

```
documents ──► loader/chunker ──► embeddings (local BGE-M3) ──► Qdrant
                                                                  │
user ──► FastAPI ──► agent loop (hand-written tool use) ──► retrieval ──► answer + citations
                         │
                  eval harness (hit-rate@k, MRR, LLM-as-judge) + cost/latency logging
```

## Tech stack

- **Python 3.12** + [uv](https://docs.astral.sh/uv/)
- **LLM**: DeepSeek via a provider-agnostic `LLMClient` abstraction (swap to any
  OpenAI-compatible or other provider with a config change)
- **Embeddings**: local `sentence-transformers` (BGE-M3) — planned (M2)
- **Vector store**: Qdrant — planned (M2)
- **API**: FastAPI + SSE streaming — planned (M6)
- **Eval**: self-built harness (retrieval metrics + LLM-as-judge) — planned (M5)

## Setup

```sh
uv sync                                # install dependencies
copy .env.example .env                 # then fill in DEEPSEEK_API_KEY
uv run pytest                          # unit tests (offline)
uv run python scripts/smoke_test.py    # live smoke test
```
