# CLAUDE.md — Cortex

Agentic RAG knowledge assistant: ask questions over your own documents, get answers with citations.

## Commands

```sh
uv sync                                # install/update dependencies
uv run pytest                          # run unit tests (no network, no API key needed)
uv run python scripts/ingest.py        # ingest docs/ into the local Qdrant index
uv run python scripts/serve.py         # start the FastAPI server + web UI (localhost:8000)
uv run python scripts/run_eval.py      # run the evaluation harness (needs LLM_API_KEY)
uv run python scripts/smoke_test.py    # live LLM smoke test (needs LLM_API_KEY in .env)
```

## Setup

Copy `.env.example` to `.env` and set `LLM_API_KEY` (plus `LLM_BASE_URL`/`LLM_MODEL`
for your provider). Never commit `.env`. On Windows set `HF_HOME` to a roomy drive
before first ingest — BGE-M3 is a 2.3 GB download.

## Architecture rules

- **Provider-agnostic LLM access**: all LLM calls go through `cortex.llm.get_llm_client()`
  (the `LLMClient` protocol in `src/cortex/llm/base.py`). Never import a vendor SDK
  (`openai`, etc.) outside `src/cortex/llm/` adapter modules.
- **No agent/RAG frameworks in Phase 1** (no LangChain/LangGraph): retrieval, prompting,
  and the agent loop are hand-written on purpose — this project is a learning/portfolio
  vehicle and the owner must be able to explain every line.
- **Config**: read via `cortex.config.get_settings()`, never `os.environ` directly.
- Unit tests must not hit the network; fake the `LLMClient` protocol instead
  (see `tests/test_llm_base.py::FakeLLMClient`).

## Conventions

- Code, comments, docs, commit messages: English. Conversation with the owner: Chinese.
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), one commit per milestone step.
- `# NOTE (learning):` comments are intentional teaching annotations for the project owner;
  keep them during Phase 1.
- Every milestone appends its key design decision to `docs/decisions.md`.

## Roadmap

See `docs/roadmap.md`. **Phase 1 (M0–M7) is complete**: ingest → RAG with citations →
hand-written agent loop → eval harness (baseline: hit-rate@5 85%, MRR 0.758) →
FastAPI service → liquid-glass web UI. Next: Phase 2 (adaptive retrieval first).
