# CLAUDE.md — Cortex

Agentic RAG knowledge assistant: ask questions over your own documents, get answers with citations.

## Commands

```sh
uv sync                                # install/update dependencies
uv run pytest                          # run unit tests (no network, no API key needed)
uv run python scripts/smoke_test.py    # live LLM smoke test (needs DEEPSEEK_API_KEY in .env)
```

## Setup

Copy `.env.example` to `.env` and set `DEEPSEEK_API_KEY`. Never commit `.env`.

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

See `docs/roadmap.md`. Current milestone: **M0** (project skeleton + LLM abstraction).
