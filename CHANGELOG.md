# Changelog

## Unreleased

### Fixed
- **Follow-up answers no longer "correct" a right answer.** An earlier answer sent back
  as history still carried its `[N]` citation markers, which pointed at the
  previous request's passages. They are now stripped before the history is
  forwarded.

## 1.1.0 — 2026-11-15

### Added
- **Auto mode.** A request can let the server decide between single-pass RAG
  and the agent. RAG answers first; the agent takes over only when that
  answer is a refusal or cites nothing. A classify-first strategy is kept
  for comparison (`AUTO_STRATEGY`).
- **Agent evaluation.** The harness now measures the agent and auto mode on
  the same questions as RAG, and records LLM calls, tool calls, tokens,
  latency and cost next to quality.
- **Retrieval trace.** Every passage reports its rank in dense and BM25
  search, its fused score and its rerank score.
- **Agent tooling.** A tool registry (`@tool`), a `read_neighbors` tool,
  run-wide citation numbers, a token budget, compaction of old tool
  results, and a timed trace of each run.
- **Knowledge-base API.** `GET /stats`, `GET /documents`, document upload
  and delete, ingest progress.
- **Evaluation dashboard** at `/eval`, backed by `GET /eval/results`.
- **Prompt-injection benchmark** (`scripts/run_injection_eval.py`).
- **Public-deployment settings:** `READ_ONLY`, `DAILY_REQUEST_BUDGET`,
  `TRUST_PROXY`.
- **One-command launcher** (`start.bat`, `start.sh`, `scripts/demo.py`).
- Chinese interface, light theme, phone layout.

### Changed
- The web UI is rebuilt: a document library, the conversation, and an
  evidence panel showing sources and a trace of the request. Answers render
  as sanitized markdown.
- Evaluation data moved from `data/` to `benchmarks/`; demo scripts moved
  to `examples/`.
- Token prices and agent limits are configuration.
- All evaluation runs were repeated on `claude-haiku-5-5`.

### Fixed
- Agent answers had no clickable citations: the API returned no sources for
  agent runs, and each search restarted its numbering at `[1]`.
- Documents added through `/ingest` were invisible to BM25 until a restart;
  the sparse index is now rebuilt when a job finishes.
- The `list_documents` tool crashed when the reranking retriever was active.
- `.env` could be copied into the Docker image: there was no `.dockerignore`.
- Reasoning tags spelled `<anthinking>` / `<antthinking>` leaked into
  answers.
- A model that rejects `temperature` made every request fail.
- A refusal written in the language of the question was not recognised as
  one.
- One dropped connection aborted a whole evaluation run.

## 1.0.0 — 2026-06-12

First release: ingest, cited RAG, hand-written agent loop, hybrid
retrieval with reranking and query rewriting, evaluation harness, FastAPI
service with streaming, web UI, MCP server.
