# Design Decisions Log

One entry per significant decision. This file doubles as interview prep:
each entry should be explainable in ~30 seconds with a trade-off.

## 2026-06-11 · M0

- **DeepSeek as first LLM provider, behind an abstraction.** OpenAI-compatible API,
  ~100x cheaper than frontier models (matters when eval runs make hundreds of calls),
  payable/reachable from China. The `LLMClient` protocol + `get_llm_client()` factory
  keep every call site provider-agnostic, so switching to Claude/GPT later is a
  config change, not a rewrite.
- **No LangChain/LangGraph in Phase 1.** The agent loop, retrieval, and prompting are
  hand-written so every design is explainable in depth. LangGraph enters in Phase 2
  (multi-agent) where orchestration genuinely needs a framework.
- **`Protocol` over ABC for the LLM interface.** Structural typing means tests can use
  a plain fake class with no inheritance; production adapters stay decoupled.
- **Adapter normalizes responses, not requests.** Requests use the de-facto-standard
  OpenAI message format; only the response object is mapped into our `LLMResponse`.
  Normalizing both would be ceremony without benefit at this stage.
- **uv + Python 3.12 + src layout + hatchling.** Current industry-standard packaging;
  src layout prevents accidental imports of uninstalled code.
- **Secrets via pydantic-settings + .env (git-ignored).** Single `get_settings()`
  entry point; `.env.example` documents required variables without values.
- **Renamed adapter to `OpenAICompatibleClient` / `LLM_API_KEY`.** Switched live
  provider from DeepSeek direct to Claude via 4SAPI (OpenAI-compatible proxy).
  Proved the abstraction works: zero call-site changes outside `config.py` and the
  adapter file itself.

## 2026-06-11 · M1

- **Recursive character splitting over fixed-size splitting.** Tries semantic
  separators in priority order (\n\n → \n → space → char) so paragraph boundaries
  are respected when possible, falling back to finer splits only when necessary.
  Trade-off: slightly more complex than a naïve fixed-size split, but produces more
  semantically coherent chunks — important because chunk quality sets the RAG ceiling.
- **Piece-level overlap, not character-level.** After saving a chunk, the tail
  pieces (up to `overlap` chars total) are carried into the next chunk's starting
  set. True character-level overlap would require splitting inside a piece; piece-level
  overlap is simpler and sufficient for preventing boundary cuts on most sentences.
- **`Document` → `Chunk` pipeline keeps metadata through every stage.** Each Chunk
  carries `source` (filename) and `chunk_index` so the final answer can always be
  traced back to its origin file — the citation requirement drives this from day one.
- **pypdf for PDF loading, lazy import.** Only imported when a .pdf path is detected,
  so the non-PDF code path has no startup cost from the PDF library.
