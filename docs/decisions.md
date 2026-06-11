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

## 2026-06-11 · M2

- **BGE-M3 for embeddings, local via sentence-transformers.** Multilingual (Chinese +
  English), 1024-dim, no API cost. HF_HOME redirected to E: drive via config to avoid
  filling the limited C: system partition.
- **Qdrant local file mode, no Docker.** `QdrantClient(path=...)` stores the index
  on disk; API is identical to Docker/cloud modes so switching later is one line.
- **Idempotent upserts via deterministic UUID.** Each chunk ID is SHA-256(source +
  chunk_index) truncated to a UUID. Re-ingesting the same file updates in place —
  no duplicates accumulate across runs.
- **`query_points()` over deprecated `search()`.** qdrant-client ≥1.7 renamed the
  search method; noted in code for future SDK upgrades.

## 2026-06-11 · M3

- **Context and question in one user message.** Putting numbered passages and the
  question in the same message (rather than system+user split) improves citation
  adherence — most models follow inline instructions more reliably than system-level
  ones when the task is referencing specific numbered items.
- **Temperature=0.1 for generation.** Factual grounded answers benefit from low
  temperature; creativity is not a goal here. Retrieval already provides the
  relevant information.
- **Generator returns passages alongside the answer.** The CLI and future API
  layers can display source provenance without re-querying the store — one round
  trip does both retrieval and generation context.

## 2026-06-11 · M4

- **Hand-written loop over any framework.** The entire agent is a for-loop over
  LLM calls; no LangChain/LangGraph. This makes every step debuggable and
  explainable, which is the Phase 1 goal. LangGraph enters at P2-C where
  multi-agent orchestration genuinely earns its complexity.
- **Max-iterations guard (default 10).** Without it, a hallucinating or confused
  LLM can call tools forever. The guard is the minimum viable safety mechanism;
  a production system would also add a budget cap in tokens.
- **Tool result re-serialises arguments to JSON string.** The OpenAI protocol
  requires `function.arguments` to be a JSON *string*, not a dict. Our ToolCall
  stores a parsed dict internally; we serialise back at the protocol boundary.
- **`<thinking>` blocks stripped in the CLI.** The extended-thinking model
  (claude-haiku-4-5-20251001-thinking) emits reasoning in `<thinking>` tags;
  stripping them keeps the displayed answer clean without losing the actual reply.

## 2026-06-11 · M7

- **Hand-written HTML/CSS/JS over Streamlit (and over React).** The owner asked for
  an iOS-style liquid-glass aesthetic; Streamlit's closed component structure can't
  deliver it, and React would add a Node toolchain plus a new knowledge domain to a
  backend-focused project. Three static files served by FastAPI need no build step
  and keep every line explainable.
- **Performance budget for glass effects.** `backdrop-filter` (real-time blur) is
  restricted to three fixed panels (navbar, composer, source cards); scrolling chat
  bubbles use pre-mixed rgba colors that look like glass but cost nothing. All
  animations touch only `transform`/`opacity` (GPU compositor properties), the
  ambient background is a `position:fixed` layer composited once, and
  `prefers-reduced-motion` disables the drift animations.
- **XSS-safe rendering without a framework.** All user/LLM text enters the DOM via
  `textContent`, never `innerHTML`. Citation markers `[N]` are parsed with a regex
  split so the only constructed elements are chips containing a validated integer.
- **`SourceRef` gained a `text` field.** The expandable citation cards need the
  passage text; returning it in `/ask` avoids a second round trip per citation.

## 2026-06-11 · M6

- **FastAPI with lifespan for component initialization.** The embedding model and
  Qdrant store are loaded once at startup via `@asynccontextmanager async def lifespan`.
  Requests reuse the already-loaded objects — no 2-second BGE-M3 load per request.
- **app.state over Depends() for heavy singletons.** The retriever, generator, store,
  and embedder live on `app.state`, injected via `request.app.state` in route handlers.
  For unit tests, `app.router.lifespan_context` is replaced with a fake that injects
  stub objects — no real models load, tests run in milliseconds.
- **Structured JSON logging in middleware.** Every HTTP request produces one JSON log
  line with method, path, status, and latency_ms. The `/ask` handler adds a second
  log line with token counts and cost estimate — separating transport observability
  (middleware) from business observability (handler) keeps each layer focused.
- **Cost estimate at the handler, not the middleware.** Only the `/ask` handler knows
  the token counts returned by the LLM, so cost logging belongs there, not in
  cross-cutting middleware.
- **Dockerfile pre-bakes BGE-M3 into the image.** `RUN uv run python -c "SentenceTransformer(...)"` 
  during build means the container starts instantly rather than downloading 2.3 GB at
  runtime. Trade-off: larger image (~3 GB) vs. fast, predictable startup.

## 2026-06-11 · M5

- **LLM-as-judge uses the same model as the generator.** In production you
  would separate them to avoid self-grading bias, but for a single-provider
  Phase 1 setup the bias is acceptable and the simplicity wins.
- **Fallback to 2.5/5 on unparseable judge output.** Rather than crashing the
  entire eval run when the judge returns malformed JSON, we log the default and
  continue. Robustness over precision for a development eval harness.
- **Baseline numbers (2026-06-11):** Hit-rate@5 = 85%, MRR = 0.758,
  Correctness = 3.45/5, Faithfulness = 4.25/5. These are the numbers to beat
  in Phase 2 with hybrid search and reranking.
