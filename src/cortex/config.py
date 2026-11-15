"""Application configuration.

All runtime configuration lives here, loaded from environment variables
(or a local `.env` file, which is git-ignored). Code anywhere in the
project asks for `get_settings()` instead of reading `os.environ`
directly, so there is exactly one place that knows how config works.

Secrets such as API keys are never hardcoded or committed: `.env` stays on
the machine, and `.env.example` documents which variables exist.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, populated from environment variables / .env."""

    # --- LLM provider ---
    # Any OpenAI-compatible endpoint (DeepSeek, OpenRouter, a Claude or GPT
    # gateway…). Switching provider is a change to these three values.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    # Resilience: per-request timeout and bounded SDK retries.
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    # Prices in USD per million tokens, used only for the cost estimate shown
    # in logs and the UI. Set them to your provider's rates.
    llm_price_input_per_m: float = 0.10
    llm_price_output_per_m: float = 0.50

    # --- local model cache ---
    # Override to redirect HuggingFace downloads away from the default C: cache.
    hf_home: str = ""

    # --- retrieval ---
    # "dense"         = embedding similarity only
    # "hybrid"        = dense + BM25 fused with reciprocal rank fusion
    # "hybrid_rerank" = hybrid candidates re-scored by a cross-encoder
    retrieval_mode: str = "hybrid"
    # Rewrite history-dependent questions into standalone queries before
    # retrieval (rag mode only; costs one extra LLM call per follow-up).
    query_rewrite: bool = True
    store_path: str = "data/qdrant"

    # --- agent ---
    # How mode="auto" decides between single-shot RAG and the agent:
    # "escalate" = answer with RAG first; hand over to the agent only when
    #              that answer is a refusal or cites nothing.
    # "classify" = one LLM call predicts, before answering, which pipeline
    #              the question needs.
    auto_strategy: str = "escalate"
    agent_max_iterations: int = 6
    # Hard cap on tokens (input + output) one agent run may spend.
    agent_token_budget: int = 120_000
    # Prompt size above which old tool results are compacted.
    agent_compact_at: int = 16_000

    # --- API ---
    # Comma-separated roots that /ingest is allowed to read. Anything outside
    # is rejected — otherwise a public deployment would let anyone index (and
    # then query) arbitrary server files.
    ingest_dirs: str = "corpus,examples,data/uploads"
    # Where uploaded documents are stored before indexing.
    upload_dir: str = "data/uploads"
    max_upload_mb: int = 5
    # Saved evaluation runs shown on the /eval dashboard.
    eval_results_dir: str = "benchmarks/results"
    # When set, /ask*, /ingest* and document writes require this value in
    # the X-API-Key header. Empty = auth disabled (local development).
    api_key: str = ""
    # Per-IP sliding-window limit for /ask* endpoints. 0 = disabled.
    rate_limit_per_minute: int = 30

    # --- public demo hardening ---
    # Disable every endpoint that changes the index (ingest, upload, delete).
    read_only: bool = False
    # Global cap on /ask* requests per UTC day, across all clients. 0 = off.
    # Bounds the worst-case LLM bill of a public deployment.
    daily_request_budget: int = 0
    # Behind a reverse proxy every request arrives from the proxy's address.
    # When true, the client IP is read from X-Forwarded-For instead. Leave
    # false unless a trusted proxy sets that header — clients can forge it.
    trust_proxy: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (parsed once, then cached)."""
    return Settings()
