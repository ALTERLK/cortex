"""Application configuration.

All runtime configuration lives here, loaded from environment variables
(or a local `.env` file, which is git-ignored). Code anywhere in the
project asks for `get_settings()` instead of reading `os.environ`
directly, so there is exactly one place that knows how config works.

NOTE (learning): secrets such as API keys must NEVER be hardcoded or
committed to git. Anyone with repo access (or a leaked public repo)
could otherwise spend your money. The `.env` file stays on your machine;
`.env.example` documents which variables exist without their values.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, populated from environment variables / .env."""

    # --- LLM provider ---
    # DeepSeek exposes an OpenAI-compatible API, so we talk to it with the
    # `openai` SDK pointed at a different base_url.
    deepseek_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    NOTE (learning): `lru_cache` makes this a cheap singleton — the .env
    file is parsed once per process, and every caller sees the same object.
    """
    return Settings()
