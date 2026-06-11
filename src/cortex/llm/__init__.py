"""LLM provider abstraction.

Import from this package, never from a concrete provider module:

    from cortex.llm import get_llm_client

so the rest of the codebase stays provider-agnostic.
"""

from cortex.config import get_settings
from cortex.llm.base import LLMClient, LLMResponse, TokenUsage, ToolCall
from cortex.llm.openai_compat import MissingAPIKeyError, OpenAICompatibleClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
    "ToolCall",
    "OpenAICompatibleClient",
    "MissingAPIKeyError",
    "get_llm_client",
]


def get_llm_client() -> LLMClient:
    """Build the configured LLM client.

    This factory is the single switch point for providers: changing
    LLM_BASE_URL and LLM_MODEL in .env is all that is needed to move
    from DeepSeek to Claude to GPT — no call site elsewhere is touched.
    """
    settings = get_settings()
    return OpenAICompatibleClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
