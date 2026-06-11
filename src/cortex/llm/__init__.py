"""LLM provider abstraction.

Import from this package, never from a concrete provider module:

    from cortex.llm import get_llm_client

so the rest of the codebase stays provider-agnostic.
"""

from cortex.config import get_settings
from cortex.llm.base import LLMClient, LLMResponse, TokenUsage, ToolCall
from cortex.llm.deepseek import DeepSeekClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
    "ToolCall",
    "DeepSeekClient",
    "get_llm_client",
]


def get_llm_client() -> LLMClient:
    """Build the configured LLM client.

    This factory is the single switch point for providers: when we add a
    second provider (e.g. Claude or GPT), only this function and a config
    value change — no call site anywhere else is touched.
    """
    settings = get_settings()
    return DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
