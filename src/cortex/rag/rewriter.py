"""Query rewriting: turn a context-dependent follow-up into a standalone query.

The problem: in rag mode, retrieval embeds the RAW question. A follow-up
like "它是在响应之前还是之后执行？" ("does IT run before or after the
response?") has no retrievable content — "it" only resolves through the
conversation. Agent mode dodges this because the LLM writes its own search
queries with history in context; rag mode needs this explicit rewrite step.

NOTE (learning): this is the cheapest of the adaptive-retrieval techniques —
one small LLM call, only when history exists. The trade-off is latency
(one extra round trip before retrieval starts).
"""

from __future__ import annotations

from cortex.llm.base import LLMClient
from cortex.llm.postprocess import strip_thinking

_SYSTEM_PROMPT = """\
You rewrite follow-up questions into standalone search queries.

Given a conversation and a follow-up question, output a single self-contained
query that contains every entity the follow-up refers to. Keep the original
language of the question. Output ONLY the rewritten query — no explanations,
no quotes.\
"""


class QueryRewriter:
    """Rewrites history-dependent questions into standalone retrieval queries."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def rewrite(self, question: str, history: list[dict] | None) -> str:
        """Return a standalone query for *question*.

        Without history there is nothing to resolve — the question is
        returned unchanged and no LLM call is made.
        """
        if not history:
            return question

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": f"Follow-up question: {question}\n\nStandalone query:"},
        ]
        try:
            response = self._llm.chat(messages, temperature=0.0)
        except Exception:
            # Rewriting is an optimisation: if it fails, fall back to the
            # raw question rather than failing the whole request.
            return question

        rewritten = strip_thinking(response.content or "").strip().strip('"')
        # A wildly long "query" usually means the model explained instead of
        # rewriting — fall back to the original.
        if not rewritten or len(rewritten) > 300:
            return question
        return rewritten
