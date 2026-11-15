"""Query router: decide whether a question needs the agent at all.

The agent loop is the more capable pipeline, and the more expensive one:
several LLM calls instead of one, each re-sending a growing context. Most
questions do not need it. The router makes one small classification call
and sends a question to the agent only when a single search is unlikely to
cover it; everything else takes the single-shot RAG path.

The classification is a hint, not a gate: any failure — provider error,
unparseable output — falls back to RAG, the cheap path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from cortex.llm.base import LLMClient, TokenUsage
from cortex.llm.postprocess import strip_thinking

_SYSTEM_PROMPT = """\
You route questions for a document question-answering system that has two
pipelines:

rag   - one search, then answer. Fast and cheap. Correct for any question
        that a single passage can answer: a definition, a fact, a default
        value, how to use one feature.
agent - the model plans several searches and reads around the results.
        Slower and several times more expensive. Correct only when one
        search cannot cover the question: comparing two things, combining
        facts from different documents, or a question with several parts.

When unsure, choose rag.

Reply with ONLY a JSON object, no other text:
{"route": "rag" or "agent", "reason": "<at most 12 words>"}\
"""

_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass(frozen=True)
class Route:
    """The router's decision for one question."""

    mode: Literal["rag", "agent"]
    reason: str
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))


class QueryRouter:
    """Classifies a question as single-shot (rag) or multi-step (agent)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def route(self, question: str, history: list[dict] | None = None) -> Route:
        """Return the pipeline to use for *question*."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *(history or []),
            {"role": "user", "content": f"Question: {question}"},
        ]
        try:
            response = self._llm.chat(messages, temperature=0.0)
        except Exception:  # noqa: BLE001 — routing must never fail the request
            return Route("rag", "router unavailable, using the default path")

        match = _JSON_OBJECT.search(strip_thinking(response.content or ""))
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = {}
            if data.get("route") in ("rag", "agent"):
                reason = str(data.get("reason", "")).strip()[:120]
                return Route(data["route"], reason, response.usage)
        return Route("rag", "router output not understood, using the default path", response.usage)
