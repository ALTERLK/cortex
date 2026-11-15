"""Run trace: a flat list of timed steps for one agent run.

An agent run is a sequence of LLM calls and tool calls. Recording each one
with its duration and token usage answers the questions that matter when an
agent is slow or expensive: which step took the time, and where the tokens
went. The same spans feed the UI timeline and the structured logs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class Span:
    """One timed step of an agent run."""

    kind: Literal["llm", "tool"]
    name: str            # "llm" spans: what the call was for; "tool" spans: the tool name
    ms: float            # wall-clock duration
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict:
        return asdict(self)
