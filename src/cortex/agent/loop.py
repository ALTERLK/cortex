"""Hand-written agentic tool-use loop.

The loop drives an LLM that has access to search tools and lets it decide
how many times to retrieve information before producing a final answer.

NOTE (learning): this is the complete "agent" pattern in ~60 lines:
  1. Send messages + tool schemas to the LLM.
  2. If the LLM calls a tool, execute it and append the result as a new message.
  3. Repeat until the LLM stops calling tools (finish_reason == "stop").
  4. Guard against runaway loops with max_iterations.

No framework required. LangGraph does this for you in Phase 2; understanding
the raw loop makes you able to explain and debug any framework that wraps it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field

from cortex.agent.tools import TOOL_SCHEMAS, ToolExecutor
from cortex.llm.base import LLMClient, TokenUsage

_SYSTEM_PROMPT = """\
You are Cortex, a personal knowledge assistant. To answer questions accurately,
use the search_knowledge_base tool to find relevant information. You may search
multiple times with different, specific queries if needed.

When you have gathered enough information, produce a final answer. Cite every
claim with [N] where N is the result number returned by the search tool.
If a question cannot be answered from the knowledge base, say so honestly.\
"""


@dataclass
class ToolCallRecord:
    """One tool invocation within a single agent run."""
    name: str
    arguments: dict
    result: str


@dataclass
class AgentResponse:
    """The complete output of one agent run.

    Attributes:
        answer:      Final answer text (may contain [N] citations).
        tool_calls:  Ordered log of every tool call made.
        iterations:  Number of LLM calls made.
        usage:       Cumulative token counts across all LLM calls.
    """
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))


class AgentLoop:
    """Drives an LLM through a tool-use loop until it produces a final answer.

    Args:
        llm:            Any LLMClient implementation.
        executor:       ToolExecutor that maps tool names to real actions.
        max_iterations: Safety cap on LLM calls per run (default 10).
    """

    def __init__(
        self,
        llm: LLMClient,
        executor: ToolExecutor,
        max_iterations: int = 10,
    ) -> None:
        self._llm = llm
        self._executor = executor
        self._max_iterations = max_iterations

    def run(self, question: str, history: list[dict] | None = None) -> AgentResponse:
        """Run the agent loop for *question* and return a cited answer."""
        for event in self.run_events(question, history):
            if isinstance(event, AgentResponse):
                return event
        raise RuntimeError("run_events ended without an AgentResponse")  # unreachable

    def run_events(
        self, question: str, history: list[dict] | None = None
    ) -> Iterator[ToolCallRecord | AgentResponse]:
        """Run the loop, yielding each tool call as it happens.

        Yields ToolCallRecord items in real time (so callers like the SSE
        endpoint can show live progress), then exactly one final AgentResponse.

        NOTE (learning): turning the loop into a generator costs nothing —
        run() above just drains it. This is the standard way to retrofit
        progress reporting onto a batch algorithm without duplicating it.

        With *history*, the LLM sees prior turns — so when it writes its own
        search queries it can resolve references like "when was IT added?"
        by itself. The plain rag mode cannot do that (its retrieval query is
        the raw question); this asymmetry is a Phase 2 (query rewriting) topic.
        """
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *(history or []),
            {"role": "user", "content": question},
        ]
        tool_calls_log: list[ToolCallRecord] = []
        total_input = total_output = 0

        for iteration in range(1, self._max_iterations + 1):
            response = self._llm.chat(messages, tools=TOOL_SCHEMAS)
            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

            if not response.wants_tool_call:
                # LLM produced a final answer — we're done.
                yield AgentResponse(
                    answer=response.content or "",
                    tool_calls=tool_calls_log,
                    iterations=iteration,
                    usage=TokenUsage(total_input, total_output),
                )
                return

            # -----------------------------------------------------------------
            # The LLM requested one or more tool calls.
            # We must:
            #   (a) append the assistant message WITH the tool_calls field, and
            #   (b) append a tool-result message for each call.
            # NOTE (learning): the OpenAI protocol requires both messages to be
            # present in order; omitting either causes a 400 error.
            # -----------------------------------------------------------------
            messages.append(_assistant_message(response.content, response.tool_calls))

            for tc in response.tool_calls:
                result_text = self._executor.execute(tc.name, tc.arguments)
                record = ToolCallRecord(name=tc.name, arguments=tc.arguments, result=result_text)
                tool_calls_log.append(record)
                yield record
                messages.append(_tool_result_message(tc.id, result_text))

        # Reached max_iterations without a "stop" response.
        yield AgentResponse(
            answer="[Agent reached the maximum number of iterations without a final answer.]",
            tool_calls=tool_calls_log,
            iterations=self._max_iterations,
            usage=TokenUsage(total_input, total_output),
        )


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _assistant_message(content: str | None, tool_calls: list) -> dict:
    """Build the assistant turn that contains tool_call requests."""
    return {
        "role": "assistant",
        "content": content,
        # NOTE (learning): we reconstruct the tool_calls field in the exact
        # shape the OpenAI API expects — a list of dicts with id/type/function.
        # Our ToolCall dataclass stores the parsed arguments as a dict; we must
        # re-serialise to JSON string here because the API schema requires it.
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ],
    }


def _tool_result_message(tool_call_id: str, content: str) -> dict:
    """Build the tool-result turn that feeds execution output back to the LLM."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
