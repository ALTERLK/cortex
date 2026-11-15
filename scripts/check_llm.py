"""Check that the configured LLM provider works end to end.

Makes two live calls: a plain chat completion, and a function-calling round
in which the model should ask for a tool instead of answering. Use it after
editing .env to confirm the key, base URL, and model name are right.

Usage:
    uv run python scripts/check_llm.py
"""

import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from cortex.llm import MissingAPIKeyError, get_llm_client

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the user's knowledge base for relevant passages.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"],
            },
        },
    }
]


def main() -> None:
    try:
        llm = get_llm_client()
    except MissingAPIKeyError as exc:
        print(f"[setup needed] {exc}")
        raise SystemExit(1) from exc

    print("1. Plain chat")
    response = llm.chat([
        {"role": "system", "content": "You are Cortex, a knowledge assistant."},
        {"role": "user", "content": "Introduce yourself in one short sentence."},
    ])
    print(f"   reply : {response.content}")
    print(f"   tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    print("\n2. Function calling")
    response = llm.chat(
        [{"role": "user", "content": "What did my notes say about vector databases?"}],
        tools=_TOOLS,
    )
    if not response.wants_tool_call:
        print(f"   FAILED: the model answered directly instead of calling the tool:\n   {response.content}")
        raise SystemExit(1)
    call = response.tool_calls[0]
    print(f"   tool requested: {call.name}({call.arguments})")
    print(f"   tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    print("\nLLM provider OK.")


if __name__ == "__main__":
    main()
