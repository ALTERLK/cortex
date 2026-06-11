"""M0 smoke test: prove the LLM plumbing works end to end.

Run with:  uv run python scripts/smoke_test.py

It performs two live calls against DeepSeek:
  1. a plain chat completion,
  2. a function-calling round (the model asks us to run a tool).

Requires DEEPSEEK_API_KEY in your .env — see .env.example.
"""

from cortex.llm import get_llm_client
from cortex.llm.deepseek import MissingAPIKeyError


def main() -> None:
    try:
        llm = get_llm_client()
    except MissingAPIKeyError as exc:
        print(f"[setup needed] {exc}")
        raise SystemExit(1)

    # --- Test 1: plain chat -------------------------------------------------
    print("=== Test 1: plain chat ===")
    response = llm.chat(
        [
            {"role": "system", "content": "You are Cortex, a knowledge assistant."},
            {"role": "user", "content": "Introduce yourself in one short sentence."},
        ]
    )
    print(f"reply:  {response.content}")
    print(f"tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    # --- Test 2: function calling -------------------------------------------
    # We describe a tool; the model should *ask us to call it* instead of
    # answering directly. We don't execute anything yet — running the tool
    # and feeding the result back is the agent loop, which is milestone M4.
    print("\n=== Test 2: function calling ===")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search the user's personal knowledge base for relevant passages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    response = llm.chat(
        [{"role": "user", "content": "What did my notes say about vector databases?"}],
        tools=tools,
    )
    if response.wants_tool_call:
        call = response.tool_calls[0]
        print(f"model requested tool: {call.name}({call.arguments})")
    else:
        print(f"model answered directly (unexpected): {response.content}")
    print(f"finish_reason: {response.finish_reason}")
    print(f"tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    print("\nSmoke test passed. M0 plumbing works.")


if __name__ == "__main__":
    main()
