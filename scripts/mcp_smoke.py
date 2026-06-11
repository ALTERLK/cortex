"""One-shot MCP round-trip check: spawn the server over stdio, list tools,
run one real search against the local Qdrant index.

Usage:
    uv run python scripts/mcp_smoke.py
"""

import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    import time

    # Spawn the server with the SAME interpreter (the project venv) instead of
    # a nested `uv run` — nesting uv inside uv can deadlock on the project lock.
    params = StdioServerParameters(
        command=sys.executable, args=["scripts/mcp_serve.py"]
    )
    # Server stderr goes to a log file so protocol problems are debuggable.
    with open("data/mcp_server.log", "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                print("Tools:", sorted(t.name for t in tools.tools), flush=True)

                t0 = time.time()
                print("calling search_knowledge_base…", flush=True)
                result = await session.call_tool(
                    "search_knowledge_base",
                    {"query": "chunking strategy", "top_k": 2},
                )
                print(f"\n--- result in {time.time() - t0:.1f}s ---", flush=True)
                print(result.content[0].text[:400])


if __name__ == "__main__":
    anyio.run(main)
