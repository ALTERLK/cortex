"""Check the MCP server with a real stdio round trip.

Spawns the server as a child process, lists its tools, and runs one search
against the local index — the same handshake an MCP client performs.

Usage:
    uv run python scripts/check_mcp.py
"""

import sys
import time
from pathlib import Path

# UTF-8 output even on consoles whose default code page cannot encode the text.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    # Spawn the server with this interpreter (the project venv) rather than a
    # nested `uv run`, which can deadlock on the project lock.
    params = StdioServerParameters(command=sys.executable, args=["scripts/mcp_serve.py"])
    # Server stderr goes to a log file so protocol problems are debuggable.
    Path("data").mkdir(exist_ok=True)
    with open("data/mcp_server.log", "w", encoding="utf-8") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                print("Tools:", sorted(t.name for t in tools.tools), flush=True)

                started = time.time()
                result = await session.call_tool(
                    "search_knowledge_base", {"query": "chunking strategy", "top_k": 2},
                )
                print(f"\nSearch returned in {time.time() - started:.1f}s:")
                print(result.content[0].text[:400])


if __name__ == "__main__":
    anyio.run(main)
