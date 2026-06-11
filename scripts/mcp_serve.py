"""Start the Cortex MCP server (stdio transport).

Not meant to be run by hand — the MCP client (Claude Code, Claude Desktop)
launches this process itself and talks to it over stdin/stdout.

Register with Claude Code from anywhere:
    claude mcp add cortex -- uv run --directory E:/projects/cortex python scripts/mcp_serve.py
"""

from cortex.mcp_server import mcp

if __name__ == "__main__":
    # Import torch & friends in the MAIN thread before serving. FastMCP runs
    # tool functions in a worker thread, and importing torch from a non-main
    # thread deadlocks on Windows. The model itself still loads lazily on
    # the first tool call — only the import is warmed here.
    import sentence_transformers  # noqa: F401

    mcp.run()
