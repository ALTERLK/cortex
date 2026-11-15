"""Start the Cortex MCP server (stdio transport).

Not meant to be run by hand: the MCP client launches this process itself
and talks to it over stdin/stdout.

Register with an MCP client, for example Claude Code:
    claude mcp add cortex -- uv run --directory <path-to-repo> python scripts/mcp_serve.py
"""

from cortex.mcp_server import mcp

if __name__ == "__main__":
    # Import torch and friends in the MAIN thread before serving. FastMCP runs
    # tool functions in a worker thread, and importing torch from a non-main
    # thread deadlocks on Windows. The model itself still loads lazily on the
    # first tool call — only the import is warmed here.
    import sentence_transformers  # noqa: F401

    mcp.run()
