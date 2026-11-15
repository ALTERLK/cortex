"""Start the Cortex API server.

Usage:
    uv run python scripts/serve.py
    uv run python scripts/serve.py --port 8080
    uv run python scripts/serve.py --host 0.0.0.0     # listen on all interfaces (containers)
    uv run python scripts/serve.py --reload           # auto-reload on code changes

For a guided start that checks the setup first, use scripts/demo.py.
"""

import argparse
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cortex API server.")
    # Localhost by default: the API has no authentication unless API_KEY is set.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    print(f"Starting Cortex on http://{args.host}:{args.port}  (API docs at /docs)")
    uvicorn.run(
        "cortex.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
