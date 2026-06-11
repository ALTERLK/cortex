"""Start the Cortex API server.

Usage:
    uv run python scripts/serve.py
    uv run python scripts/serve.py --port 8080
    uv run python scripts/serve.py --reload          # auto-reload on code changes
"""

import argparse
import io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cortex API server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    print(f"Starting Cortex API on http://{args.host}:{args.port}")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(
        "cortex.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
