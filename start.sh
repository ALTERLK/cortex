#!/usr/bin/env sh
# One-command launcher for macOS/Linux: checks the setup, starts Cortex, opens the browser.
# Arguments are passed through, e.g.  ./start.sh --port 8080
set -e
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
    # An existing environment starts immediately and needs no network.
    exec .venv/bin/python scripts/demo.py "$@"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it from https://docs.astral.sh/uv/ and run this again."
    exit 1
fi
echo "First run: creating the environment with uv. This downloads the dependencies once."
exec uv run python scripts/demo.py "$@"
