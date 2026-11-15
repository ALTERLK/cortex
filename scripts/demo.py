"""One-command launcher: check the setup, start the server, open the browser.

    uv run python scripts/demo.py            (or double-click start.bat on Windows)

It runs the checks that otherwise fail late or confusingly — a missing API
key, an empty index, a model that would have to be downloaded, a port that
is already taken — and says what to do about each one before starting.

Options:
    --port N        preferred port (the next free one is used if it is taken)
    --no-browser    do not open a browser window
    --skip-llm      skip the live LLM check (start even if the provider is down)
"""

import argparse
import io
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # relative paths (.env, data/) are resolved from the repo root

from cortex.config import get_settings

EMBEDDING_MODEL_DIR = "models--BAAI--bge-m3"


def ok(message: str) -> None:
    print(f"  [ok]   {message}")


def warn(message: str) -> None:
    print(f"  [warn] {message}")


def fail(message: str, fix: str) -> None:
    print(f"  [FAIL] {message}\n         -> {fix}")
    raise SystemExit(1)


def check_api_key() -> None:
    settings = get_settings()
    if not Path(".env").exists():
        fail(".env not found", "copy .env.example to .env and set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL")
    if not settings.llm_api_key:
        fail("LLM_API_KEY is empty", "set LLM_API_KEY in .env")
    ok(f"LLM configured: {settings.llm_model} at {settings.llm_base_url}")


def check_index() -> None:
    store = Path(get_settings().store_path)
    size_mb = sum(f.stat().st_size for f in store.rglob("*") if f.is_file()) / 1e6 if store.exists() else 0
    if size_mb < 0.1:
        fail(
            f"no index found at {store}",
            "index some documents first:  uv run python scripts/fetch_corpus.py  "
            "then  uv run python scripts/ingest.py corpus   (or: scripts/ingest.py examples for a tiny demo)",
        )
    ok(f"index found: {store} ({size_mb:.0f} MB)")


def check_embedding_model() -> None:
    """Use the local model cache without touching the network when possible.

    With the model already cached, HF_HUB_OFFLINE=1 stops the Hugging Face
    libraries from checking for updates at startup — a check that hangs for
    a long time on networks where huggingface.co is unreachable.
    """
    default_cache = Path.home() / ".cache" / "huggingface"
    hf_home = os.environ.get("HF_HOME") or get_settings().hf_home or str(default_cache)
    if (Path(hf_home) / "hub" / EMBEDDING_MODEL_DIR).exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        ok(f"embedding model cached in {hf_home} (offline mode on)")
    else:
        warn("embedding model not cached yet: the first start downloads BGE-M3 (~2.3 GB)")


def check_llm() -> None:
    """One tiny live call, so a bad key or a dead network shows up now."""
    from cortex.llm import get_llm_client
    from cortex.llm.base import LLMUnavailableError

    try:
        started = time.perf_counter()
        get_llm_client().chat([{"role": "user", "content": "Reply with the single word: ok"}])
        ok(f"LLM provider answered in {time.perf_counter() - started:.1f}s")
    except LLMUnavailableError as exc:
        fail(f"LLM provider call failed: {str(exc)[:160]}",
             "check the key, base URL, model name and your network; or pass --skip-llm to start anyway")


def free_port(preferred: int) -> int:
    """Return `preferred` if it is free, else the next free port after it."""
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:  # nothing is listening
                return port
    fail(f"no free port in {preferred}-{preferred + 49}", "close another server or pass --port")
    raise AssertionError  # unreachable: fail() exits


def open_when_ready(url: str) -> None:
    """Poll /health, then open the browser. Runs beside the server thread."""
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1) as response:
                if response.status == 200:
                    print(f"\n  Cortex is ready: {url}\n  (Ctrl+C here to stop)\n")
                    webbrowser.open(url)
                    return
        except OSError:
            time.sleep(0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the setup and start Cortex.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    print("Cortex — checking the setup")
    check_api_key()
    check_index()
    check_embedding_model()
    if not args.skip_llm:
        check_llm()

    port = free_port(args.port)
    if port != args.port:
        warn(f"port {args.port} is in use; using {port}")
    url = f"http://localhost:{port}"
    print(f"\nStarting the server on {url} (loading the embedding model takes a few seconds)…")
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    import uvicorn

    # Bound to localhost: a demo launcher should not expose the API to the network.
    uvicorn.run("cortex.api.app:create_app", factory=True, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
