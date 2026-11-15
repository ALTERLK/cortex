FROM python:3.12-slim

WORKDIR /app

# uv, copied from its official image. Pinned so a rebuild is reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/hf_cache

# Dependencies first: this layer is rebuilt only when the lock file changes,
# not on every source edit.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Bake the embedding model into the image so a container starts without a
# 2.3 GB download. Costs image size, buys a fast and predictable start.
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY scripts/ ./scripts/
COPY benchmarks/ ./benchmarks/
COPY examples/ ./examples/

# The model is in the image: do not contact Hugging Face at runtime.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

# 0.0.0.0 so the port is reachable from outside the container.
CMD ["uv", "run", "--no-sync", "uvicorn", "cortex.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
