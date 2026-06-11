FROM python:3.12-slim

WORKDIR /app

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency manifest first (layer cache: only re-runs install when deps change)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install production dependencies only (no pytest etc.)
RUN uv sync --frozen --no-dev

# Pre-download BGE-M3 so the container starts instantly (no model fetch at runtime)
ENV HF_HOME=/app/hf_cache
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy remaining files (docs, data, scripts)
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "cortex.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
