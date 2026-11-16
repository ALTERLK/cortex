# Docker Space for the public Cortex demo.
# Replace <user> with the Hugging Face account that holds the index dataset.
# See docs/deployment.md in the main repository for the full procedure.

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; everything it writes must be owned by it.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/hf_cache \
    PIP_NO_CACHE_DIR=1
WORKDIR /home/user/app

# The application, at a pinned release.
ARG CORTEX_REF=v1.1.0
RUN git clone --depth 1 --branch ${CORTEX_REF} https://github.com/ALTERLK/cortex.git .

# CPU-only PyTorch first: the default Linux wheel bundles CUDA libraries that
# add gigabytes to the image and are never used here.
RUN pip install --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --user .

# Bake in the embedding model and the prebuilt index, so the Space starts
# without downloading or embedding anything.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')" \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download('<user>/cortex-demo-index', repo_type='dataset', local_dir='data/qdrant')"

# Public-demo settings (see docs/deployment.md → Settings for anything public).
# LLM_API_KEY, LLM_BASE_URL and LLM_MODEL come from the Space's secrets.
ENV HF_HUB_OFFLINE=1 \
    READ_ONLY=true \
    TRUST_PROXY=true \
    RATE_LIMIT_PER_MINUTE=10 \
    DAILY_REQUEST_BUDGET=300 \
    RETRIEVAL_MODE=hybrid

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "cortex.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
