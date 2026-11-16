# Deployment

Three ways to run Cortex, from simplest to most exposed.

| | Local launcher | Docker | Hugging Face Space |
|---|---|---|---|
| For | development, demos | a server you control | a public demo link |
| Needs | Python 3.12+, uv | Docker | a Hugging Face account |
| Index | `data/qdrant` on disk | mounted volume | baked into the image |
| Status | tested | **not tested** — see below | prepared, not yet built |

## Local launcher

```sh
start.bat            # Windows: double-click, or run in a terminal
./start.sh           # macOS / Linux
```

Both run `scripts/demo.py`, which checks the setup before starting:

1. `.env` exists and `LLM_API_KEY` is set;
2. the index at `data/qdrant` is not empty;
3. the embedding model is cached — if it is, `HF_HUB_OFFLINE=1` is set so
   startup does not wait on huggingface.co;
4. the LLM provider answers one small request (skip with `--skip-llm`).

It then picks a free port starting at 8000, starts the server bound to
`127.0.0.1`, waits for `/health`, and opens the browser. A failed check
prints what is wrong and what to run to fix it.

First-time setup, before the launcher can pass its checks:

```sh
uv sync                                   # install dependencies
copy .env.example .env                    # then set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
uv run python scripts/fetch_corpus.py     # optional: the 360-document demo corpus
uv run python scripts/ingest.py corpus    # build the index (downloads BGE-M3, ~2.3 GB, once)
```

To try it without the corpus, index the sample document instead:
`uv run python scripts/ingest.py examples`.

## Docker

```sh
docker compose up --build
```

The image installs dependencies from `uv.lock`, bakes in the embedding
model, and starts uvicorn on port 8000. `data/` is mounted as a volume, so
the index and uploaded documents outlive the container; `.env` is passed at
run time and `.dockerignore` keeps it out of the image.

**This path has not been run.** The machine this project is developed on
has no Docker, so the Dockerfile is written against the documented behaviour
of uv and has not been built. Expect to adjust it on first use — the most
likely issue is image size, since `uv.lock` resolves the default PyTorch
build, which on Linux includes CUDA libraries the CPU-only image does not
need.

## Hugging Face Space (public demo)

`deploy/huggingface/` holds what a Docker Space needs: a `README.md` with
the Space metadata and a `Dockerfile`. The Space builds from the public
GitHub repository at a pinned tag, installs CPU-only PyTorch, and downloads
a prebuilt index from a Hugging Face dataset repository.

Steps, with `<user>` standing for your Hugging Face username:

1. **Publish the index.** The index is 84 MB of derived data and is not in
   git. Upload it once:
   ```sh
   hf auth login
   hf upload <user>/cortex-demo-index data/qdrant . --repo-type dataset
   ```
2. **Create the Space.** New Space → SDK: Docker → blank. Copy the two files
   from `deploy/huggingface/` into it and replace `<user>` in the Dockerfile.
3. **Add secrets** in the Space settings: `LLM_API_KEY`, `LLM_BASE_URL`,
   `LLM_MODEL`. Use a key with a small balance — the demo is public.
4. Push. The first build downloads PyTorch and the embedding model and takes
   several minutes.

The Space runs with the public-demo settings below. Like the Docker path,
it has **not been built yet**; the first build is the test.

## Settings for anything public

By default Cortex assumes a trusted local user: no authentication, and the
API can add documents to the index. Before exposing it:

| Setting | Effect |
|---|---|
| `READ_ONLY=true` | Disables `/ingest`, uploads, and deletes. The UI hides those controls. |
| `RATE_LIMIT_PER_MINUTE=10` | Sliding-window limit on `/ask*`, per client address. |
| `DAILY_REQUEST_BUDGET=300` | Cap on `/ask*` requests per UTC day across all clients — the upper bound on a day's LLM bill. |
| `TRUST_PROXY=true` | Take the client address from `X-Forwarded-For`. Required behind a reverse proxy, where every connection otherwise appears to come from the proxy and all users share one rate-limit bucket. Leave it off when there is no trusted proxy: a client can send any value in that header. |
| `API_KEY=…` | Require `X-API-Key` on `/ask*` and on every write. The bundled UI does not send a key, so use this for API-only deployments. |

The limits are kept in process memory. That is correct for one instance;
several instances behind a load balancer would each count separately and
would need a shared store such as Redis.

## What runs where

- **Embedding and reranking** run locally on CPU. BGE-M3 loads in a few
  seconds and embeds a query in tens of milliseconds; the cross-encoder
  reranker (`RETRIEVAL_MODE=hybrid_rerank`) adds a ~1.1 GB download and
  noticeably more time per query on CPU.
- **The LLM** is the only network dependency at query time. If the provider
  is unreachable, `/ask` returns 502 and `/ask/stream` emits an `error`
  event; nothing else is affected.
- **The index** is opened by one process at a time. Qdrant's local mode
  locks its directory, so the evaluation scripts should be pointed at a copy
  (`--store-path`) while the server is running.
