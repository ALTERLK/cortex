"""Knowledge-base endpoints: what is indexed, and how to add to it.

  GET    /stats                 — system summary for the UI
  GET    /documents             — indexed documents with chunk counts
  PUT    /documents/{filename}  — upload one document and index it (202 + job)
  DELETE /documents/{source}    — remove an uploaded document
  POST   /ingest                — index an allowlisted directory (202 + job)
  GET    /ingest/{job_id}       — poll a job's progress

Indexing takes seconds to minutes (embedding is the bottleneck), so writes
return 202 Accepted with a job id and the client polls for progress instead
of holding a request open.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response

from cortex.api.schemas import (
    DocumentInfo,
    DocumentList,
    IngestAccepted,
    IngestRequest,
    IngestStatus,
    Stats,
)
from cortex.config import get_settings
from cortex.ingest.pipeline import ingest_directory

router = APIRouter()
logger = logging.getLogger("cortex.api")

# Uploaded documents are labelled "uploads/<name>" so they can never collide
# with a corpus document of the same name, and so the API can tell which
# documents it is allowed to delete.
UPLOAD_PREFIX = "uploads/"
_ALLOWED_SUFFIXES = {".md", ".txt", ".pdf"}
_UNSAFE_CHARS = re.compile(r"[^\w.\-]+")

# In-memory job table. Survives only the process lifetime — acceptable for a
# single-instance service; a multi-instance deployment would move it to Redis.
_INGEST_JOBS: dict[str, dict] = {}


def package_version() -> str:
    try:
        return version("cortex")
    except PackageNotFoundError:  # running from a checkout that was never installed
        return "0.0.0"


def _require_writable() -> None:
    if get_settings().read_only:
        raise HTTPException(status_code=403, detail="This deployment is read-only")


def _document_counts(state: Any) -> dict[str, int]:
    """Chunk count per document, cached until the index next changes.

    Counting means scrolling every point in the collection, so the result
    is kept on app.state and dropped whenever a job modifies the index.
    """
    counts = getattr(state, "document_counts", None)
    if counts is None:
        counts = state.store.source_counts()
        state.document_counts = counts
    return counts


def _index_changed(state: Any) -> None:
    """Bring derived state back in line after documents were added or removed."""
    state.document_counts = None
    # BM25 is built in memory from the chunks in Qdrant. Without this, a
    # newly indexed document would be reachable by dense search only.
    refresh = getattr(state.retriever, "refresh", None)
    if refresh is not None:
        refresh()


# --------------------------------------------------------------------- read


@router.get("/stats", response_model=Stats)
def stats(request: Request) -> Stats:
    settings = get_settings()
    counts = _document_counts(request.app.state)
    return Stats(
        version=package_version(),
        documents=len(counts),
        chunks=sum(counts.values()),
        retrieval_mode=settings.retrieval_mode,
        query_rewrite=settings.query_rewrite,
        llm_model=settings.llm_model,
        read_only=settings.read_only,
        max_upload_mb=settings.max_upload_mb,
    )


@router.get("/documents", response_model=DocumentList)
def documents(request: Request) -> DocumentList:
    counts = _document_counts(request.app.state)
    docs = [
        DocumentInfo(source=source, chunks=n, uploaded=source.startswith(UPLOAD_PREFIX))
        for source, n in sorted(counts.items())
    ]
    # Uploaded documents first: they are the ones a user just added.
    docs.sort(key=lambda d: not d.uploaded)
    return DocumentList(documents=docs, total_chunks=sum(counts.values()))


# -------------------------------------------------------------------- write


def _run_ingest_job(job_id: str, directory: Path, state: Any, *,
                    chunk_size: int = 400, overlap: int = 50, source_prefix: str = "") -> None:
    t0 = time.perf_counter()
    job = _INGEST_JOBS[job_id]

    def on_progress(done: int, total: int) -> None:
        job.update(files_done=done, files_total=total)

    try:
        n = ingest_directory(
            directory, state.store, state.embedder,
            chunk_size=chunk_size, overlap=overlap, verbose=False,
            source_prefix=source_prefix, on_progress=on_progress,
        )
        _index_changed(state)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        job.update(status="done", chunks_stored=n, latency_ms=latency_ms)
        logger.info(json.dumps({
            "event": "ingest", "job_id": job_id, "directory": str(directory),
            "chunks_stored": n, "latency_ms": latency_ms,
        }))
    except Exception as exc:  # noqa: BLE001 — the job must record any failure
        job.update(status="failed", error=str(exc))
        logger.error(json.dumps({"event": "ingest_failed", "job_id": job_id, "error": str(exc)}))


def _allowed_ingest_roots() -> list[Path]:
    dirs = get_settings().ingest_dirs.split(",")
    return [Path(d.strip()).resolve() for d in dirs if d.strip()]


@router.post("/ingest", response_model=IngestAccepted, status_code=202)
def ingest(body: IngestRequest, request: Request, background: BackgroundTasks) -> IngestAccepted:
    """Start indexing a directory in the background; poll GET /ingest/{job_id}."""
    _require_writable()
    directory = Path(body.directory).resolve()
    # Only allowlisted roots may be indexed. Without this, anyone who can
    # reach the API could index arbitrary server files and read them back
    # through /ask.
    if not any(directory == root or directory.is_relative_to(root)
               for root in _allowed_ingest_roots()):
        raise HTTPException(
            status_code=403,
            detail=f"Directory not in the ingest allowlist ({get_settings().ingest_dirs})",
        )
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    job_id = uuid.uuid4().hex
    _INGEST_JOBS[job_id] = {"status": "running", "directory": str(directory)}
    background.add_task(
        _run_ingest_job, job_id, directory, request.app.state,
        chunk_size=body.chunk_size, overlap=body.overlap,
    )
    return IngestAccepted(job_id=job_id, status="running")


@router.get("/ingest/{job_id}", response_model=IngestStatus)
def ingest_status(job_id: str) -> IngestStatus:
    job = _INGEST_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown ingest job: {job_id}")
    return IngestStatus(**job)


def safe_filename(name: str) -> str:
    """Reduce a client-supplied name to a safe file name, or raise 400/415.

    The name is untrusted: `Path(name).name` drops any directory part (so
    "../../etc/passwd" becomes "passwd"), and everything outside word
    characters, dots, and dashes is replaced. Only document types the
    loader understands are accepted.
    """
    cleaned = _UNSAFE_CHARS.sub("_", Path(name.replace("\\", "/")).name).strip("._")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid file name")
    if Path(cleaned).suffix.lower() not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(_ALLOWED_SUFFIXES))}",
        )
    return cleaned


@router.put("/documents/{filename}", response_model=IngestAccepted, status_code=202)
async def upload_document(filename: str, request: Request, background: BackgroundTasks) -> IngestAccepted:
    """Upload one document (the raw file is the request body) and index it.

    The body is read as a stream and counted as it arrives, so an oversized
    upload is rejected at the limit instead of after buffering all of it.
    A raw body also avoids a multipart parser dependency.
    """
    _require_writable()
    settings = get_settings()
    name = safe_filename(filename)
    limit = settings.max_upload_mb * 1024 * 1024

    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(status_code=413, detail=f"File larger than {settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / name).write_bytes(bytes(data))

    job_id = uuid.uuid4().hex
    _INGEST_JOBS[job_id] = {"status": "running", "directory": str(upload_dir)}
    # Re-indexes the upload folder. Ingest is incremental (content hash per
    # file), so only the new or changed file is actually embedded.
    background.add_task(
        _run_ingest_job, job_id, upload_dir, request.app.state, source_prefix=UPLOAD_PREFIX,
    )
    return IngestAccepted(job_id=job_id, status="running", source=UPLOAD_PREFIX + name)


@router.delete("/documents/{source:path}", status_code=204)
def delete_document(source: str, request: Request) -> Response:
    """Remove an uploaded document from the index and from disk."""
    _require_writable()
    # Only documents that came in through the upload endpoint are deletable;
    # the base corpus is managed by the ingest scripts.
    if not source.startswith(UPLOAD_PREFIX):
        raise HTTPException(status_code=403, detail="Only uploaded documents can be deleted")
    state = request.app.state
    if source not in _document_counts(state):
        raise HTTPException(status_code=404, detail=f"Unknown document: {source}")

    state.store.delete_source(source)
    name = safe_filename(source.removeprefix(UPLOAD_PREFIX))
    (Path(get_settings().upload_dir) / name).unlink(missing_ok=True)
    _index_changed(state)
    return Response(status_code=204)
