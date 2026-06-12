"""Fetch a real-world corpus into corpus/ (git-ignored).

Sources (all public):
  - FastAPI official docs, English + Chinese (markdown, ~250 files)
  - uv official docs (markdown, ~80 files)
  - Classic arXiv papers (PDF): Attention, RAG, ReAct, RAG survey

The corpus itself is NOT committed — this script makes it reproducible.
Each source is fetched independently: one failure doesn't stop the rest,
and already-fetched sources are skipped on re-runs (delete a subfolder or
pass --force to refresh).

Usage:
    uv run python scripts/fetch_corpus.py
    uv run python scripts/fetch_corpus.py --force
"""

import argparse
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CORPUS = Path("corpus")

# (repo URL, branch, [(repo subdir, corpus target subdir), ...])
GIT_SOURCES = [
    (
        "https://github.com/fastapi/fastapi.git",
        "master",
        [("docs/en/docs", "fastapi/en"), ("docs/zh/docs", "fastapi/zh")],
    ),
    (
        "https://github.com/astral-sh/uv.git",
        "main",
        [("docs", "uv")],
    ),
]

# arXiv id -> human-readable filename
ARXIV_PAPERS = {
    "1706.03762": "attention-is-all-you-need.pdf",
    "2005.11401": "retrieval-augmented-generation.pdf",
    "2210.03629": "react-reasoning-and-acting.pdf",
    "2312.10997": "rag-for-llms-survey.pdf",
}


def fetch_git_source(url: str, branch: str, mappings: list[tuple[str, str]], force: bool) -> None:
    todo = [(src, CORPUS / dst) for src, dst in mappings
            if force or not (CORPUS / dst).exists()]
    if not todo:
        print(f"[skip] {url} — all targets already fetched")
        return

    print(f"[clone] {url} (shallow)…")
    with tempfile.TemporaryDirectory(dir=str(CORPUS)) as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch,
             "--filter=blob:none", "--no-checkout", url, tmp],
            check=True, capture_output=True,
        )
        # Sparse checkout: only materialise the doc folders we need.
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", "--no-cone",
                        *[src for src, _ in mappings]], check=True, capture_output=True)
        subprocess.run(["git", "-C", tmp, "checkout", branch], check=True, capture_output=True)

        for src, target in todo:
            src_dir = Path(tmp) / src
            if not src_dir.exists():
                print(f"  [warn] {src} not found in repo, skipping")
                continue
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            copied = 0
            for md in src_dir.rglob("*.md"):
                rel = md.relative_to(src_dir)
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(md, dest)
                copied += 1
            print(f"  [ok] {src} -> {target} ({copied} md files)")


def fetch_arxiv(force: bool) -> None:
    import httpx

    papers_dir = CORPUS / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    for arxiv_id, filename in ARXIV_PAPERS.items():
        dest = papers_dir / filename
        if dest.exists() and not force:
            print(f"[skip] {filename}")
            continue
        # export.arxiv.org is the mirror intended for programmatic access.
        url = f"https://export.arxiv.org/pdf/{arxiv_id}"
        print(f"[pdf] {url} …")
        try:
            with httpx.Client(follow_redirects=True, timeout=120) as client:
                resp = client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
            print(f"  [ok] {filename} ({len(resp.content) // 1024} KB)")
        except Exception as exc:
            print(f"  [fail] {filename}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the public-docs corpus.")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if present")
    args = parser.parse_args()

    CORPUS.mkdir(exist_ok=True)

    for url, branch, mappings in GIT_SOURCES:
        try:
            fetch_git_source(url, branch, mappings, args.force)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode(errors="replace")[-300:]
            print(f"[fail] {url}: {stderr}")

    fetch_arxiv(args.force)

    n_md = sum(1 for _ in CORPUS.rglob("*.md"))
    n_pdf = sum(1 for _ in CORPUS.rglob("*.pdf"))
    print(f"\nCorpus: {n_md} markdown + {n_pdf} pdf files under {CORPUS}/")
    print("Next: uv run python scripts/ingest.py corpus")


if __name__ == "__main__":
    main()
