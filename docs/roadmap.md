# Roadmap

What is built, what is next, and what was considered and dropped.

## Done

| Area | What exists |
|---|---|
| Ingest | `.md` / `.txt` / `.pdf` loading, recursive chunking with overlap, BGE-M3 embeddings, Qdrant local mode, incremental re-ingest by content hash |
| Retrieval | Dense, hybrid (hand-written BM25 + reciprocal rank fusion), cross-encoder reranking, follow-up query rewriting; per-passage retrieval trace |
| Answering | Cited single-pass RAG; a hand-written tool-use loop with a tool registry, stable citation numbers, step and token limits, context compaction; auto mode that escalates to the agent on failure |
| Evaluation | 50-question set in five categories; hit-rate, MRR, LLM-judged correctness, refusal and citation checks; cost and latency per question; RAG / agent / auto compared on the same questions; prompt-injection benchmark |
| Service | FastAPI with SSE streaming, document upload, background ingest with progress, structured logs with request ids, API key, rate limit, daily budget, read-only mode |
| Interfaces | Web UI (chat, evidence panel, evaluation dashboard; light/dark, English/Chinese); MCP server exposing search and ask |

## Next

Ordered by how much the evaluation says each would help.

1. **A second judge.** Correctness is graded by the same model that
   answers. A stronger, separate judge would remove the self-grading bias
   and is the cheapest way to make every number here more trustworthy.
2. **Larger, held-out eval set.** Fifty questions give wide confidence
   intervals — one question is two points — and the prompts were tuned while
   looking at this set. A second set that is never looked at would show how
   much of the measured gain is real.
3. **Streaming the agent's final answer.** Agent steps stream as they
   happen, but the answer arrives in one piece because the loop uses
   non-streaming calls. Streaming it needs tool-call deltas to be
   reassembled; the wait is the most visible rough edge in the UI.
4. **Conversation memory.** History is sent whole, capped at twelve turns.
   Summarising older turns would keep long conversations coherent without
   growing the prompt.
5. **Index at scale.** BM25 is rebuilt in memory from every chunk at
   startup. That is instant at 8k chunks and would not be at 1M; Qdrant's
   sparse vectors would move it into the store.

## Considered and dropped

- **An orchestration framework for the agent.** The loop is about a hundred
  lines. A framework would replace code that is easy to read with
  configuration that is not, and nothing measured so far needs a graph of
  agents.
- **Fine-tuning the embedding model.** Retrieval misses in the eval set are
  mostly phrasing mismatches that hybrid search and reranking already
  recover; fine-tuning costs far more than it would return here.
- **A JavaScript framework for the UI.** Three pages and a handful of
  modules do not need a build step.
