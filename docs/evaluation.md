# Evaluation

How Cortex is measured, what the measurements say, and how far they can be
trusted. Every number here is produced by `scripts/run_eval.py`, saved under
`benchmarks/results/`, and shown on the `/eval` dashboard.

## Setup

**Corpus.** 360 public documents: the FastAPI documentation in English and
Chinese, the uv documentation, and four papers (Transformer, RAG, ReAct, a
RAG survey) — 8,342 chunks of about 400 characters.
`scripts/fetch_corpus.py` reproduces it.

**Questions.** 50, written for this corpus and paraphrased rather than
copied from it, in five categories:

| Category | n | What it tests |
|---|---|---|
| Factual | 20 | One fact from one document |
| Cross-lingual | 10 | A Chinese question whose answer is documented in English and Chinese |
| Multi-hop | 10 | An answer that needs two documents, or two parts of one |
| Follow-up | 5 | A question that only makes sense with its conversation history |
| Unanswerable | 5 | A question the corpus does not answer; the right response is a refusal |

**Model.** `claude-haiku-5-5` for answering and for judging. Runs dated
2026-11-15 and 2026-11-16.

## Metrics

| Metric | How it is computed |
|---|---|
| Hit-rate@5 | Share of answerable questions with an expected source document among the top 5 passages |
| MRR | Mean reciprocal rank of the first passage from an expected document |
| Correctness | 1–5, an LLM judge comparing the answer with a reference answer |
| Refusal accuracy | Share of unanswerable questions answered with a refusal, detected by string match — no judge |
| Valid citations | Share of answers that cite at least one retrieved passage and no number that does not exist |
| Cost | Tokens of every LLM call the pipeline made, times the configured prices; the judge is excluded |

For the agent there is no single ranked list, so its "source found" figure
is computed over every passage it retrieved during the run. It reads as
"found the document at some point" and is not comparable to hit-rate@5.

## Retrieval

Each row adds one technique to the row above. Single-pass RAG, top 5.

| Configuration | Hit-rate@5 | MRR | Correctness | Latency |
|---|---|---|---|---|
| Dense (BGE-M3) | 77.8% | 0.567 | 3.44 | 3.1 s |
| + Hybrid (BM25 + RRF) | 82.2% | 0.589 | 3.48 | 3.5 s |
| + Cross-encoder rerank | 84.4% | 0.611 | 3.76 | 13.0 s |
| + Query rewriting | 88.9% | 0.644 | 3.90 | 12.9 s |

Hit-rate by category shows that each technique fixed something different:

| | Factual | Cross-lingual | Multi-hop | Follow-up |
|---|---|---|---|---|
| Dense | 90% | 70% | 70% | 60% |
| + Hybrid | 90% | **90%** | 70% | 60% |
| + Rerank | 95% | 80% | **90%** | 40% |
| + Rewriting | 95% | 80% | 90% | **80%** |

- **Hybrid** recovered cross-lingual questions. A Chinese question embeds
  near Chinese passages; BM25 on the shared technical terms brings in the
  rest.
- **Reranking** recovered multi-hop questions, where the right passage is
  retrieved but ranked too low — and it costs about ten seconds per query on
  CPU, which is why the server default is `hybrid` without it.
- **Rewriting** recovered follow-ups. Without it, "does it run before or
  after?" is searched as written.
- Reranking lowered follow-up hit-rate (60% → 40%): a cross-encoder scoring
  an unresolved pronoun against passages has nothing to go on. Rewriting,
  applied first, removes the problem.

The retrieval metrics are deterministic. These four runs reproduce the
hit-rate and MRR first measured in June 2026 on a different model exactly.

## Answer modes

The same 50 questions answered five ways, all on the `hybrid` retriever
with query rewriting — the server's default configuration.

| Mode | Correctness | Cost / 1,000 questions | Latency | LLM calls | Tool calls | Sent to agent |
|---|---|---|---|---|---|---|
| RAG, 5 passages | 3.72 | $0.39 | 3.2 s | 1.1 | — | — |
| RAG, 10 passages | 3.88 | $0.50 | 4.4 s | 1.1 | — | — |
| Agent | **4.68** | $0.99 | 5.5 s | 2.7 | 2.1 | 100% |
| Auto — escalate on failure | 4.02 | $0.61 | 4.4 s | 1.7 | 0.5 | 22% |
| Auto — classify first | 3.78 | $0.53 | 4.7 s | 2.2 | 0.1 | 4% |

Refusal accuracy is 100% in every mode. The share of answers with valid
citations is 100% everywhere except the agent (97.8%: one answer cited a
passage number it had not retrieved). Prices are $0.10 / $0.50 per million
input / output tokens, set in `.env`.

Correctness by category:

| Mode | Factual (20) | Cross-lingual (10) | Multi-hop (10) | Follow-up (5) | Unanswerable (5) |
|---|---|---|---|---|---|
| RAG, 5 passages | 3.65 | 4.20 | 2.60 | 4.00 | 5.00 |
| RAG, 10 passages | 4.15 | 3.70 | 2.60 | 4.60 | 5.00 |
| Agent | 4.75 | 4.80 | 4.20 | 4.80 | 5.00 |
| Auto — escalate | 3.95 | 3.50 | 3.80 | 4.80 | 5.00 |
| Auto — classify | 3.85 | 3.70 | 2.60 | 4.80 | 5.00 |

What the numbers say:

- **The agent is the only change that moves multi-hop questions** (2.6 →
  4.2). Passing more passages to a single call does not: ten passages
  raised hit-rate to 93% but left multi-hop correctness at 2.6, because the
  model still has to find the second document, not just see more text. The
  agent also scored higher on factual questions, where its ability to read
  the neighbours of a hit helps a fragment become a complete answer.
- **It costs 2.5× as much and takes 1.7× as long**, and it spends tokens on
  questions that did not need it: for the five unanswerable questions every
  mode scores 5.0, and the agent is the most expensive way to get there.
- **Doubling the passages is a cheap baseline**: +0.16 correctness for +29%
  cost. That is inside the noise (see Limitations), so it is evidence
  against assuming agents are necessary, not a measured improvement.
- **The classifier did not work.** Asked to pick the hard questions up
  front, it sent 2 of 50 to the agent, correctness stayed at 3.78 (RAG:
  3.72) and every request paid for the extra call. It was written to test
  that hypothesis and is kept, selectable, so the test can be re-run.
- **Escalating on failure works partly.** It closes 31% of the
  RAG-to-agent gap (4.02 against 3.72 and 4.68) for 57% extra cost, where
  the agent costs 150% extra. It handed 11 questions to the agent: four
  multi-hop questions that RAG had failed to cite, and seven others. Five of
  those seven were the unanswerable questions, which RAG had already refused
  correctly — a refusal is the trigger, and the check cannot tell a correct
  one from a failure. Those five cost agent prices to reach the same answer.
  The weaker multi-hop answers RAG produced with citations were never
  escalated, because nothing cheap detects "grounded but incomplete".

## Prompt injection

`scripts/run_injection_eval.py` checks whether instructions hidden in a
document are obeyed. Eight attacks, each a poisoned passage placed next to
an honest one; retrieval is bypassed so the prompt is the only variable.
Every attack tries to get a canary string into the answer, so scoring needs
no judge. Three trials per case.

| | Attacks that succeeded |
|---|---|
| Passages inlined, no guard | 2 / 24 |
| Passages fenced in `<context>` and declared to be data | 0 / 24 |

Both successes were the same attack: a passage that closes a fake context
block and poses as a system message. Plain "ignore your instructions"
overrides, a role-play jailbreak, a planted false fact, a phishing link and
a request to reveal the system prompt failed in both configurations — the
model resists those on its own.

This is a small benchmark of known attack shapes. It shows the guard helps
against one of them; it does not show the system is safe against an
attacker who adapts.

## Limitations

- **Fifty questions.** One question is two points of correctness-average
  and 2.2 points of hit-rate. The retrieval ladder's steps are two to four
  questions each.
- **The judge is the system under test.** Correctness is graded by the same
  model that wrote the answer, which favours its own phrasing. Scores are
  comparable between runs with the same judge; they are not absolute.
- **No sampling control.** The model in use rejects `temperature`, so
  answers and grades vary between identical runs. Repeating a configuration
  moved its correctness by about 0.1. Differences under about 0.2 should be
  read as noise.
- **Tuned on the test set.** The prompts were adjusted while looking at
  these questions — the refusal wording and the agent's stopping rule in
  particular. The changes are general behaviours rather than answers to
  specific questions, but there is no held-out set to prove it.
- **Latency is indicative.** Runs were executed on a laptop, several at a
  time. The ten-second cost of reranking is real; small differences between
  modes are not.
- **One corpus.** Technical documentation with clear headings is friendly to
  chunking and retrieval. Results on messier documents would be lower.

## Reproducing

```sh
# Retrieval ladder
uv run python scripts/run_eval.py --retriever dense
uv run python scripts/run_eval.py --retriever hybrid
uv run python scripts/run_eval.py --retriever hybrid_rerank
uv run python scripts/run_eval.py --retriever hybrid_rerank --rewrite

# Answer modes
uv run python scripts/run_eval.py --mode rag   --rewrite
uv run python scripts/run_eval.py --mode rag   --rewrite --top-k 10
uv run python scripts/run_eval.py --mode agent --rewrite
uv run python scripts/run_eval.py --mode auto  --rewrite --auto-strategy escalate
uv run python scripts/run_eval.py --mode auto  --rewrite --auto-strategy classify

# Prompt injection
uv run python scripts/run_injection_eval.py
```

Add `--limit 5` for a quick check. While the server is running, point the
scripts at a copy of the index with `--store-path`: Qdrant's local mode
allows one process per directory.
