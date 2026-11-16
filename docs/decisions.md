# Design decisions

Why Cortex is built the way it is. Each entry states the choice, what it
was chosen over, and what it costs. Numbers quoted here come from
[evaluation.md](evaluation.md), which has the full tables.

## Scope

**No RAG or agent framework.** Retrieval, prompting, the agent loop and the
evaluation harness are written directly against an LLM API. The pieces are
small — the agent loop is about a hundred lines, BM25 about eighty — and
owning them means every behaviour measured in the evaluation can be traced
to a line of code and changed. The cost is that nothing comes for free:
streaming, retries, tool-call plumbing and tracing each had to be written.

**One process, local state.** The vector index is a directory on disk and
rate limits live in memory. This keeps the project runnable with one
command and no services. Everything that would break with a second
instance — the job table, the limiters, the BM25 index — is marked as such
where it is defined.

## LLM access

**A protocol, and one adapter.** Application code depends on `LLMClient`,
a `typing.Protocol` with `chat` and `chat_stream`. A protocol is checked
structurally, so test fakes are plain classes with no inheritance. The one
real implementation speaks the OpenAI chat-completions format, which nearly
every provider and gateway accepts; switching provider is three environment
variables.

**Provider quirks stop at the adapter.** Three were met in practice:

- Tool-call arguments arrive as a JSON string and are parsed once, at the
  boundary.
- Some reasoning models write their reasoning into the content inside
  `<thinking>` tags — and not always spelled that way. The filter accepts
  the variants actually observed, and its streaming form holds back any
  trailing fragment that could still become a tag, because a tag can be
  split across two chunks.
- Newer models reject `temperature`. The adapter retries once without it
  and remembers, so callers keep asking for low randomness and the request
  succeeds either way. The price is one failed request per process.

**Token prices are configuration.** The cost shown in the UI is an
estimate from two numbers in `.env`, not something the code knows about any
model.

## Ingest

**Recursive character chunking.** Text is split on paragraph breaks, then
lines, then spaces, falling to a finer separator only for pieces that are
still too large. Chunks are 400 characters with 50 of overlap. Overlap is
carried in whole pieces rather than exact characters: hitting the number
exactly would mean cutting a sentence in half.

**Deterministic ids and a content hash.** A chunk's id is derived from its
source path and index, so re-ingesting overwrites instead of duplicating.
Each chunk also stores the hash of its file: an unchanged file is skipped,
and a changed file has its old chunks deleted first — otherwise a file that
shrinks would leave its last chunks behind.

**Source labels are paths, not file names.** The FastAPI docs exist in
English and Chinese with identical file names. Labelling chunks by path
relative to the ingest root keeps them apart, and uploaded documents get an
`uploads/` prefix so they cannot collide with the corpus.

## Retrieval

**Hybrid, because the two retrievers fail differently.** Embeddings find
passages that mean the same thing in different words and across languages;
BM25 finds exact rare tokens — flag names, error codes — that embeddings
blur. On the evaluation set, adding BM25 raised hit-rate@5 from 77.8% to
82.2%, almost all of it on Chinese questions over English documents.

**BM25 is written by hand.** It is short, and the part that matters for
this corpus is the tokenizer: Chinese has no spaces, so text containing CJK
characters is segmented with jieba while Latin text is split by regex. A
library would have hidden that choice.

**Reciprocal rank fusion, not score mixing.** Cosine similarities and BM25
scores are on unrelated scales, so a weighted sum needs normalising and
tuning. RRF uses only each passage's rank in each list; it has one constant
and no tuning.

**The sparse index is derived, and rebuilt atomically.** Qdrant is the only
persistent store; BM25 is built in memory from its chunks. When an ingest
job finishes, a new index is built on the side and swapped in with one
assignment, so a concurrent query sees the old index or the new one, never
a mixture. Before this, documents added through the API were reachable by
dense search only until the next restart.

**Reranking works, and is off by default.** A cross-encoder that reads
query and passage together re-scores the fused top 20. It raises hit-rate
to 84.4% and helps multi-hop questions most — and on CPU it adds about ten
seconds to a query. That is the wrong trade for an interactive demo, so the
default strategy is `hybrid` and reranking is a setting.

**Follow-ups are rewritten, single questions are not.** "Does it run before
or after?" contains nothing to search for. With history present, one small
LLM call turns it into a standalone query; without history there is no
call. Any failure falls back to the original question.

**Every passage carries its trace.** Each stage records what it knows — the
rank in the dense list, the rank in the BM25 list, the fused score, the
rerank score — on the passage itself. This was added for the UI and turned
out to be the fastest way to debug retrieval.

## Generation

**Passages and question share one user message.** Numbered passages sit
directly above the question. Citation instructions are followed more
reliably when the things to cite are adjacent to the request.

**The refusal sentence is fixed, in English.** When the passages do not
answer the question, the model must open with one exact sentence, whatever
language the question is in. A fixed sentence is what makes refusals
detectable without a judge — in the evaluation, and at run time for auto
mode. Models do not always comply: one agent answer refused correctly in
Chinese, so the detector also accepts the usual Chinese phrasings.

**Retrieved text is fenced and declared to be data.** Passages go inside
`<context>` tags and the system prompt says nothing inside them is an
instruction. The effect is measured: on eight attacks run three times each,
2 of 24 succeeded without the guard and 0 of 24 with it. The one attack
that worked imitated a system message — the kind a fence addresses.

## Agent

**A plain loop.** Send the conversation and the tool schemas; run the tools
the model asks for; repeat until it answers in text. The loop is a
generator that yields each step as it happens, and the non-streaming entry
point just drains it, so there is one implementation.

**Schemas live next to their implementation.** Tools are methods marked
with `@tool`, and the executor collects them at construction. Adding a tool
is adding a method; the schema cannot drift from the code because they are
written together.

**Citation numbers belong to a run, not to a search.** Originally each
search numbered its results from `[1]`, so after two searches `[2]` was
ambiguous — and the API returned no sources for agent answers at all, so
citations could not be clicked. Now a per-run table gives each passage one
number for the whole run. It is per-run rather than on the executor
because the executor is shared by every request.

**Tool failures are observations.** An unknown tool, a wrong argument type
or an exception inside a tool is returned to the model as an error message
it can act on, rather than ending the run.

**A repeated search says so.** Passages the run already has are not printed
again; a search that returns nothing new says exactly that. Without it the
model kept rephrasing the same query — in an early run one question took
ten searches — and every repeat was re-sent on each later turn. In the
final evaluation the longest run used five.

**Limits end in an answer, not an error.** At the step limit or the token
budget the model is told to answer from what it has. Only if it still
produces nothing does the run report that it stopped.

**Compaction is triggered by the provider's own count.** Every turn
re-sends the whole history, so cost grows with the square of the run
length. When the prompt size the provider reported for the last call
passes a threshold, older tool results are replaced by one-line stubs that
keep their citation numbers. Using the reported count avoids guessing with
a tokenizer. On this corpus runs are short and the guard rarely fires; it
exists for long runs and is covered by tests.

## Auto mode

**The hypothesis was wrong, and the default reflects that.** The plan was a
router: classify each question up front, send hard ones to the agent, and
get agent quality at close to RAG cost. Measured, the classifier sent 4% of
questions to the agent, gained almost nothing, and cost more than RAG on
every request because it adds a call to each one.

**Escalate on failure instead.** Auto mode answers with RAG and hands over
to the agent only when that answer is a refusal or cites nothing — two
conditions that need no LLM call to detect. It scored 4.02 against the
classifier's 3.78 for 15% more cost, and an answer that is not escalated
costs exactly what RAG costs.

**Its ceiling is known, and so is its waste.** Escalation closed about a
third of the gap between RAG (3.72) and the agent (4.68). Weak RAG answers
are mostly grounded but incomplete, and nothing short of a judge can see
that. And a refusal is one of the triggers, so a correctly refused
unanswerable question is escalated too: five of the eleven escalations in
the evaluation were exactly that, each paying agent prices to reach the same
answer. If quality matters more than cost, the agent is the right mode;
auto is the cost-aware default, not a free lunch. The classifier is kept,
selectable, so the comparison can be re-run.

## Evaluation

**Categories, including questions with no answer.** The set has factual,
cross-lingual, multi-hop and follow-up questions, plus traps the corpus
cannot answer. A single blended score would hide where each change helps;
the retrieval stages each fixed a different category.

**Pipelines behind one interface.** RAG, agent and auto each implement
`run(question)`, so the loop that scores them cannot tell them apart and
the comparison is like for like.

**Cost is measured, not estimated.** The system under test talks to the LLM
through a wrapper that counts every call — including the rewrite and
routing calls a pipeline makes internally. The judge uses an unwrapped
client so grading is not charged to the pipeline.

**Deterministic where possible.** Trap questions are scored by the refusal
check, not the judge. Citations are checked by matching numbers against the
source list. Only the correctness of real answers needs an LLM.

**Known weaknesses are written down.** Fifty questions, one judge that is
also the answering model, no sampling control on current models, and
prompts adjusted while looking at this set. [evaluation.md](evaluation.md)
lists what each of these does to the numbers.

## API

**Two endpoints, one generator.** `/ask/stream` forwards `(event, payload)`
pairs as Server-Sent Events; `/ask` folds the same pairs into one JSON
body. They cannot disagree because there is nothing to keep in sync.

**Errors after the first byte go in-band.** Once a stream has started the
status code is already sent, so a provider failure is reported as an
`error` event. Before that point it is a 502.

**The server holds no conversation.** The client sends prior turns with
each request and the server forwards at most twelve. Roles are restricted
to user and assistant so a client cannot inject a system message through
the history.

**Long work returns 202.** Ingest and upload start a background job and
return its id; the client polls for progress. Embedding a large folder
takes minutes, which is longer than a request should stay open.

**Uploads are a raw request body.** The file is streamed and counted as it
arrives, so an oversized upload is cut off at the limit rather than
buffered first, and no multipart parser is needed. The file name is
reduced to its base name and a safe character set, and only document types
the loader understands are accepted.

**Limits in layers.** An optional API key compared in constant time; a
sliding-window limit per client; a daily budget across all clients, which
is what actually bounds the bill of a public demo; and a read-only switch.
Behind a proxy the client address must come from `X-Forwarded-For`, which
is only trusted when explicitly enabled.

## Web UI

**Plain HTML, CSS and ES modules.** There is no build step: the browser
loads the modules as written, and the server ships them as static files.
For three views this is less machinery than a framework's toolchain. The
cost is that state handling is by hand — each message is one object,
stream events update it, and small functions redraw the part that changed.

**Untrusted text never becomes markup by concatenation.** Answers are
markdown from a model that has read arbitrary documents. They are converted
to HTML, sanitized against an allowlist of formatting tags, and only then
are citation markers replaced — by walking text nodes and creating
elements, never by editing the HTML string. Images are not allowed, since
an image URL in an answer would make the browser fetch it.

**Dependencies are vendored.** The two libraries the UI uses are checked in
at pinned versions. The interface works with no internet connection, which
matters for a demo.

**Blur only where nothing scrolls under it.** `backdrop-filter` is limited
to the composer and the mobile top bar. A blur over scrolling content is
recomputed every frame.

## Deployment

**The launcher checks before it starts.** A missing key, an empty index or
a provider that does not answer each produce one line saying what to do,
instead of a stack trace thirty seconds into startup.

**Offline when the model is cached.** With the embedding model on disk the
launcher sets `HF_HUB_OFFLINE`, because the libraries otherwise check for
updates at startup and that check hangs where the hub is unreachable.

**Secrets stay out of images.** The original Dockerfile copied the whole
project into the image and there was no `.dockerignore`, so `.env` would
have gone with it. Secrets are now excluded at build time and injected at
run time.
