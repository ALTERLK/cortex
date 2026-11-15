# Retrieval-Augmented Generation: A Technical Overview

## What is RAG?

Retrieval-Augmented Generation (RAG) is an architecture that augments a large
language model with an external knowledge base at inference time. Instead of
relying solely on knowledge baked into the model's weights during training, the
system first retrieves relevant passages from a document store and then passes
those passages to the model as additional context.

The key motivation is grounding: a model answering from memory alone is prone to
hallucination — confident-sounding statements that are factually wrong. By
providing source passages alongside the question, we shift the task from
"recall from weights" to "read and summarize," which is far more reliable and
auditable.

## Core Components

A minimal RAG pipeline has six stages.

**Loading** reads raw files (.md, .txt, .pdf) from disk and produces plain text
plus metadata (filename, page number). The metadata travels with the text
through every subsequent stage so that the final answer can cite its sources.

**Chunking** splits each document into overlapping windows of ~300–500
characters. Smaller chunks improve retrieval precision but lose surrounding
context; larger chunks preserve context but dilute the relevance signal.
Chunk size is a hyperparameter tuned against a held-out evaluation set.

**Embedding** converts each chunk into a dense vector using a neural encoder
such as BGE-M3 or OpenAI's text-embedding-3-small. Semantically similar text
maps to nearby points in the vector space, enabling similarity search.

**Indexing** stores all chunk vectors in a vector database (Qdrant, Weaviate,
Pinecone). An approximate nearest-neighbor (ANN) index — typically HNSW —
makes retrieval sub-linear in the number of chunks.

**Retrieval** embeds the user's query and finds the top-k most similar chunks
via cosine similarity. The retrieved chunks become the context window passed to
the language model.

**Generation** sends a prompt containing the retrieved chunks and the original
question to the LLM. The prompt instructs the model to answer only from the
provided passages and to cite them with numbered references like [1] or [2].

## Chunking Strategies

The simplest strategy is fixed-size character splitting: every N characters
becomes a chunk, with an overlap of M characters carried into the next chunk.
Overlap prevents important sentences from being cut in half at a boundary.

Recursive character splitting improves on this by trying semantic separators in
priority order: paragraph breaks (\n\n), then line breaks (\n), then spaces,
then individual characters as a last resort. This keeps paragraphs together
when they are short enough, falling back to finer splits only when necessary.

Document-aware splitting goes further: a Markdown splitter respects heading
boundaries; a code splitter keeps functions intact; a PDF splitter aligns to
page or section boundaries. Each strategy trades implementation complexity for
better semantic coherence in the resulting chunks.

## Evaluation and Iteration

RAG quality is measured along two axes. Retrieval quality — did the right
chunks come back? — is tracked with hit-rate@k and mean reciprocal rank (MRR).
Answer quality — is the generated response correct and faithful to the sources?
— is evaluated with an LLM-as-judge that scores correctness and citation
faithfulness on a held-out question set.

The evaluation loop is the most important engineering asset in a RAG system.
Without it, changes to chunk size, embedding model, or retrieval depth are
guesswork. With it, every parameter becomes a measurable experiment.
