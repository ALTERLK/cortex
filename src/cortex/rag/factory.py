"""Build a retriever from a strategy name.

The retrieval strategy is configuration, not code: the API server and the
evaluation harness both call this function, so an A/B comparison measures
exactly the retriever that serves requests.
"""

from __future__ import annotations

from cortex.ingest.embedder import Embedder
from cortex.ingest.store import VectorStore
from cortex.rag.hybrid import HybridRetriever
from cortex.rag.retriever import Retriever

STRATEGIES = ("dense", "hybrid", "hybrid_rerank")


def build_retriever(store: VectorStore, embedder: Embedder, strategy: str, top_k: int = 5) -> object:
    """Return the retriever for *strategy*.

    dense          embedding similarity only
    hybrid         dense + BM25, fused with reciprocal rank fusion
    hybrid_rerank  hybrid candidates re-scored by a cross-encoder
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown retrieval strategy '{strategy}' (expected one of {STRATEGIES})")
    if strategy == "dense":
        return Retriever(store, embedder, top_k=top_k)
    retriever = HybridRetriever(store, embedder, top_k=top_k)  # builds the BM25 index
    if strategy == "hybrid_rerank":
        # Imported lazily: the cross-encoder is a ~1.1 GB model that the
        # other strategies never need.
        from cortex.rag.reranker import Reranker, RerankingRetriever

        retriever = RerankingRetriever(retriever, Reranker(), top_k=top_k)
    return retriever
