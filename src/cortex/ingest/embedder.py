"""Text embedder wrapping sentence-transformers.

Converts a list of strings into a list of normalised float vectors ready for
cosine-similarity search.

Embeddings are normalised to unit length, so cosine similarity reduces to a
dot product — the same ranking, computed faster.
"""

from __future__ import annotations

import os


# Apply HF_HOME from .env before the huggingface libraries read it.
# Must happen before any sentence_transformers / transformers import.
def _apply_hf_home() -> None:
    from cortex.config import get_settings
    hf_home = os.environ.get("HF_HOME") or getattr(get_settings(), "hf_home", None)
    if hf_home:
        os.environ["HF_HOME"] = hf_home

_apply_hf_home()


class Embedder:
    """Thin wrapper around a sentence-transformers model.

    Args:
        model_name: HuggingFace model identifier.  BGE-M3 is multilingual
                    (Chinese + English) and produces 1024-dim vectors.
    """

    DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # Imported lazily: sentence-transformers pulls in torch, which code
        # paths that never embed (the LLM check, the tests) should not pay for.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Output vector length (e.g. 1024 for BGE-M3)."""
        get_dim = getattr(self._model, "get_embedding_dimension", None) or \
                  getattr(self._model, "get_sentence_embedding_dimension", None)
        dim = get_dim() if get_dim else None
        if dim is None:
            raise RuntimeError(f"Could not determine embedding dimension for {self._model_name}")
        return int(dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Returns:
            List of unit-normalised float vectors, one per input text.
        """
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
