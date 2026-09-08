"""Dense retrieval over a fixed list of paragraphs using sentence embeddings + FAISS."""

from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model() -> SentenceTransformer:
    model = _model_cache.get(_MODEL_NAME)
    if model is None:
        model = SentenceTransformer(_MODEL_NAME)
        _model_cache[_MODEL_NAME] = model
    return model


class DenseRetriever:
    def __init__(self, paragraphs: list[str]):
        self.paragraphs = paragraphs
        model = _get_model()
        embeddings = model.encode(paragraphs, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def retrieve(self, query: str, k: int) -> list[int]:
        """Return indices of the top-`k` paragraphs, ranked by cosine similarity."""
        model = _get_model()
        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")
        k = min(k, len(self.paragraphs))
        _, indices = self._index.search(query_embedding, k)
        return [int(i) for i in indices[0] if i != -1]
