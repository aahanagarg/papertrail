"""Hybrid retrieval: reciprocal rank fusion of BM25 and dense rankings."""

from __future__ import annotations

_RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[int]], k: int, rrf_k: int = _RRF_K) -> list[int]:
    """Fuse multiple ranked-index lists via RRF and return the top-`k` indices."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    ranked = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
    return ranked[:k]


class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, rrf_k: int = _RRF_K):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[int]:
        n = len(self.bm25_retriever.paragraphs)
        bm25_ranking = self.bm25_retriever.retrieve(query, n)
        dense_ranking = self.dense_retriever.retrieve(query, n)
        return reciprocal_rank_fusion([bm25_ranking, dense_ranking], k, self.rrf_k)
