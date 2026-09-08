"""BM25 sparse retrieval over a fixed list of paragraphs."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    def __init__(self, paragraphs: list[str]):
        self.paragraphs = paragraphs
        tokenized = [_tokenize(p) for p in paragraphs]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int) -> list[int]:
        """Return indices of the top-`k` paragraphs, ranked by BM25 score."""
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:k]
