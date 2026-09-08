"""Evaluate BM25 retrieval recall@k on the QASPER validation split.

For every question with `scorable_retrieval == True`, builds a fresh BM25
index over that question's own paper's paragraphs, retrieves the top-10,
and records hit@1/3/5/10. Writes per-question rows to
results/retrieval_bm25.csv and prints mean recall@k plus the scored count.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.data import load_questions
from src.eval import recall_at_k
from src.retrieval import BM25Retriever

K_VALUES = (1, 3, 5, 10)
MAX_K = max(K_VALUES)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "retrieval_bm25.csv"


def main() -> None:
    questions = load_questions("validation")
    scorable = [q for q in questions if q.scorable_retrieval]

    retriever_cache: dict[str, BM25Retriever] = {}
    rows = []

    for q in scorable:
        retriever = retriever_cache.get(q.paper_id)
        if retriever is None:
            retriever = BM25Retriever(q.paragraphs)
            retriever_cache[q.paper_id] = retriever

        retrieved = retriever.retrieve(q.question, MAX_K)

        row = {
            "question_id": q.question_id,
            "paper_id": q.paper_id,
            "n_paragraphs": len(q.paragraphs),
        }
        for k in K_VALUES:
            row[f"hit@{k}"] = recall_at_k(retrieved, q.evidence_idx, k)
        rows.append(row)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question_id", "paper_id", "n_paragraphs"] + [f"hit@{k}" for k in K_VALUES]
    with RESULTS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print(f"scored questions: {n}")
    for k in K_VALUES:
        mean_recall = sum(row[f"hit@{k}"] for row in rows) / n
        print(f"mean recall@{k}: {mean_recall:.4f}")


if __name__ == "__main__":
    main()
