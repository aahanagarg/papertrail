"""Evaluate dense (MiniLM+FAISS) and hybrid (BM25+dense RRF) retrieval on the
same scorable QASPER validation questions as scripts/eval_retrieval.py, so
all three methods are directly comparable.

Writes per-question rows to results/retrieval_dense.csv and
results/retrieval_hybrid.csv, and a summary table to
results/retrieval_comparison.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.data import load_questions
from src.dense_retrieval import DenseRetriever
from src.eval import recall_at_k
from src.hybrid_retrieval import HybridRetriever
from src.retrieval import BM25Retriever

K_VALUES = (1, 3, 5, 10)
MAX_K = max(K_VALUES)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BM25_PATH = RESULTS_DIR / "retrieval_bm25.csv"
DENSE_PATH = RESULTS_DIR / "retrieval_dense.csv"
HYBRID_PATH = RESULTS_DIR / "retrieval_hybrid.csv"
COMPARISON_PATH = RESULTS_DIR / "retrieval_comparison.csv"


def _write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = ["question_id", "paper_id", "n_paragraphs"] + [f"hit@{k}" for k in K_VALUES]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    questions = load_questions("validation")
    scorable = [q for q in questions if q.scorable_retrieval]

    bm25_cache: dict[str, BM25Retriever] = {}
    dense_cache: dict[str, DenseRetriever] = {}

    dense_rows = []
    hybrid_rows = []
    bm25_wins_for_dense = []  # dense hit@10, bm25 miss@10

    for q in scorable:
        bm25 = bm25_cache.get(q.paper_id)
        if bm25 is None:
            bm25 = BM25Retriever(q.paragraphs)
            bm25_cache[q.paper_id] = bm25

        dense = dense_cache.get(q.paper_id)
        if dense is None:
            dense = DenseRetriever(q.paragraphs)
            dense_cache[q.paper_id] = dense

        hybrid = HybridRetriever(bm25, dense)

        bm25_retrieved = bm25.retrieve(q.question, MAX_K)
        dense_retrieved = dense.retrieve(q.question, MAX_K)
        hybrid_retrieved = hybrid.retrieve(q.question, MAX_K)

        dense_row = {
            "question_id": q.question_id,
            "paper_id": q.paper_id,
            "n_paragraphs": len(q.paragraphs),
        }
        hybrid_row = dict(dense_row)
        for k in K_VALUES:
            dense_row[f"hit@{k}"] = recall_at_k(dense_retrieved, q.evidence_idx, k)
            hybrid_row[f"hit@{k}"] = recall_at_k(hybrid_retrieved, q.evidence_idx, k)
        dense_rows.append(dense_row)
        hybrid_rows.append(hybrid_row)

        bm25_hit10 = recall_at_k(bm25_retrieved, q.evidence_idx, 10)
        dense_hit10 = recall_at_k(dense_retrieved, q.evidence_idx, 10)
        if dense_hit10 == 1.0 and bm25_hit10 == 0.0:
            bm25_wins_for_dense.append(
                {
                    "question_id": q.question_id,
                    "paper_id": q.paper_id,
                    "question": q.question,
                    "dense_top1": q.paragraphs[dense_retrieved[0]],
                    "bm25_top1": q.paragraphs[bm25_retrieved[0]],
                    "evidence": q.paragraphs[q.evidence_idx[0]],
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_rows(DENSE_PATH, dense_rows)
    _write_rows(HYBRID_PATH, hybrid_rows)

    with BM25_PATH.open() as f:
        bm25_rows = list(csv.DictReader(f))
    bm25_by_id = {r["question_id"]: r for r in bm25_rows}
    dense_by_id = {r["question_id"]: r for r in dense_rows}
    hybrid_by_id = {r["question_id"]: r for r in hybrid_rows}

    common_ids = set(bm25_by_id) & set(dense_by_id) & set(hybrid_by_id)
    n = len(common_ids)

    summary_rows = []
    for method, by_id in (("bm25", bm25_by_id), ("dense", dense_by_id), ("hybrid", hybrid_by_id)):
        row = {"method": method, "n_scored": n}
        for k in K_VALUES:
            row[f"recall@{k}"] = sum(float(by_id[qid][f"hit@{k}"]) for qid in common_ids) / n
        summary_rows.append(row)

    with COMPARISON_PATH.open("w", newline="") as f:
        fieldnames = ["method", "n_scored"] + [f"recall@{k}" for k in K_VALUES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"scored questions: {n}")
    print(f"{'method':<8}" + "".join(f"{'recall@'+str(k):>12}" for k in K_VALUES))
    for row in summary_rows:
        print(f"{row['method']:<8}" + "".join(f"{row[f'recall@{k}']:>12.4f}" for k in K_VALUES))

    print()
    print(f"dense-hit/bm25-miss @10 cases: {len(bm25_wins_for_dense)}")
    for case in bm25_wins_for_dense[:5]:
        print(f"\n=== {case['question_id']} (paper {case['paper_id']}) ===")
        print(f"Question: {case['question'][:300]}")
        print(f"Dense top-1: {case['dense_top1'][:300]}")
        print(f"BM25 top-1: {case['bm25_top1'][:300]}")
        print(f"Ground-truth evidence: {case['evidence'][:300]}")


if __name__ == "__main__":
    main()
