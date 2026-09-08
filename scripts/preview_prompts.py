"""Preview generation prompts on a small sample, with no model call.

Picks 2 extractive, 2 yes_no, and 2 unanswerable validation questions,
retrieves top-5 paragraphs with the BM25+dense hybrid retriever, builds the
generation prompt, and prints it in full (rendered through the
Qwen2.5-1.5B-Instruct chat template) along with its token count -- so the
prompts can be read and their context-window fit checked before spending
any GPU time.

Pass an answer_type (extractive, yes_no, unanswerable) as a CLI argument to
print only questions of that type, e.g.:

    uv run python scripts/preview_prompts.py unanswerable
"""

from __future__ import annotations

import sys

from transformers import AutoTokenizer

from src.data import load_questions
from src.dense_retrieval import DenseRetriever
from src.generation import apply_chat_template
from src.hybrid_retrieval import HybridRetriever
from src.retrieval import BM25Retriever

TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 5
WANTED = {"extractive": 2, "yes_no": 2, "unanswerable": 2}


def _select_questions(questions: list) -> list:
    selected = []
    counts = {k: 0 for k in WANTED}
    for q in questions:
        if q.answer_type in WANTED and counts[q.answer_type] < WANTED[q.answer_type]:
            selected.append(q)
            counts[q.answer_type] += 1
        if all(counts[k] >= WANTED[k] for k in WANTED):
            break
    return selected


def main() -> None:
    answer_type_filter = sys.argv[1] if len(sys.argv) > 1 else None

    questions = load_questions("validation")
    selected = _select_questions(questions)
    if answer_type_filter:
        selected = [q for q in selected if q.answer_type == answer_type_filter]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    for q in selected:
        bm25 = BM25Retriever(q.paragraphs)
        dense = DenseRetriever(q.paragraphs)
        hybrid = HybridRetriever(bm25, dense)
        retrieved_idx = hybrid.retrieve(q.question, TOP_K)
        retrieved_paragraphs = [q.paragraphs[i] for i in retrieved_idx]

        prompt = apply_chat_template(tokenizer, q.question, retrieved_paragraphs)
        n_tokens = len(tokenizer.encode(prompt))

        print("=" * 80)
        print(f"question_id: {q.question_id}  paper_id: {q.paper_id}  answer_type: {q.answer_type}")
        print(f"token count: {n_tokens}")
        print("-" * 80)
        print(prompt)
        print()


if __name__ == "__main__":
    main()
