"""Generate answers over a QASPER split with Qwen2.5-1.5B-Instruct.

Designed to run on a GPU (Kaggle T4/P100). For each question: retrieve
top-k paragraphs with the chosen retriever, build the chat prompt
(looks_like_yes_no decides the yes/no instruction, never the ground-truth
answer_type), generate greedily, and record retrieval/generation/confidence
fields to results/generation_{split}_{retriever}_{marker}.csv.

Usage:
    uv run python scripts/run_generation.py \\
        --split validation --retriever hybrid --abstention-marker unanswerable \\
        --top-k 5 --batch-size 8 --limit 20
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_questions
from src.dense_retrieval import DenseRetriever
from src.generation import (
    ALT_NOT_STATED,
    DEFAULT_MAX_NEW_TOKENS,
    NOT_STATED,
    apply_chat_template,
    looks_like_yes_no,
)
from src.hybrid_retrieval import HybridRetriever
from src.retrieval import BM25Retriever

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MARKER_CHOICES = {"unanswerable": NOT_STATED, "notstated": ALT_NOT_STATED}
RETRIEVER_CHOICES = ("bm25", "dense", "hybrid")

PROGRESS_EVERY = 50
CHECKPOINT_EVERY = 100

FIELDNAMES = [
    "question_id",
    "paper_id",
    "answer_type",
    "unanswerable",
    "retrieved_idx",
    "evidence_idx",
    "retrieval_hit",
    "generated_answer",
    "sum_logprob",
    "mean_logprob",
    "n_generated_tokens",
    "hit_token_cap",
    "latency_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--retriever", choices=RETRIEVER_CHOICES, default="hybrid")
    parser.add_argument("--abstention-marker", choices=list(MARKER_CHOICES), default="unanswerable")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N questions (testing).")
    return parser.parse_args()


def build_retriever(paragraphs: list[str], retriever_name: str):
    """Build only what's needed for `retriever_name` -- dense embedding is the
    expensive step, so bm25-only runs should never trigger it."""
    bm25 = BM25Retriever(paragraphs) if retriever_name in ("bm25", "hybrid") else None
    dense = DenseRetriever(paragraphs) if retriever_name in ("dense", "hybrid") else None
    if retriever_name == "bm25":
        return bm25
    if retriever_name == "dense":
        return dense
    return HybridRetriever(bm25, dense)


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer, device


def _eos_ids(model, tokenizer) -> set[int]:
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        return {eos}
    return set(eos)


@torch.no_grad()
def generate_batch(model, tokenizer, device, prompts: list[str], max_new_tokens: int) -> list[dict]:
    """Greedy-generate a batch of prompts; return per-item text, token count,
    summed/mean logprob of the generated tokens, and whether it hit the cap."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    gen_tokens = output.sequences[:, input_len:]
    eos_ids = _eos_ids(model, tokenizer)

    results = []
    for i in range(gen_tokens.shape[0]):
        token_ids = gen_tokens[i].tolist()
        stop = next((j for j, t in enumerate(token_ids) if t in eos_ids), None)
        hit_cap = stop is None
        actual_tokens = token_ids if hit_cap else token_ids[:stop]

        logprobs = []
        for step, token_id in enumerate(actual_tokens):
            step_logits = output.scores[step][i]
            log_probs_step = torch.log_softmax(step_logits.float(), dim=-1)
            logprobs.append(log_probs_step[token_id].item())

        n_tokens = len(actual_tokens)
        sum_logprob = sum(logprobs)
        mean_logprob = sum_logprob / n_tokens if n_tokens else 0.0
        text = tokenizer.decode(actual_tokens, skip_special_tokens=True).strip()

        results.append(
            {
                "generated_answer": text,
                "n_generated_tokens": n_tokens,
                "sum_logprob": sum_logprob,
                "mean_logprob": mean_logprob,
                "hit_token_cap": hit_cap,
            }
        )
    return results


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    abstention_marker = MARKER_CHOICES[args.abstention_marker]
    output_path = RESULTS_DIR / f"generation_{args.split}_{args.retriever}_{args.abstention_marker}.csv"

    questions = load_questions(args.split)
    if args.limit:
        questions = questions[: args.limit]

    model, tokenizer, device = load_model()

    retriever_cache: dict[str, object] = {}
    rows: list[dict] = []
    pending_batch: list[dict] = []  # metadata + prompt for the in-flight batch

    def flush_batch():
        if not pending_batch:
            return
        prompts = [item["prompt"] for item in pending_batch]
        start = time.perf_counter()
        gen_results = generate_batch(model, tokenizer, device, prompts, DEFAULT_MAX_NEW_TOKENS)
        elapsed = time.perf_counter() - start
        per_item_latency = elapsed / len(pending_batch)
        for item, gen in zip(pending_batch, gen_results):
            row = dict(item["row"])
            row.update(gen)
            row["latency_seconds"] = per_item_latency
            rows.append(row)
        pending_batch.clear()

    n_processed = 0
    for q in questions:
        retriever = retriever_cache.get(q.paper_id)
        if retriever is None:
            retriever = build_retriever(q.paragraphs, args.retriever)
            retriever_cache[q.paper_id] = retriever

        retrieved_idx = retriever.retrieve(q.question, args.top_k)
        retrieved_paragraphs = [q.paragraphs[i] for i in retrieved_idx]
        is_yes_no = looks_like_yes_no(q.question)

        prompt = apply_chat_template(
            tokenizer, q.question, retrieved_paragraphs, is_yes_no=is_yes_no, abstention_marker=abstention_marker
        )

        retrieval_hit = bool(set(retrieved_idx) & set(q.evidence_idx))

        pending_batch.append(
            {
                "prompt": prompt,
                "row": {
                    "question_id": q.question_id,
                    "paper_id": q.paper_id,
                    "answer_type": q.answer_type,
                    "unanswerable": q.unanswerable,
                    "retrieved_idx": ";".join(str(i) for i in retrieved_idx),
                    "evidence_idx": ";".join(str(i) for i in q.evidence_idx),
                    "retrieval_hit": retrieval_hit,
                },
            }
        )

        if len(pending_batch) >= args.batch_size:
            flush_batch()

        n_processed += 1
        if n_processed % PROGRESS_EVERY == 0:
            print(f"processed {n_processed}/{len(questions)}")
        if n_processed % CHECKPOINT_EVERY == 0:
            flush_batch()
            _write_csv(output_path, rows)
            print(f"checkpoint saved: {len(rows)} rows -> {output_path}")

    flush_batch()
    _write_csv(output_path, rows)

    n = len(rows)
    mean_latency = sum(r["latency_seconds"] for r in rows) / n if n else 0.0
    truncation_rate = sum(1 for r in rows if r["hit_token_cap"]) / n if n else 0.0

    print(f"done: {n} rows -> {output_path}")
    print(f"mean latency: {mean_latency:.4f}s")
    print(f"truncation rate: {truncation_rate:.4f}")


if __name__ == "__main__":
    main()
