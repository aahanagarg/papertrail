"""Load allenai/qasper into flat per-question records with matched evidence.

QASPER's dataset-script loader is no longer supported by recent versions of
`datasets`, so we pull the HuggingFace-hosted parquet conversion instead
(`revision="refs/convert/parquet"`), which exposes the same fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from datasets import load_dataset

_FLOAT_SELECTED = "FLOAT SELECTED"


@dataclass
class Question:
    question_id: str
    paper_id: str
    question: str
    paragraphs: list[str]
    evidence_idx: list[int]
    answers: list[str]
    answer_type: str
    unanswerable: bool
    unanswerable_any: bool
    scorable_retrieval: bool


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _flatten_paragraphs(full_text: dict) -> list[str]:
    paragraphs: list[str] = []
    for section_paragraphs in full_text["paragraphs"]:
        paragraphs.extend(section_paragraphs)
    return paragraphs


def _match_evidence(evidence: list[str], paragraphs: list[str], normalized_lookup: dict[str, int]) -> list[int]:
    matched: list[int] = []
    for ev in evidence:
        if _FLOAT_SELECTED in ev:
            continue
        if ev in paragraphs:
            matched.append(paragraphs.index(ev))
            continue
        idx = normalized_lookup.get(_normalize_whitespace(ev))
        if idx is not None:
            matched.append(idx)
    return matched


def _answer_type_and_text(answer: dict) -> tuple[str, str]:
    if answer["unanswerable"]:
        return "unanswerable", ""
    if answer["yes_no"] is not None:
        return "yes_no", "Yes" if answer["yes_no"] else "No"
    if answer["free_form_answer"]:
        return "abstractive", answer["free_form_answer"]
    if answer["extractive_spans"]:
        return "extractive", "; ".join(answer["extractive_spans"])
    return "unanswerable", ""


def load_questions(split: str) -> list[Question]:
    """Load one flattened, evidence-matched record per question in `split`."""
    dataset = load_dataset("allenai/qasper", split=split, revision="refs/convert/parquet")

    records: list[Question] = []
    for paper in dataset:
        paper_id = paper["id"]
        paragraphs = _flatten_paragraphs(paper["full_text"])
        normalized_lookup = {
            _normalize_whitespace(p): i for i, p in enumerate(paragraphs)
        }

        qas = paper["qas"]
        for i in range(len(qas["question_id"])):
            question_id = qas["question_id"][i]
            question = qas["question"][i]
            annotations = qas["answers"][i]["answer"]

            evidence_idx: set[int] = set()
            answers: list[str] = []
            answer_types: list[str] = []
            unanswerable_votes: list[bool] = []

            for annotation in annotations:
                evidence_idx.update(
                    _match_evidence(annotation["evidence"], paragraphs, normalized_lookup)
                )
                answer_type, answer_text = _answer_type_and_text(annotation)
                answer_types.append(answer_type)
                if answer_text:
                    answers.append(answer_text)
                unanswerable_votes.append(bool(annotation["unanswerable"]))

            unanswerable = all(unanswerable_votes) if unanswerable_votes else False
            unanswerable_any = any(unanswerable_votes)
            answer_type = "unanswerable" if unanswerable else next(
                (t for t in answer_types if t != "unanswerable"), "unanswerable"
            )
            sorted_evidence_idx = sorted(evidence_idx)
            scorable_retrieval = not unanswerable and len(sorted_evidence_idx) > 0

            records.append(
                Question(
                    question_id=question_id,
                    paper_id=paper_id,
                    question=question,
                    paragraphs=paragraphs,
                    evidence_idx=sorted_evidence_idx,
                    answers=answers,
                    answer_type=answer_type,
                    unanswerable=unanswerable,
                    unanswerable_any=unanswerable_any,
                    scorable_retrieval=scorable_retrieval,
                )
            )

    return records
