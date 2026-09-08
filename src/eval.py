"""Scoring functions for answer quality and retrieval recall.

`answer_f1` follows the SQuAD-style token-overlap F1 convention (as used by
QASPER's official evaluation script), with two special cases layered on top:
yes/no questions are scored by exact match, and unanswerable questions are
scored by whether the prediction is an abstention.
"""

from __future__ import annotations

import re
import string

from src.generation import NOT_STATED

_ARTICLES = {"a", "an", "the"}

# Legacy phrasings from before the abstention marker was standardized to
# NOT_STATED ("UNANSWERABLE"); kept as a fallback for predictions that don't
# follow the current prompt instructions.
_LEGACY_ABSTENTION_PATTERNS = [
    "not stated",
    "not mentioned",
    "not specified",
    "not provided",
    "not given",
    "not answered",
    "not addressed",
    "not discussed",
    "cannot be answered",
    "can not be answered",
    "cannot answer",
    "can not answer",
    "does not answer",
    "does not state",
    "does not mention",
    "does not specify",
    "does not provide",
    "does not discuss",
    "does not address",
    "does not say",
    "doesn't answer",
    "doesn't state",
    "doesn't mention",
    "doesn't specify",
    "doesn't provide",
    "the paper does not",
    "the paper doesn't",
    "no answer",
    "no information",
    "i don't know",
    "i do not know",
    "unknown",
]

_YES_NO_VALUES = {"yes", "no"}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(tokens)


def is_abstention(prediction: str) -> bool:
    """Detect whether a prediction is an abstention ("I don't know" etc.)."""
    normalized_pred = normalize(prediction)
    if not normalized_pred:
        return True
    if normalize(NOT_STATED) in normalized_pred:
        return True
    for pattern in _LEGACY_ABSTENTION_PATTERNS:
        if normalize(pattern) in normalized_pred:
            return True
    return False


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize(prediction).split()
    ref_tokens = normalize(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    num_common = 0
    for tok in ref_tokens:
        if pred_counts.get(tok, 0) > 0:
            num_common += 1
            pred_counts[tok] -= 1

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_f1(prediction: str, references: list[str], unanswerable: bool = False) -> float:
    """SQuAD-style token F1 of `prediction` against `references`, max over refs.

    Special cases:
    - unanswerable questions: 1.0 if `prediction` is an abstention, else 0.0.
    - yes/no references: scored by exact match after normalization, not F1.
    """
    if unanswerable:
        return 1.0 if is_abstention(prediction) else 0.0

    if not references:
        return 0.0

    normalized_refs = {normalize(r) for r in references}
    if normalized_refs <= _YES_NO_VALUES:
        return 1.0 if normalize(prediction) in normalized_refs else 0.0

    return max(_token_f1(prediction, reference) for reference in references)


def recall_at_k(retrieved_idx: list[int], evidence_idx: list[int], k: int) -> float:
    """1.0 if any of `evidence_idx` appears in the top `k` of `retrieved_idx`."""
    if not evidence_idx:
        return 0.0
    top_k = set(retrieved_idx[:k])
    return 1.0 if top_k & set(evidence_idx) else 0.0
