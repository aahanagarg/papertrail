"""Prompt construction for answer generation from retrieved context."""

from __future__ import annotations

NOT_STATED = "Not stated in the paper"

_INSTRUCTIONS = (
    "Answer the question using only the context below. "
    "Do not use any outside knowledge. "
    f'If the context does not contain the answer, reply exactly "{NOT_STATED}".'
)


def build_prompt(question: str, paragraphs: list[str]) -> str:
    """Format retrieved paragraphs as numbered context plus the question."""
    numbered_context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(paragraphs))
    return (
        f"{_INSTRUCTIONS}\n\n"
        f"Context:\n{numbered_context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
