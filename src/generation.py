"""Prompt construction for answer generation from retrieved context."""

from __future__ import annotations

NOT_STATED = "UNANSWERABLE"

# "Not stated in the paper" used to bleed into yes/no answers (the model
# produced "No stated in the paper"). UNANSWERABLE can't collide with a
# yes/no answer, so it replaced that phrasing here.
_SYSTEM_INSTRUCTIONS = (
    "Answer the question using only the context below. "
    "Do not use any outside knowledge. "
    f'If the context does not contain the answer, reply exactly "{NOT_STATED}".'
)

_REMINDER = f'Remember: if the context above does not contain the answer, reply exactly "{NOT_STATED}".'

# Smoke test truncated mid-sentence at 64 new tokens; raised to 128. Log the
# truncation rate (generations that hit this cap) once the GPU run exists.
DEFAULT_MAX_NEW_TOKENS = 128


def _numbered_context(paragraphs: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(paragraphs))


def build_messages(question: str, paragraphs: list[str]) -> list[dict[str, str]]:
    """Build a chat-format messages list: system instructions + user context/question.

    Repeats the abstention instruction immediately before the question, since
    a single instruction given many tokens earlier is easy for a small model
    to lose track of.
    """
    user_content = (
        f"Context:\n{_numbered_context(paragraphs)}\n\n"
        f"{_REMINDER}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_content},
    ]


def apply_chat_template(tokenizer, question: str, paragraphs: list[str]) -> str:
    """Render the chat messages through the tokenizer's chat template."""
    messages = build_messages(question, paragraphs)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt(question: str, paragraphs: list[str]) -> str:
    """Raw-string prompt (ablation): no chat template, single instruction block."""
    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{_numbered_context(paragraphs)}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
