"""Prompt construction for answer generation from retrieved context."""

from __future__ import annotations

NOT_STATED = "UNANSWERABLE"

# "Not stated in the paper" used to bleed into yes/no answers (the model
# produced "No stated in the paper"). UNANSWERABLE can't collide with a
# yes/no answer, so it replaced that phrasing as the default. Both are kept
# as valid markers so the phrasing choice can be run as an ablation.
ALT_NOT_STATED = "Not stated in the paper"
ABSTENTION_MARKERS = (NOT_STATED, ALT_NOT_STATED)

# Smoke test truncated mid-sentence at 64 new tokens; raised to 128. Log the
# truncation rate (generations that hit this cap) once the GPU run exists.
DEFAULT_MAX_NEW_TOKENS = 128


def _numbered_context(paragraphs: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(paragraphs))


def _system_instructions(is_yes_no: bool, abstention_marker: str) -> str:
    base = "Answer the question using only the context below. Do not use any outside knowledge. "
    if is_yes_no:
        return (
            base + "This is a yes/no question: reply exactly \"Yes\" or \"No\" when the "
            "context supports an answer either way. Use the abstention marker only "
            f'if the context is genuinely silent on this: reply exactly "{abstention_marker}".'
        )
    return base + f'If the context does not contain the answer, reply exactly "{abstention_marker}".'


def _reminder(is_yes_no: bool, abstention_marker: str) -> str:
    if is_yes_no:
        return (
            'Remember: reply exactly "Yes" or "No" if the context above supports it; '
            f'reply exactly "{abstention_marker}" only if the context is silent on this.'
        )
    return f'Remember: if the context above does not contain the answer, reply exactly "{abstention_marker}".'


def build_messages(
    question: str,
    paragraphs: list[str],
    is_yes_no: bool = False,
    abstention_marker: str = NOT_STATED,
) -> list[dict[str, str]]:
    """Build a chat-format messages list: system instructions + user context/question.

    Repeats the abstention instruction immediately before the question, since
    a single instruction given many tokens earlier is easy for a small model
    to lose track of. `abstention_marker` selects which phrasing the model is
    told to reply with (default "UNANSWERABLE", ablation "Not stated in the
    paper"). `is_yes_no` tells the model to prefer a direct "Yes"/"No" answer
    over the abstention marker when the context supports one.
    """
    user_content = (
        f"Context:\n{_numbered_context(paragraphs)}\n\n"
        f"{_reminder(is_yes_no, abstention_marker)}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": _system_instructions(is_yes_no, abstention_marker)},
        {"role": "user", "content": user_content},
    ]


def apply_chat_template(
    tokenizer,
    question: str,
    paragraphs: list[str],
    is_yes_no: bool = False,
    abstention_marker: str = NOT_STATED,
) -> str:
    """Render the chat messages through the tokenizer's chat template."""
    messages = build_messages(question, paragraphs, is_yes_no, abstention_marker)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt(
    question: str,
    paragraphs: list[str],
    is_yes_no: bool = False,
    abstention_marker: str = NOT_STATED,
) -> str:
    """Raw-string prompt (ablation): no chat template, single instruction block."""
    return (
        f"{_system_instructions(is_yes_no, abstention_marker)}\n\n"
        f"Context:\n{_numbered_context(paragraphs)}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
