"""Check how well looks_like_yes_no (text-only heuristic) agrees with
QASPER's answer_type == "yes_no" ground truth on the validation split.

The heuristic is what a deployed system must use (answer_type is a label
that doesn't exist at inference time), so this measures how much signal is
lost by not having the label.
"""

from __future__ import annotations

from src.data import load_questions
from src.generation import looks_like_yes_no


def main() -> None:
    questions = load_questions("validation")

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for q in questions:
        predicted_yes_no = looks_like_yes_no(q.question)
        actual_yes_no = q.answer_type == "yes_no"

        if predicted_yes_no and actual_yes_no:
            true_positive += 1
        elif predicted_yes_no and not actual_yes_no:
            false_positive += 1
        elif not predicted_yes_no and actual_yes_no:
            false_negative += 1
        else:
            true_negative += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0

    print(f"n questions: {len(questions)}")
    print(f"true_positive: {true_positive}  false_positive: {false_positive}")
    print(f"false_negative: {false_negative}  true_negative: {true_negative}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")


if __name__ == "__main__":
    main()
