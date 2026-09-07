from src.data import load_questions


def test_validation_split_shape():
    questions = load_questions("validation")

    assert len(questions) > 0

    unanswerable = [q for q in questions if q.unanswerable]
    assert len(unanswerable) > 0

    answerable = [q for q in questions if not q.unanswerable]
    matched = [q for q in answerable if len(q.evidence_idx) > 0]
    assert len(matched) / len(answerable) >= 0.90
