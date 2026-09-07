from src.eval import answer_f1, is_abstention, normalize, recall_at_k


def test_normalize_strips_case_punctuation_articles_and_whitespace():
    assert normalize("The Cat, sat!!  on   a Mat.") == "cat sat on mat"


def test_answer_f1_identical_strings_scores_one():
    assert answer_f1("the cat sat on the mat", ["the cat sat on the mat"]) == 1.0


def test_answer_f1_disjoint_strings_scores_zero():
    assert answer_f1("completely unrelated text", ["totally different words"]) == 0.0


def test_answer_f1_partial_overlap():
    # prediction normalizes to: "cat sat on mat"      (4 tokens, articles dropped)
    # reference  normalizes to: "cat sat on mattress"  (4 tokens, articles dropped)
    # common tokens (multiset intersection): cat, sat, on -> num_common = 3
    # ("mat" != "mattress", so that pair doesn't count)
    # precision = num_common / len(pred_tokens) = 3/4 = 0.75
    # recall    = num_common / len(ref_tokens)  = 3/4 = 0.75
    # f1 = 2 * precision * recall / (precision + recall)
    #    = 2 * 0.75 * 0.75 / (0.75 + 0.75)
    #    = 1.125 / 1.5
    #    = 0.75
    prediction = "the cat sat on the mat"
    reference = "a cat sat on a mattress"
    assert answer_f1(prediction, [reference]) == 0.75


def test_answer_f1_takes_max_over_references():
    prediction = "the cat sat on the mat"
    references = ["totally different words", "the cat sat on the mat"]
    assert answer_f1(prediction, references) == 1.0


def test_answer_f1_yes_no_scored_by_exact_match_not_token_overlap():
    assert answer_f1("Yes", ["Yes"]) == 1.0
    assert answer_f1("No", ["Yes"]) == 0.0
    # token overlap would be nonzero-ish for near words, but yes/no must be exact
    assert answer_f1("Yes, it does", ["Yes"]) == 0.0


def test_answer_f1_unanswerable_scores_abstention_only():
    assert answer_f1("The paper does not state this.", [], unanswerable=True) == 1.0
    assert answer_f1("42", [], unanswerable=True) == 0.0


def test_is_abstention_detects_common_phrasings():
    assert is_abstention("Not stated in the paper.")
    assert is_abstention("This question is unanswerable.")
    assert is_abstention("The paper does not answer this.")
    assert is_abstention("There is no answer to this question.")
    assert not is_abstention("The accuracy was 92.3%.")


def test_recall_at_k_hit_and_miss():
    assert recall_at_k([5, 2, 9], [9], k=2) == 0.0
    assert recall_at_k([5, 2, 9], [9], k=3) == 1.0


def test_recall_at_k_empty_evidence_is_zero():
    assert recall_at_k([1, 2, 3], [], k=3) == 0.0


def test_recall_at_k_k_larger_than_retrieved_list():
    assert recall_at_k([1, 2], [2], k=100) == 1.0
    assert recall_at_k([1, 2], [3], k=100) == 0.0


def test_recall_at_k_empty_retrieved_list():
    assert recall_at_k([], [1], k=5) == 0.0
