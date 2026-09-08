from src.generation import (
    ALT_NOT_STATED,
    NOT_STATED,
    build_messages,
    build_prompt,
    looks_like_yes_no,
)


def test_looks_like_yes_no_detects_auxiliary_starters():
    assert looks_like_yes_no("Did they use BERT?")
    assert looks_like_yes_no("do they report results on English data?")
    assert looks_like_yes_no("Is the dataset publicly available?")
    assert looks_like_yes_no("Are the annotations manually created?")
    assert looks_like_yes_no("Was the model pretrained?")
    assert looks_like_yes_no("Can the method generalize to other languages?")
    assert looks_like_yes_no("Has this been evaluated before?")
    assert looks_like_yes_no("Will the code be released?")


def test_looks_like_yes_no_rejects_wh_questions():
    assert not looks_like_yes_no("What datasets did they use?")
    assert not looks_like_yes_no("How many parameters does the model have?")
    assert not looks_like_yes_no("Which baselines do they compare with?")


def test_build_messages_default_marker_is_unanswerable():
    messages = build_messages("What is X?", ["para one"])
    system, user = messages[0]["content"], messages[1]["content"]
    assert NOT_STATED in system
    assert NOT_STATED in user


def test_build_messages_alt_marker_ablation():
    messages = build_messages("What is X?", ["para one"], abstention_marker=ALT_NOT_STATED)
    system, user = messages[0]["content"], messages[1]["content"]
    assert ALT_NOT_STATED in system
    assert ALT_NOT_STATED in user
    assert NOT_STATED not in system


def test_build_messages_yes_no_instructs_yes_no_over_abstention():
    messages = build_messages("Did they use BERT?", ["para one"], is_yes_no=True)
    system = messages[0]["content"]
    assert "Yes" in system and "No" in system
    assert NOT_STATED in system


def test_build_prompt_non_yes_no_has_no_yes_no_instruction():
    prompt = build_prompt("What is X?", ["para one"])
    assert "yes/no question" not in prompt.lower()


def test_build_prompt_yes_no_has_yes_no_instruction():
    prompt = build_prompt("Did they use BERT?", ["para one"], is_yes_no=True)
    assert "yes/no question" in prompt.lower()
