from src.generation import ALT_NOT_STATED, NOT_STATED, build_messages, build_prompt


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
