from env.policies import PROMPTED_BASELINE_SYSTEM_PROMPT, _build_messages


def test_build_messages_includes_system_prompt():
    messages = _build_messages("Question: how many singers?")

    assert messages[0] == {"role": "system", "content": PROMPTED_BASELINE_SYSTEM_PROMPT}


def test_build_messages_passes_transcript_through_as_user_turn():
    transcript = "Question: how many singers?\n\nAction: run_sql\nAction Input: SELECT 1"
    messages = _build_messages(transcript)

    assert messages[1] == {"role": "user", "content": transcript}
    assert len(messages) == 2


def test_system_prompt_documents_all_three_tools():
    for tool in ("inspect_schema", "run_sql", "final_answer"):
        assert tool in PROMPTED_BASELINE_SYSTEM_PROMPT
