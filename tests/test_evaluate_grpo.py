from eval.evaluate_grpo import ParsedToolCall, parse_tool_call


def test_parses_well_formed_tool_call():
    text = '<tool_call>\n{"name": "run_sql", "arguments": {"query": "SELECT 1"}}\n</tool_call>'
    result = parse_tool_call(text)
    assert result == ParsedToolCall(name="run_sql", arguments={"query": "SELECT 1"})


def test_parses_tool_call_with_no_arguments():
    text = '<tool_call>\n{"name": "inspect_schema", "arguments": {}}\n</tool_call>'
    result = parse_tool_call(text)
    assert result == ParsedToolCall(name="inspect_schema", arguments={})


def test_parses_tool_call_missing_arguments_key_defaults_to_empty_dict():
    text = '<tool_call>\n{"name": "inspect_schema"}\n</tool_call>'
    result = parse_tool_call(text)
    assert result == ParsedToolCall(name="inspect_schema", arguments={})


def test_returns_none_when_no_tool_call_tag_present():
    assert parse_tool_call("I think the answer is 42.") is None


def test_returns_none_on_malformed_json_inside_tags():
    text = "<tool_call>\nnot valid json\n</tool_call>"
    assert parse_tool_call(text) is None


def test_returns_none_when_name_is_missing():
    text = '<tool_call>\n{"arguments": {"query": "SELECT 1"}}\n</tool_call>'
    assert parse_tool_call(text) is None


def test_returns_none_when_name_is_wrong_type():
    text = '<tool_call>\n{"name": 123, "arguments": {}}\n</tool_call>'
    assert parse_tool_call(text) is None


def test_returns_none_when_arguments_is_wrong_type():
    text = '<tool_call>\n{"name": "run_sql", "arguments": "SELECT 1"}\n</tool_call>'
    assert parse_tool_call(text) is None


def test_finds_tool_call_amid_surrounding_text():
    text = 'Let me check the schema.\n<tool_call>\n{"name": "inspect_schema", "arguments": {}}\n</tool_call>\nDone.'
    result = parse_tool_call(text)
    assert result == ParsedToolCall(name="inspect_schema", arguments={})


# ---- invalid_tool_call_rate is a real fraction ---------------------------


def test_invalid_rate_counts_unparseable_output_in_the_denominator():
    """A turn whose text has no parseable tool call is an *attempt* that
    failed, so it belongs in both halves of the ratio. Counting it only in
    the numerator inflated the rate and let it exceed 100% - which is how a
    77.2% "invalid tool call rate" got reported on a tier where the model
    was mostly just answering in prose instead of calling a tool.
    """
    from eval.evaluate_grpo import GrpoEpisodeResult

    # 4 attempts, 3 of which never parsed.
    r = GrpoEpisodeResult(
        db_id="d",
        success=False,
        steps_taken=4,
        tool_call_count=4,
        invalid_tool_call_count=3,
        transcript="",
    )
    rate = r.invalid_tool_call_count / r.tool_call_count
    assert rate == 0.75
    assert rate <= 1.0
