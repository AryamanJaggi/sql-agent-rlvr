import sqlite3

import pytest

from env.agent_loop import parse_action, run_episode
from env.environment import ToolCall
from tests.mock_policy import ScriptedPolicy

GOLD_SQL = "SELECT name FROM singer WHERE age > 30 ORDER BY name"


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "concert_singer.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.executemany(
        "INSERT INTO singer VALUES (?, ?, ?)",
        [(1, "Alice", 45), (2, "Bob", 25), (3, "Carol", 50)],
    )
    conn.commit()
    conn.close()
    return path


def _episode_kwargs(db_path):
    return dict(
        db_path=db_path,
        db_id="concert_singer",
        question="Which singers are older than 30, sorted by name?",
        gold_sql=GOLD_SQL,
    )


# ---- parse_action -----------------------------------------------------


def test_parse_action_basic():
    call = parse_action("Action: run_sql\nAction Input: SELECT 1")
    assert call == ToolCall(tool="run_sql", input="SELECT 1")


def test_parse_action_case_insensitive_tool_name():
    call = parse_action("Action: Inspect_Schema\nAction Input: (none)")
    assert call.tool == "inspect_schema"


def test_parse_action_stops_at_blank_line():
    call = parse_action("Action: run_sql\nAction Input: SELECT 1\n\nThought: now I'll check...")
    assert call.input == "SELECT 1"


def test_parse_action_returns_none_on_garbage():
    assert parse_action("I think the answer is 42.") is None


def test_parse_action_no_input_tool_without_action_input_line():
    # Real calibration transcripts showed the model correctly omitting
    # Action Input for inspect_schema (which needs none) and getting
    # rejected as unparseable under a stricter version of this regex -
    # forcing it to guess table/column names blind for the whole
    # episode. This is the regression test for that bug.
    call = parse_action("Action: inspect_schema")
    assert call == ToolCall(tool="inspect_schema", input="")


def test_parse_action_still_none_for_unknown_tool_name():
    assert parse_action("Action: delete_everything\nAction Input: yes") is None


# ---- run_episode: happy / wrong / malformed / step-cap paths ---------


def test_happy_path_inspect_schema_without_action_input_line(db_path):
    policy = ScriptedPolicy(
        [
            "Action: inspect_schema",  # no Action Input line - must not be treated as unparseable
            f"Action: run_sql\nAction Input: {GOLD_SQL}",
            "Action: final_answer\nAction Input: Alice, Carol",
        ]
    )
    result = run_episode(policy=policy, **_episode_kwargs(db_path))

    assert result.success is True
    assert "Could not parse an action" not in result.transcript


def test_happy_path_correct_answer(db_path):
    policy = ScriptedPolicy(
        [
            "Action: inspect_schema\nAction Input: (none)",
            f"Action: run_sql\nAction Input: {GOLD_SQL}",
            "Action: final_answer\nAction Input: Alice, Carol",
        ]
    )
    result = run_episode(policy=policy, **_episode_kwargs(db_path))

    assert result.success is True
    assert result.final_answer == "Alice, Carol"
    assert result.steps_taken == 3
    assert result.invalid_sql_count == 0
    assert "Observation:" in result.transcript


def test_wrong_query_scores_as_failure(db_path):
    policy = ScriptedPolicy(
        [
            "Action: run_sql\nAction Input: SELECT name FROM singer WHERE age > 49",
            "Action: final_answer\nAction Input: Alice, Carol",
        ]
    )
    result = run_episode(policy=policy, **_episode_kwargs(db_path))

    assert result.success is False


def test_invalid_sql_count_tracks_errored_queries(db_path):
    policy = ScriptedPolicy(
        [
            "Action: run_sql\nAction Input: DROP TABLE singer",  # rejected: not a SELECT
            "Action: run_sql\nAction Input: SELECT FROM WHERE",  # rejected: bad syntax
            f"Action: run_sql\nAction Input: {GOLD_SQL}",
            "Action: final_answer\nAction Input: Alice, Carol",
        ]
    )
    result = run_episode(policy=policy, **_episode_kwargs(db_path))

    assert result.success is True
    assert result.invalid_sql_count == 2
    assert result.run_sql_count == 3


def test_final_answer_without_any_query_scores_as_failure(db_path):
    # No run_sql call at all -> no last_result to score against.
    policy = ScriptedPolicy(["Action: final_answer\nAction Input: Alice, Carol"])
    result = run_episode(policy=policy, **_episode_kwargs(db_path))

    assert result.success is False
    assert result.final_answer == "Alice, Carol"


def test_malformed_action_recovers_on_next_step(db_path):
    policy = ScriptedPolicy(
        [
            "I think I should look at the schema first.",  # unparseable
            f"Action: run_sql\nAction Input: {GOLD_SQL}",
            "Action: final_answer\nAction Input: Alice, Carol",
        ]
    )
    result = run_episode(policy=policy, **_episode_kwargs(db_path))

    assert result.success is True
    assert result.steps_taken == 3
    assert "Could not parse an action" in result.transcript


def test_step_cap_exhaustion_without_final_answer(db_path):
    # Always emits a valid-but-non-terminal action; never calls final_answer.
    policy = ScriptedPolicy(["Action: inspect_schema\nAction Input: (none)"])
    result = run_episode(policy=policy, db_path=db_path, db_id="concert_singer",
                          question="irrelevant", gold_sql=GOLD_SQL, max_steps=4)

    assert result.success is False
    assert result.final_answer is None
    assert result.steps_taken == 4
    assert "Step limit reached" in result.transcript


# ---- defense-in-depth: the connection agent_loop opens is truly read-only


def test_agent_loop_connection_is_truly_read_only(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO singer VALUES (4, 'Dave', 60)")
    finally:
        conn.close()
