import sqlite3

import pytest

from env.grpo_env import MAX_STEPS, SqlAgentGrpoEnv, grpo_reward_func

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


@pytest.fixture
def env(db_path):
    e = SqlAgentGrpoEnv()
    e.reset(db_path=str(db_path), gold_sql=GOLD_SQL, db_id="concert_singer")
    return e


def test_inspect_schema_lists_table(env):
    schema = env.inspect_schema()
    assert "singer" in schema
    assert env.steps_taken == 1


def test_run_sql_success_updates_last_result(env):
    result = env.run_sql(GOLD_SQL)
    assert "Alice" in result and "Carol" in result
    assert env.last_result == [("Alice",), ("Carol",)]


def test_run_sql_failure_returns_error_string_not_crash(env):
    result = env.run_sql("DROP TABLE singer")
    assert result.startswith("Error:")
    assert env.last_result is None
    # A subsequent valid query still works - the failure didn't poison the connection.
    result2 = env.run_sql(GOLD_SQL)
    assert "Alice" in result2


def test_final_answer_correct_scores_reward_one(env):
    env.run_sql(GOLD_SQL)
    env.final_answer("Alice, Carol")
    assert env.reward == 1.0
    assert env.done is True


def test_final_answer_wrong_query_scores_reward_zero(env):
    env.run_sql("SELECT name FROM singer WHERE age > 49")  # only Carol, not Alice
    env.final_answer("Carol")
    assert env.reward == 0.0


def test_final_answer_without_any_query_scores_reward_zero(env):
    env.final_answer("Alice, Carol")
    assert env.reward == 0.0


def test_step_budget_raises_after_max_steps(env):
    for _ in range(MAX_STEPS):
        env.inspect_schema()
    assert env.done is False

    with pytest.raises(ValueError, match="Step limit reached"):
        env.inspect_schema()
    assert env.done is True


def test_calling_a_tool_after_done_raises(env):
    env.final_answer("done")
    assert env.done is True
    with pytest.raises(ValueError, match="already finished"):
        env.inspect_schema()


def test_reset_reusable_for_a_new_episode(env, db_path):
    env.run_sql(GOLD_SQL)
    env.final_answer("Alice, Carol")
    assert env.done is True

    env.reset(db_path=str(db_path), gold_sql=GOLD_SQL, db_id="concert_singer")
    assert env.done is False
    assert env.steps_taken == 0
    assert env.last_result is None
    assert env.reward == 0.0


def test_grpo_reward_func_reads_reward_off_each_environment(env, db_path):
    env.run_sql(GOLD_SQL)
    env.final_answer("Alice, Carol")

    other = SqlAgentGrpoEnv()
    other.reset(db_path=str(db_path), gold_sql=GOLD_SQL, db_id="concert_singer")
    other.final_answer("nothing queried")

    assert grpo_reward_func([env, other]) == [1.0, 0.0]
