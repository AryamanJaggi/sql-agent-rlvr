import sqlite3

import pytest

from data.spider_loader import SpiderExample
from eval.evaluate import evaluate
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


def _example(db_path, difficulty="medium", gold_sql=GOLD_SQL):
    return SpiderExample(
        db_id="concert_singer",
        question="Which singers are older than 30, sorted by name?",
        gold_sql=gold_sql,
        difficulty=difficulty,
        db_path=db_path,
    )


CORRECT_SCRIPT = [
    f"Action: run_sql\nAction Input: {GOLD_SQL}",
    "Action: final_answer\nAction Input: Alice, Carol",
]
WRONG_SCRIPT = [
    "Action: run_sql\nAction Input: SELECT name FROM singer WHERE age > 49",
    "Action: final_answer\nAction Input: Carol",
]
ONE_INVALID_THEN_CORRECT_SCRIPT = [
    "Action: run_sql\nAction Input: DROP TABLE singer",
    f"Action: run_sql\nAction Input: {GOLD_SQL}",
    "Action: final_answer\nAction Input: Alice, Carol",
]


def test_success_rate_and_avg_steps(db_path):
    examples = [_example(db_path), _example(db_path)]
    policy = ScriptedPolicy(CORRECT_SCRIPT)

    report = evaluate(policy, examples, max_steps=10)

    assert report.n_examples == 2
    assert report.success_rate == 1.0
    assert report.avg_steps == 2.0
    assert report.avg_steps_on_success == 2.0


def test_mixed_success_and_failure(db_path):
    examples = [_example(db_path), _example(db_path)]
    correct_result = evaluate(ScriptedPolicy(CORRECT_SCRIPT), [examples[0]])
    wrong_result = evaluate(ScriptedPolicy(WRONG_SCRIPT), [examples[1]])

    assert correct_result.success_rate == 1.0
    assert wrong_result.success_rate == 0.0


def test_invalid_sql_rate(db_path):
    examples = [_example(db_path)]
    policy = ScriptedPolicy(ONE_INVALID_THEN_CORRECT_SCRIPT)

    report = evaluate(policy, examples)

    assert report.success_rate == 1.0
    assert report.invalid_sql_rate == pytest.approx(0.5)  # 1 of 2 run_sql calls errored


def test_invalid_sql_rate_is_zero_when_no_queries_run(db_path):
    examples = [_example(db_path)]
    policy = ScriptedPolicy(["Action: final_answer\nAction Input: nothing queried"])

    report = evaluate(policy, examples)

    assert report.invalid_sql_rate == 0.0


def test_by_difficulty_breakdown(db_path):
    examples = [
        _example(db_path, difficulty="easy"),
        _example(db_path, difficulty="hard"),
    ]
    report = evaluate(ScriptedPolicy(CORRECT_SCRIPT), examples)

    assert report.by_difficulty == {"easy": 1.0, "hard": 1.0}


def test_by_difficulty_only_includes_tiers_present(db_path):
    examples = [_example(db_path, difficulty="easy")]
    report = evaluate(ScriptedPolicy(CORRECT_SCRIPT), examples)

    assert set(report.by_difficulty.keys()) == {"easy"}


def test_broken_example_does_not_abort_the_whole_run(db_path, tmp_path):
    missing_db = tmp_path / "does_not_exist.sqlite"
    broken_example = _example(missing_db)
    good_example = _example(db_path)

    report = evaluate(ScriptedPolicy(CORRECT_SCRIPT), [broken_example, good_example])

    assert report.n_examples == 2
    assert report.success_rate == 0.5
    assert not report.results[0].success
    assert report.results[1].success
