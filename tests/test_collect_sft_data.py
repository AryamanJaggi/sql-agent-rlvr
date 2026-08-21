import json
import sqlite3

import pytest

from data.spider_loader import SpiderExample
from env.policies import PROMPTED_BASELINE_SYSTEM_PROMPT
from tests.mock_policy import ScriptedPolicy
from train.collect_sft_data import collect_from_episode, collect_sft_data

GOLD_SQL = "SELECT name FROM singer WHERE age > 30 ORDER BY name"

CORRECT_SCRIPT = [
    f"Action: run_sql\nAction Input: {GOLD_SQL}",
    "Action: final_answer\nAction Input: Alice, Carol",
]
WRONG_SCRIPT = [
    "Action: run_sql\nAction Input: SELECT name FROM singer WHERE age > 49",
    "Action: final_answer\nAction Input: Carol",
]


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


def _example(db_path, difficulty="hard"):
    return SpiderExample(
        db_id="concert_singer",
        question="Which singers are older than 30, sorted by name?",
        gold_sql=GOLD_SQL,
        difficulty=difficulty,
        db_path=db_path,
    )


# ---- collect_from_episode (pure function) -----------------------------


def test_collect_from_episode_empty_for_failed_episode(db_path):
    result = _run(ScriptedPolicy(WRONG_SCRIPT), db_path)
    assert result.success is False
    assert collect_from_episode(result, "sys prompt") == []


def test_collect_from_episode_one_record_per_turn(db_path):
    result = _run(ScriptedPolicy(CORRECT_SCRIPT), db_path)
    assert result.success is True

    records = collect_from_episode(result, "sys prompt")

    assert len(records) == len(result.turns) == 2
    for record, turn in zip(records, result.turns):
        assert record["messages"] == [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": turn.prompt},
            {"role": "assistant", "content": turn.completion},
        ]


def _run(policy, db_path):
    from env.agent_loop import run_episode

    return run_episode(
        policy=policy,
        db_path=db_path,
        db_id="concert_singer",
        question="Which singers are older than 30, sorted by name?",
        gold_sql=GOLD_SQL,
        max_steps=10,
    )


# ---- collect_sft_data (end to end, no network) -------------------------


def test_collect_sft_data_writes_only_successful_episodes(db_path, tmp_path):
    output_path = tmp_path / "out.jsonl"
    examples = [_example(db_path), _example(db_path)]

    correct_summary = collect_sft_data(
        ScriptedPolicy(CORRECT_SCRIPT), [examples[0]], str(output_path)
    )
    wrong_summary = collect_sft_data(
        ScriptedPolicy(WRONG_SCRIPT), [examples[1]], str(output_path)
    )

    assert correct_summary["episodes_succeeded"] == 1
    assert correct_summary["sft_examples_written"] == 2
    assert wrong_summary["episodes_succeeded"] == 0
    assert wrong_summary["sft_examples_written"] == 0

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 2  # only the correct episode's 2 turns
    for line in lines:
        record = json.loads(line)
        assert record["messages"][0]["content"] == PROMPTED_BASELINE_SYSTEM_PROMPT
        assert record["messages"][1]["role"] == "user"
        assert record["messages"][2]["role"] == "assistant"


def test_collect_sft_data_summary_counts_attempted(db_path, tmp_path):
    output_path = tmp_path / "out.jsonl"
    examples = [_example(db_path), _example(db_path), _example(db_path)]

    summary = collect_sft_data(ScriptedPolicy(CORRECT_SCRIPT), examples, str(output_path))

    assert summary["episodes_attempted"] == 3
    assert summary["episodes_succeeded"] == 3
    assert summary["sft_examples_written"] == 6
