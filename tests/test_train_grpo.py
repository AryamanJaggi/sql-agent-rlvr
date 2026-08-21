from pathlib import Path

from data.spider_loader import SpiderExample
from train.train_grpo import build_grpo_dataset


def _example(question="How many singers?", gold_sql="SELECT COUNT(*) FROM singer"):
    return SpiderExample(
        db_id="concert_singer",
        question=question,
        gold_sql=gold_sql,
        difficulty="hard",
        db_path=Path("/fake/concert_singer.sqlite"),
    )


def test_dataset_has_expected_columns():
    dataset = build_grpo_dataset([_example()], system_prompt="sys prompt")

    assert set(dataset.column_names) == {"prompt", "db_path", "gold_sql", "db_id"}
    assert len(dataset) == 1


def test_prompt_column_is_system_plus_user_chat_messages():
    dataset = build_grpo_dataset([_example(question="How many singers?")], system_prompt="sys prompt")

    prompt = dataset[0]["prompt"]
    assert prompt == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "How many singers?"},
    ]


def test_extra_columns_carry_per_example_context():
    ex = _example(gold_sql="SELECT name FROM singer")
    dataset = build_grpo_dataset([ex], system_prompt="sys prompt")

    assert dataset[0]["db_path"] == str(ex.db_path)
    assert dataset[0]["gold_sql"] == "SELECT name FROM singer"
    assert dataset[0]["db_id"] == "concert_singer"


def test_multiple_examples_preserve_order():
    examples = [_example(question="Q1"), _example(question="Q2"), _example(question="Q3")]
    dataset = build_grpo_dataset(examples, system_prompt="sys prompt")

    assert [row["prompt"][1]["content"] for row in dataset] == ["Q1", "Q2", "Q3"]
