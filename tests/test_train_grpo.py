import argparse
from pathlib import Path

from data.spider_loader import SpiderExample
from train.train_grpo import LORA_TARGET_MODULES, build_grpo_dataset, training_hyperparams


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        output_dir="grpo_adapter",
        lora_rank=32,
        num_generations=8,
        kl_beta=0.04,
        learning_rate=1e-6,
        max_completion_length=4096,
        epochs=1.0,
        seed=3407,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


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


# ---- training_hyperparams -----------------------------------------------


def test_lora_rank_and_alpha_come_from_cli():
    hp = training_hyperparams(_args(lora_rank=16))
    assert hp["lora"]["r"] == 16
    assert hp["lora"]["lora_alpha"] == 16
    assert hp["lora"]["target_modules"] == LORA_TARGET_MODULES
    assert hp["lora"]["task_type"] == "CAUSAL_LM"


def test_group_size_and_kl_beta_come_from_cli():
    hp = training_hyperparams(_args(num_generations=4, kl_beta=0.1))
    assert hp["grpo_config"]["num_generations"] == 4
    assert hp["grpo_config"]["beta"] == 0.1


def test_run_name_encodes_rank_and_group_size():
    hp = training_hyperparams(_args(lora_rank=64, num_generations=16))
    assert hp["grpo_config"]["run_name"] == "grpo-lora64-G16"


def test_uses_vllm_colocate_mode():
    hp = training_hyperparams(_args())
    assert hp["grpo_config"]["use_vllm"] is True
    assert hp["grpo_config"]["vllm_mode"] == "colocate"


def test_thinking_disabled_via_chat_template_kwargs():
    hp = training_hyperparams(_args())
    assert hp["grpo_config"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_gradient_checkpointing_enabled():
    hp = training_hyperparams(_args())
    assert hp["grpo_config"]["gradient_checkpointing"] is True


def test_seed_threaded_into_grpo_config():
    hp = training_hyperparams(_args(seed=42))
    assert hp["grpo_config"]["seed"] == 42
