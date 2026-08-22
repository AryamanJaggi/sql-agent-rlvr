import argparse
from pathlib import Path

import pytest

from data.spider_loader import SpiderExample
from env.grpo_env import MAX_STEPS
from train.train_grpo import LORA_TARGET_MODULES, build_grpo_dataset, training_hyperparams


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        output_dir="grpo_adapter",
        lora_rank=32,
        num_generations=8,
        batch_size=4,
        grad_accum=4,
        kl_beta=0.04,
        learning_rate=1e-6,
        max_completion_length=8192,
        epochs=1.0,
        seed=3407,
        use_vllm=False,
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


def test_vllm_off_by_default():
    # TRL's vLLM weight sync merges the LoRA and pushes raw tensors with no
    # dequantization, which shape-mismatches against a bitsandbytes-quantized
    # vLLM engine (huggingface/trl#3654, open). Must stay opt-in.
    hp = training_hyperparams(_args())
    assert hp["grpo_config"]["use_vllm"] is False


def test_vllm_can_be_opted_into():
    hp = training_hyperparams(_args(use_vllm=True))
    assert hp["grpo_config"]["use_vllm"] is True
    assert hp["grpo_config"]["vllm_mode"] == "colocate"


def test_batch_and_grad_accum_come_from_cli():
    hp = training_hyperparams(_args(batch_size=2, grad_accum=8))
    assert hp["grpo_config"]["per_device_train_batch_size"] == 2
    assert hp["grpo_config"]["gradient_accumulation_steps"] == 8


def test_rejects_batch_not_divisible_by_group_size():
    # 4 x 4 = 16 completions per generation batch, which can't split into
    # whole groups of 5.
    with pytest.raises(ValueError, match="divisible by --num-generations"):
        training_hyperparams(_args(batch_size=4, grad_accum=4, num_generations=5))


def test_accepts_batch_divisible_by_group_size():
    hp = training_hyperparams(_args(batch_size=4, grad_accum=4, num_generations=2))
    assert hp["grpo_config"]["num_generations"] == 2


def test_truncated_completions_are_masked_out_of_loss():
    # Multi-turn episodes that blow the token budget are context-budget
    # artifacts, not policy failures - they must not train the model.
    hp = training_hyperparams(_args())
    assert hp["grpo_config"]["mask_truncated_completions"] is True


def test_tool_call_iterations_bounded_by_env_step_budget():
    # TRL feeds a tool exception back as an observation and continues the
    # rollout, so SqlAgentGrpoEnv's own step guard can't end an episode by
    # itself - this is what actually bounds the turn count.
    hp = training_hyperparams(_args())
    assert hp["grpo_config"]["max_tool_calling_iterations"] == MAX_STEPS


def test_checkpoints_often_enough_to_survive_a_crash():
    # A full run is ~150 steps, so HF Trainer's save_steps default of 500
    # would write nothing until train() returns - a run dying at 95% would
    # lose everything. These runs take many hours on a preemptible VM.
    hp = training_hyperparams(_args())
    assert hp["grpo_config"]["save_strategy"] == "steps"
    assert hp["grpo_config"]["save_steps"] <= 25
    assert hp["grpo_config"]["save_total_limit"] >= 1


def test_checkpoints_keep_optimizer_state_so_a_run_can_resume():
    # save_only_model=True would shrink checkpoints a lot but strip the
    # optimizer, scheduler and RNG state, which makes resume impossible.
    # main() auto-resumes from the last checkpoint, so this must stay off.
    hp = training_hyperparams(_args())
    assert hp["grpo_config"].get("save_only_model", False) is False


def test_classic_grpo_loss_is_pinned():
    # TRL >=1.0 defaults loss_type to "dapo"; this project is comparing GRPO.
    hp = training_hyperparams(_args())
    assert hp["grpo_config"]["loss_type"] == "grpo"


def test_thinking_disabled_via_chat_template_kwargs():
    hp = training_hyperparams(_args())
    assert hp["grpo_config"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_gradient_checkpointing_enabled():
    hp = training_hyperparams(_args())
    assert hp["grpo_config"]["gradient_checkpointing"] is True


def test_seed_threaded_into_grpo_config():
    hp = training_hyperparams(_args(seed=42))
    assert hp["grpo_config"]["seed"] == 42
