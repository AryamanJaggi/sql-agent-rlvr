import argparse
import json

import pytest

from train.train_sft import (
    LORA_TARGET_MODULES,
    load_and_validate_sft_jsonl,
    training_hyperparams,
)


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        output_dir="sft_adapter",
        lora_rank=32,
        epochs=3.0,
        batch_size=2,
        grad_accum=4,
        learning_rate=2e-4,
        max_seq_length=8192,
        seed=3407,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---- training_hyperparams -----------------------------------------------


def test_lora_rank_and_alpha_come_from_cli():
    hp = training_hyperparams(_args(lora_rank=16))
    assert hp["lora"]["r"] == 16
    assert hp["lora"]["lora_alpha"] == 16
    assert hp["lora"]["target_modules"] == LORA_TARGET_MODULES


def test_max_length_is_not_left_at_trl_default():
    # TRL's SFTConfig defaults max_length to 1024, which would silently
    # truncate our multi-turn transcripts. Must be overridden.
    hp = training_hyperparams(_args(max_seq_length=8192))
    assert hp["sft_config"]["max_length"] == 8192
    assert hp["sft_config"]["max_length"] != 1024


def test_assistant_only_loss_is_enabled():
    hp = training_hyperparams(_args())
    assert hp["sft_config"]["assistant_only_loss"] is True


def test_seed_threaded_into_both_lora_and_sft_config():
    hp = training_hyperparams(_args(seed=42))
    assert hp["lora"]["random_state"] == 42
    assert hp["sft_config"]["seed"] == 42


def test_batch_and_grad_accum_come_from_cli():
    hp = training_hyperparams(_args(batch_size=4, grad_accum=8))
    assert hp["sft_config"]["per_device_train_batch_size"] == 4
    assert hp["sft_config"]["gradient_accumulation_steps"] == 8


# ---- load_and_validate_sft_jsonl -----------------------------------------


def _write(tmp_path, lines):
    path = tmp_path / "sft.jsonl"
    path.write_text("\n".join(lines))
    return str(path)


def _valid_record(user="hi", assistant="Action: final_answer\nAction Input: 1"):
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
    )


def test_valid_file_parses(tmp_path):
    path = _write(tmp_path, [_valid_record(), _valid_record(user="bye")])
    records = load_and_validate_sft_jsonl(path)
    assert len(records) == 2


def test_blank_lines_are_skipped(tmp_path):
    path = _write(tmp_path, [_valid_record(), "", "   ", _valid_record()])
    records = load_and_validate_sft_jsonl(path)
    assert len(records) == 2


def test_rejects_invalid_json(tmp_path):
    path = _write(tmp_path, ["not json at all"])
    with pytest.raises(ValueError, match="not valid JSON"):
        load_and_validate_sft_jsonl(path)


def test_rejects_wrong_message_count(tmp_path):
    bad = json.dumps({"messages": [{"role": "user", "content": "hi"}]})
    path = _write(tmp_path, [bad])
    with pytest.raises(ValueError, match="3-message"):
        load_and_validate_sft_jsonl(path)


def test_rejects_wrong_roles(tmp_path):
    bad = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "system", "content": "b"},
                {"role": "assistant", "content": "c"},
            ]
        }
    )
    path = _write(tmp_path, [bad])
    with pytest.raises(ValueError, match="expected roles"):
        load_and_validate_sft_jsonl(path)


def test_rejects_empty_file(tmp_path):
    path = _write(tmp_path, [])
    with pytest.raises(ValueError, match="no records found"):
        load_and_validate_sft_jsonl(path)
