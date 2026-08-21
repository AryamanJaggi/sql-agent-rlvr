import argparse
import json

import pytest

from train.train_sft import (
    LORA_TARGET_MODULES,
    load_and_validate_sft_jsonl,
    tokenize_and_mask,
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


def test_assistant_only_loss_flag_not_used():
    # Masking is done ourselves in tokenize_and_mask (see its tests below) -
    # relying on TRL/Unsloth's own assistant_only_loss auto-detection is
    # exactly what crashed real training (unsloth-zoo#323: patched
    # SFTTrainer doesn't recognize a plain "messages" column). Passing a
    # pre-tokenized dataset with this flag still set would be misleading
    # dead config, so it must not be here.
    hp = training_hyperparams(_args())
    assert "assistant_only_loss" not in hp["sft_config"]


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


# ---- tokenize_and_mask -----------------------------------------------


class FakeTokenizer:
    """apply_chat_template stand-in that mimics the one property
    tokenize_and_mask's masking logic depends on: add_generation_prompt
    renders the SAME assistant-opening tag that's already the first
    thing the assistant message's own block produces in the full
    render, so a prefix render is a literal token prefix of the full
    one - exactly how real chat templates work, since they render each
    message's role-open tag before its content regardless of what
    comes after.
    """

    _ROLE_IDS = {"system": 1, "user": 2, "assistant": 3}
    OPEN_TAG_LEN = 2
    CONTENT_LEN = 3
    CLOSE_TAG_LEN = 1

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize is True
        ids = []
        for m in messages:
            role_id = self._ROLE_IDS[m["role"]]
            ids += [100 + role_id] * self.OPEN_TAG_LEN
            ids += [200 + i for i in range(self.CONTENT_LEN)]
            ids += [900 + role_id] * self.CLOSE_TAG_LEN
        if add_generation_prompt:
            ids += [100 + self._ROLE_IDS["assistant"]] * self.OPEN_TAG_LEN
        return ids


_MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "Action: final_answer\nAction Input: 1"},
]


def test_tokenize_and_mask_masks_exactly_through_the_assistant_open_tag():
    tok = FakeTokenizer()
    example = tokenize_and_mask(_MESSAGES, tok, max_length=1000)

    full_ids = tok.apply_chat_template(_MESSAGES, tokenize=True, add_generation_prompt=False)
    assert example["input_ids"] == full_ids
    assert example["attention_mask"] == [1] * len(full_ids)

    # Everything through the assistant's opening tag is masked; its
    # content + closing tag (the last CONTENT_LEN + CLOSE_TAG_LEN
    # tokens) are the only ones left for the model to learn from.
    unmasked_len = FakeTokenizer.CONTENT_LEN + FakeTokenizer.CLOSE_TAG_LEN
    assert example["labels"][:-unmasked_len] == [-100] * (len(full_ids) - unmasked_len)
    assert example["labels"][-unmasked_len:] == full_ids[-unmasked_len:]


def test_tokenize_and_mask_truncates_to_max_length():
    tok = FakeTokenizer()
    example = tokenize_and_mask(_MESSAGES, tok, max_length=5)

    assert len(example["input_ids"]) == 5
    assert len(example["labels"]) == 5
    assert len(example["attention_mask"]) == 5


def test_tokenize_and_mask_labels_same_length_as_input_ids():
    tok = FakeTokenizer()
    example = tokenize_and_mask(_MESSAGES, tok, max_length=1000)
    assert len(example["labels"]) == len(example["input_ids"])
