"""Does PEFT on collect_sft_data.py's output

Trains a QLoRA adapter only, doesn't run a held-out eval. Running eval 
right after training means loading the model a second time
in the same GPU process (a fresh fast_inference=True load on top of the
training load), unnecessary VRAM pressure for no real benefit. Once this
finishes and prints where the adapter landed, can run:

    python -m eval.evaluate --policy unsloth --lora-path <output-dir> --split validation

which reuses the already-tested eval harness directly, evaluating this
adapter the same way the untrained baseline was evaluated during
headroom calibration.

Run as a module so from data.../from env... resolve:
    python -m train.train_sft --data sft_data.jsonl --output-dir sft_adapter
"""


from __future__ import annotations

import argparse
import json

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def training_hyperparams(args: argparse.Namespace) -> dict:
    """Pure mapping from CLI args to the kwargs the Colab-only main()
    hands to Unsloth/TRL. No peft/trl import needed, so this is testable locally
    without those packages installed.
    """
    return {
        "lora": {
            "r": args.lora_rank,
            "lora_alpha": args.lora_rank,
            "target_modules": LORA_TARGET_MODULES,
            "use_gradient_checkpointing": "unsloth",
            "random_state": args.seed,
        },
        "sft_config": {
            "output_dir": args.output_dir,
            "num_train_epochs": args.epochs,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "learning_rate": args.learning_rate,
            "optim": "paged_adamw_8bit",
            "seed": args.seed,
            "max_length": args.max_seq_length,
            "report_to": "wandb",
            "run_name": f"sft-lora{args.lora_rank}-{args.epochs}ep",
        },
    }


def tokenize_and_mask(messages: list[dict], tokenizer, max_length: int) -> dict:
    """Pre-tokenize one SFT record with the loss masked to only the
    assistant turn's tokens.

    Two real problems with letting Unsloth/TRL handle this
    automatically, both hit during actual training runs: Unsloth's
    patched SFTTrainer doesn't recognize a plain "messages" column and
    crashes demanding a `formatting_func` (unsloth-zoo#323); and its
    own train_on_responses_only() completion-masking helper is
    documented as fragile for Qwen3's chat template
    (unslothai/unsloth#2771), since it matches on hand-guessed marker
    substrings that silently mask everything if they're a whitespace
    off from what the template actually renders.

    Instead, render the prompt prefix (system+user, with
    add_generation_prompt=True so it includes the assistant turn's
    opening tag) and the full conversation through the SAME
    apply_chat_template call. add_generation_prompt just emits the
    same tag that's already the first thing the assistant message's
    own rendering produces, so prefix_ids is a literal token prefix of
    full_ids for any well-formed chat template - masking up to that
    length can't drift out of sync with whatever the template does.
    A dataset that already has input_ids also sidesteps Unsloth's
    formatting_func requirement entirely, since that's only enforced
    when a dataset needs converting to text first.
    """
    prefix_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )[:max_length]
    prefix_len = min(len(prefix_ids), len(full_ids))

    labels = list(full_ids)
    labels[:prefix_len] = [-100] * prefix_len

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


_REQUIRED_ROLES = ("system", "user", "assistant")


def load_and_validate_sft_jsonl(path: str) -> list[dict]:
    """Parse collect_sft_data.py's output, check each record's shape up
    front, so a malformed file fails with a clear message here instead
    of deep inside datasets/TRL's dataset prep.
    """
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: not valid JSON: {e}") from e

            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                raise ValueError(
                    f"{path}:{line_no}: expected a 3-message [system, user, assistant] "
                    f"record, got {messages!r}"
                )
            roles = tuple(m.get("role") for m in messages)
            if roles != _REQUIRED_ROLES:
                raise ValueError(
                    f"{path}:{line_no}: expected roles {_REQUIRED_ROLES}, got {roles!r}"
                )
            records.append(record)

    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def _wandb_login() -> None:
    import os

    key = None
    try:
        from google.colab import userdata
        key = userdata.get("WANDB_API_KEY")
    except Exception:
        key = os.environ.get("WANDB_API_KEY")
    if key:
        import wandb
        wandb.login(key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="collect_sft_data.py's JSONL output")
    parser.add_argument("--output-dir", default="sft_adapter")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--wandb-project", default="sql-agent-rlvr")
    args = parser.parse_args()

    records = load_and_validate_sft_jsonl(args.data)
    print(f"Loaded {len(records)} SFT examples from {args.data}")
    hp = training_hyperparams(args)

    from datasets import Dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    _wandb_login()
    import wandb
    wandb.init(project=args.wandb_project, name=hp["sft_config"]["run_name"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit",
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, **hp["lora"])

    dataset = Dataset.from_list(records)
    dataset = dataset.map(
        lambda ex: tokenize_and_mask(ex["messages"], tokenizer, args.max_seq_length),
        remove_columns=["messages"],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**hp["sft_config"]),
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter saved to {args.output_dir}")
    print(
        "Next: python -m eval.evaluate --policy unsloth "
        f"--lora-path {args.output_dir} --split validation"
    )


if __name__ == "__main__":
    main()