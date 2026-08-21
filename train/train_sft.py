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
            "assistant_only_loss": True,
            "report_to": "wandb",
            "run_name": f"sft-lora{args.lora_rank}-{args.epochs}ep",
        },
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
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    _wandb_login()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit",
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(model, **hp["lora"])

    dataset = Dataset.from_list(records)

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