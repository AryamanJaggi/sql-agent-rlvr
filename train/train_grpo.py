"""GRPO training via TRL's environment_factory.

Training drives rollouts through native tool-calling instead of ReAct-text
agent_loop.py. Tools and reward underneath are identical to the other two
conditions (same env.tools functions, same env.verifier scoring). Only the
calling protocol differs.

environment_factory needs TRL>=1.0, which Unsloth doesn't support (every
release, including live main, caps trl<=0.24.0 - confirmed against both
PyPI and Unsloth's own pyproject.toml). So unlike train_sft.py, this script
does NOT use Unsloth: plain transformers/peft/bitsandbytes QLoRA + a real
trl>=1.0 instead, same base model repo either way. Run this in its own
Colab runtime, separate from every other Unsloth-based cell in this repo's
notebook (trl==0.24.0 and trl>=1.0 can't both be installed at once).

Run as a module so from data.../from env... resolve:
    python -m train.train_grpo --limit 150 --output-dir grpo_adapter
"""

from __future__ import annotations

import argparse

from data.spider_loader import Difficulty, SpiderExample, load_spider
from env.grpo_env import SqlAgentGrpoEnv, grpo_reward_func
from env.policies import GRPO_SYSTEM_PROMPT

MODEL_NAME = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

SPLIT = "train"
DIFFICULTY_TIERS: tuple[Difficulty, ...] = ("hard", "extra")


def build_grpo_dataset(examples: list[SpiderExample], system_prompt: str):
    """Pure function, only needs `datasets` (already a local dependency),
    no peft/trl. db_path/gold_sql/db_id ride along as extra dataset
    columns because TRL's environment_factory passes every dataset
    column into reset(**kwargs) - how each rollout's SqlAgentGrpoEnv
    instance learns which Spider example it's supposed to be answering.
    """
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ex.question},
                ]
                for ex in examples
            ],
            "db_path": [str(ex.db_path) for ex in examples],
            "gold_sql": [ex.gold_sql for ex in examples],
            "db_id": [ex.db_id for ex in examples],
        }
    )


def training_hyperparams(args: argparse.Namespace) -> dict:
    """Pure mapping from CLI args to the kwargs the Colab-only main()
    hands to peft/TRL. No peft/trl import needed, so this is testable
    locally without those packages installed.
    """
    run_name = f"grpo-lora{args.lora_rank}-G{args.num_generations}"
    return {
        "lora": {
            "r": args.lora_rank,
            "lora_alpha": args.lora_rank,
            "target_modules": LORA_TARGET_MODULES,
            "task_type": "CAUSAL_LM",
        },
        "grpo_config": {
            "output_dir": args.output_dir,
            "num_train_epochs": args.epochs,
            "num_generations": args.num_generations,
            "beta": args.kl_beta,
            "learning_rate": args.learning_rate,
            "max_completion_length": args.max_completion_length,
            "seed": args.seed,
            "use_vllm": True,
            "vllm_mode": "colocate",
            "chat_template_kwargs": {"enable_thinking": False},
            "gradient_checkpointing": True,
            "report_to": "wandb",
            "run_name": run_name,
            "log_completions": True,
        },
    }


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
    parser.add_argument("--limit", type=int, default=150, help="examples per difficulty tier")
    parser.add_argument("--output-dir", default="grpo_adapter")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-generations", type=int, default=8, help="group size G")
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--wandb-project", default="sql-agent-rlvr")
    args = parser.parse_args()

    examples: list[SpiderExample] = []
    for tier in DIFFICULTY_TIERS:
        examples.extend(load_spider(split=SPLIT, difficulty=tier, limit=args.limit))
    print(f"Loaded {len(examples)} training examples across {DIFFICULTY_TIERS}")

    dataset = build_grpo_dataset(examples, GRPO_SYSTEM_PROMPT)
    hp = training_hyperparams(args)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    _wandb_login()
    import wandb
    wandb.init(project=args.wandb_project, name=hp["grpo_config"]["run_name"])

    # Seeds LoRA's own random init - get_peft_model() below has no seed
    # param of its own (Unsloth's from_pretrained/get_peft_model path
    # elsewhere in this repo takes care of this itself; here we're not
    # going through Unsloth, so it's on us).
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto"
    )
    model = get_peft_model(model, LoraConfig(**hp["lora"]))

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_func,
        train_dataset=dataset,
        args=GRPOConfig(**hp["grpo_config"]),
        environment_factory=SqlAgentGrpoEnv,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter saved to {args.output_dir}")
    print(f"Next: python -m eval.evaluate_grpo --lora-path {args.output_dir} --split validation")


if __name__ == "__main__":
    main()
