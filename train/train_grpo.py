"""GRPO training via TRL's environment_factory.

training drives rollouts through native tool-calling instead of ReAct-text 
agent_loop.py Tools and reward underneath are identical to the other two conditions 
(same env.tools functions, same env.verifier scoring). Only the calling protocol differs.

Run as a module so from data.../from env... resolve:
    python -m train.train_grpo --limit 150 --output-dir grpo_adapter
"""

from __future__ import annotations

import argparse

from data.spider_loader import Difficulty, SpiderExample, load_spider
from env.grpo_env import SqlAgentGrpoEnv, grpo_reward_func
from env.policies import GRPO_SYSTEM_PROMPT

SPLIT = "train"
DIFFICULTY_TIERS: tuple[Difficulty, ...] = ("hard", "extra")


def build_grpo_dataset(examples: list[SpiderExample], system_prompt: str):
    """Pure function, only needs `datasets` (already a local dependency),
    no unsloth/trl. db_path/gold_sql/db_id ride along as extra dataset
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

    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    _wandb_login()
    import wandb
    run_name = f"grpo-lora{args.lora_rank}-G{args.num_generations}"
    wandb.init(project=args.wandb_project, name=run_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit",
        max_seq_length=args.max_completion_length + 1024,
        load_in_4bit=True,
        fast_inference=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        num_generations=args.num_generations,
        beta=args.kl_beta,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        seed=args.seed,
        use_vllm=True,
        vllm_mode="colocate",
        chat_template_kwargs={"enable_thinking": False},
        report_to="wandb",
        run_name=run_name,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_func,
        train_dataset=dataset,
        args=grpo_config,
        environment_factory=SqlAgentGrpoEnv,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter saved to {args.output_dir}")
    print(f"Next: python -m eval.evaluate_grpo --lora-path {args.output_dir} --split validation")


if __name__ == "__main__":
    main()