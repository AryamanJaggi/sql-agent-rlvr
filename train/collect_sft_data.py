"""Run the untrained prompted baseline over the train split, 
keep only the episodes it actually got right, turn them into an 
SFT-ready chat dataset.

Don't just use succesful turns from GRPO train-runs because that would
then just be distilling the GRPO trained model and couldn't be used 
for experimental comparsion.

Run as a module so from data.../from env... resolve:
    python -m train.collect_sft_data --limit 150 --output sft_data.jsonl
"""

from __future__ import annotations

import argparse
import json

from data.spider_loader import Difficulty, SpiderExample, load_spider
from env.agent_loop import EpisodeResult, Policy, run_episode
from env.policies import PROMPTED_BASELINE_SYSTEM_PROMPT

SPLIT = "train"

# hard/extra are what headroom calibration found in the 20-60% band for
# the prompted baseline. easy/medium were too easy (>60%) to be worth
# collecting. No point in imitation data for stuff it already gets
# right most of the time
DIFFICULTY_TIERS: tuple[Difficulty, ...] = ("hard", "extra")


def collect_from_episode(result: EpisodeResult, system_prompt: str) -> list[dict]:
    """One chat-format SFT record per turn of a successful episode, []
    for a failed one. Each record is the standard
    {"messages": [system, user, assistant]} shape datasets.load_dataset
    ("json", ...) and TRL's SFTTrainer expect, so train_sft.py can
    consume this file directly.
    """
    if not result.success:
        return []
    return [
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": turn.prompt},
                {"role": "assistant", "content": turn.completion},
            ]
        }
        for turn in result.turns
    ]


def _append_jsonl(path: str, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def collect_sft_data(
    policy: Policy,
    examples: list[SpiderExample],
    output_path: str,
    max_steps: int = 10,
) -> dict:
    """Run every example through run_episode, write SFT records for
    successful ones to output_path incrementally (one open-write-close
    per record) so a crash or disconnect partway through a multi-hundred
    episode run doesn't lose everything already collected.

    Takes `examples` directly instead of calling load_spider itself, so
    the actual logic here is testable with a handful of local fixture
    examples and no network dependency - same separation
    eval.evaluate.evaluate() uses. Tier/split selection lives in main().

    Returns a small summary dict (attempted / succeeded / examples
    written) for the caller to print or log.
    """
    attempted = 0
    succeeded = 0
    examples_written = 0

    for i, ex in enumerate(examples):
        attempted += 1
        result = run_episode(
            policy=policy,
            db_path=ex.db_path,
            db_id=ex.db_id,
            question=ex.question,
            gold_sql=ex.gold_sql,
            max_steps=max_steps,
        )
        records = collect_from_episode(result, PROMPTED_BASELINE_SYSTEM_PROMPT)
        if records:
            succeeded += 1
            for record in records:
                _append_jsonl(output_path, record)
                examples_written += 1
        print(
            f"  [{ex.difficulty}] {i + 1}/{len(examples)}  success={result.success}  "
            f"turns_collected={len(records)}"
        )

    return {
        "episodes_attempted": attempted,
        "episodes_succeeded": succeeded,
        "sft_examples_written": examples_written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=150, help="examples per difficulty tier"
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--output", default="sft_data.jsonl")
    args = parser.parse_args()

    from env.policies import UnslothPolicy

    policy = UnslothPolicy()

    examples: list[SpiderExample] = []
    for tier in DIFFICULTY_TIERS:
        examples.extend(load_spider(split=SPLIT, difficulty=tier, limit=args.limit))

    summary = collect_sft_data(
        policy=policy,
        examples=examples,
        output_path=args.output,
        max_steps=args.max_steps,
    )

    print(f"\n=== done: {summary} ===")
    print(f"SFT dataset written to {args.output}")


if __name__ == "__main__":
    main()