"""The eval harness: run any policy through the shared agent loop over a
Spider split and report success_rate, avg_steps, and invalid-query rate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from data.spider_loader import Difficulty, SpiderExample, load_spider
from env.agent_loop import EpisodeResult, Policy, run_episode

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard", "extra")


@dataclass
class EvalReport:
    n_examples: int
    success_rate: float
    avg_steps: float
    avg_steps_on_success: float
    invalid_sql_rate: float
    by_difficulty: dict[str, float]
    results: list[EpisodeResult] = field(repr=False)


def evaluate(
    policy: Policy,
    examples: list[SpiderExample],
    max_steps: int = 10,
) -> EvalReport:
    """Run policy through run_episode for every example, aggregate.

    One example blowing up (missing DB file, malformed gold SQL, policy
    that raises) gets recorded as a failed zero-step episode instead of
    aborting the whole run.
    """
    results: list[EpisodeResult] = []
    for ex in examples:
        try:
            result = run_episode(
                policy=policy,
                db_path=ex.db_path,
                db_id=ex.db_id,
                question=ex.question,
                gold_sql=ex.gold_sql,
                max_steps=max_steps,
            )
        except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
            result = EpisodeResult(
                transcript=f"[episode raised: {e!r}]",
                steps_taken=0,
                final_answer=None,
                success=False,
                gold_sql=ex.gold_sql,
                db_id=ex.db_id,
                invalid_sql_count=0,
                run_sql_count=0,
                turns=[],
            )
        results.append(result)

    return _build_report(examples, results)


def _build_report(examples: list[SpiderExample], results: list[EpisodeResult]) -> EvalReport:
    n = len(results)
    if n == 0:
        return EvalReport(0, 0.0, 0.0, 0.0, 0.0, {}, results)

    successes = [r for r in results if r.success]
    total_invalid = sum(r.invalid_sql_count for r in results)
    total_run_sql_calls = sum(r.run_sql_count for r in results)

    by_difficulty: dict[str, float] = {}
    for tier in DIFFICULTIES:
        tier_results = [r for ex, r in zip(examples, results) if ex.difficulty == tier]
        if tier_results:
            tier_successes = sum(1 for r in tier_results if r.success)
            by_difficulty[tier] = tier_successes / len(tier_results)

    return EvalReport(
        n_examples=n,
        success_rate=len(successes) / n,
        avg_steps=sum(r.steps_taken for r in results) / n,
        avg_steps_on_success=(sum(r.steps_taken for r in successes) / len(successes)) if successes else 0.0,
        invalid_sql_rate=(total_invalid / total_run_sql_calls) if total_run_sql_calls else 0.0,
        by_difficulty=by_difficulty,
        results=results,
    )


def _print_report(title: str, report: EvalReport) -> None:
    print(f"\n=== {title} (n={report.n_examples}) ===")
    print(f"success_rate:         {report.success_rate:.1%}")
    print(f"avg_steps:             {report.avg_steps:.2f}")
    print(f"avg_steps_on_success:  {report.avg_steps_on_success:.2f}")
    print(f"invalid_sql_rate:      {report.invalid_sql_rate:.1%}")
    if report.by_difficulty:
        print("by_difficulty:")
        for tier in DIFFICULTIES:
            if tier in report.by_difficulty:
                print(f"  {tier:8s} {report.by_difficulty[tier]:.1%}")


def _wandb_login() -> None:
    """Never hardcode tokens - Colab Secrets panel first, env var locally."""
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


def _wandb_log_report(title: str, report: EvalReport) -> None:
    import wandb

    wandb.log(
        {
            f"{title}/success_rate": report.success_rate,
            f"{title}/avg_steps": report.avg_steps,
            f"{title}/avg_steps_on_success": report.avg_steps_on_success,
            f"{title}/invalid_sql_rate": report.invalid_sql_rate,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default=None,
                         help="omit to evaluate all four tiers, broken out separately")
    parser.add_argument("--limit", type=int, default=30,
                         help="examples per difficulty tier (or total, if --difficulty is set)")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--policy", choices=["prompted", "unsloth"], default="prompted")
    parser.add_argument("--lora-path", default=None,
                         help="trained adapter to evaluate (unsloth policy only); omit for the untrained baseline")
    parser.add_argument("--wandb-project", default=None,
                         help="if set, logs each report's metrics to this W&B project")
    args = parser.parse_args()

    if args.policy == "prompted":
        from env.policies import PromptedPolicy
        policy = PromptedPolicy()
    elif args.policy == "unsloth":
        from env.policies import UnslothPolicy
        policy = UnslothPolicy(lora_path=args.lora_path)

    if args.wandb_project:
        import wandb
        _wandb_login()
        wandb.init(project=args.wandb_project)

    if args.difficulty is not None:
        examples = load_spider(split=args.split, difficulty=args.difficulty, limit=args.limit)
        report = evaluate(policy, examples, max_steps=args.max_steps)
        title = f"{args.split}/{args.difficulty}"
        _print_report(title, report)
        if args.wandb_project:
            _wandb_log_report(title, report)
        return

    for tier in DIFFICULTIES:
        examples = load_spider(split=args.split, difficulty=tier, limit=args.limit)
        if not examples:
            continue
        report = evaluate(policy, examples, max_steps=args.max_steps)
        title = f"{args.split}/{tier}"
        _print_report(title, report)
        if args.wandb_project:
            _wandb_log_report(title, report)


if __name__ == "__main__":
    main()