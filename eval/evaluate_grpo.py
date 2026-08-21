"""GRPO's own eval harness: native tool-calling, consistent with how
train_grpo.py trained it. eval/evaluate.py's ReAct harness can't score a
GRPO adapter fairly because it was trained under diff protocol. 
Reuses the same tools/verifier underneath (SqlAgentGrpoEnv). 
Only the calling protocol and this eval loop are GRPO-specific.

NOTE: `parse_tool_call` is potentially fragile. Qwen3's native tool-call
serialization apparently has inconsistencies across versions/parsers. This 
assumes the JSON-inside-<tool_call> form (preferred over XML tags 
because JSON is easier to safely parse). May need to adjust this.

Run as a module:
    python -m eval.evaluate_grpo --lora-path grpo_adapter --split validation
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

from data.spider_loader import Difficulty, SpiderExample, load_spider
from env.grpo_env import SqlAgentGrpoEnv
from env.policies import GRPO_SYSTEM_PROMPT

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard", "extra")

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict


def parse_tool_call(text: str) -> ParsedToolCall | None:
    """Extract a {"name": ..., "arguments": {...}} payload from a
    <tool_call>...</tool_call> block. Returns None on parse failure 
    (callers should treat this as a failed step, not a hard crash).
    """
    match = _TOOL_CALL_RE.search(text)
    if not match:
        return None
        
    try:
        payload = json.loads(match.group("body"))
    except json.JSONDecodeError:
        return None
        
    if not isinstance(payload, dict):
        return None
        
    name = payload.get("name")
    arguments = payload.get("arguments", {})
    
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
        
    return ParsedToolCall(name=name, arguments=arguments)


@dataclass
class GrpoEpisodeResult:
    db_id: str
    success: bool
    steps_taken: int
    tool_call_count: int
    invalid_tool_call_count: int
    transcript: str


@dataclass
class GrpoEvalReport:
    n_examples: int
    success_rate: float
    avg_steps: float
    invalid_tool_call_rate: float
    by_difficulty: dict[str, float]
    results: list[GrpoEpisodeResult]


def _build_tools_schema() -> list[dict]:
    """Auto-generate the tool schema from SqlAgentGrpoEnv's methods.
    Ensures eval sees the exact same tool surface that TRL's 
    environment_factory uses during training.
    """
    from transformers.utils import get_json_schema

    env = SqlAgentGrpoEnv()
    return [
        get_json_schema(env.inspect_schema),
        get_json_schema(env.run_sql),
        get_json_schema(env.final_answer),
    ]


def run_grpo_episode(
    model,
    tokenizer,
    sampling_params,
    lora_request,
    tools_schema: list[dict],
    example: SpiderExample,
    max_steps: int = 10,
) -> GrpoEpisodeResult:
    env = SqlAgentGrpoEnv()
    env.reset(db_path=str(example.db_path), gold_sql=example.gold_sql, db_id=example.db_id)

    messages = [
        {"role": "system", "content": GRPO_SYSTEM_PROMPT},
        {"role": "user", "content": example.question},
    ]
    transcript_lines = [f"Question: {example.question}"]
    
    tool_call_count = 0
    invalid_tool_call_count = 0
    steps_taken = 0

    while not env.done and steps_taken < max_steps:
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tools=tools_schema,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        output = model.fast_generate(prompt_text, sampling_params=sampling_params, lora_request=lora_request)
        text = output[0].outputs[0].text
        
        messages.append({"role": "assistant", "content": text})
        transcript_lines.append(f"Assistant: {text}")
        steps_taken += 1

        parsed = parse_tool_call(text)
        if not parsed:
            observation = "Error: Failed to parse <tool_call> block."
            invalid_tool_call_count += 1
        else:
            tool_call_count += 1
            method = getattr(env, parsed.name, None)
            
            if method is None or parsed.name.startswith("_") or parsed.name == "reset":
                observation = f"Error: Unknown tool '{parsed.name}'."
                invalid_tool_call_count += 1
            else:
                try:
                    observation = method(**parsed.arguments)
                except Exception as e:
                    # Catch-all: malformed tool arguments shouldn't crash the episode loop
                    observation = f"Error: {e}"
                    invalid_tool_call_count += 1

        messages.append({"role": "tool", "content": str(observation)})
        transcript_lines.append(f"Tool result: {observation}")

    return GrpoEpisodeResult(
        db_id=example.db_id,
        success=env.reward >= 1.0,
        steps_taken=steps_taken,
        tool_call_count=tool_call_count,
        invalid_tool_call_count=invalid_tool_call_count,
        transcript="\n".join(transcript_lines),
    )


def evaluate_grpo(
    model,
    tokenizer,
    sampling_params,
    lora_request,
    examples: list[SpiderExample],
    max_steps: int = 10,
) -> GrpoEvalReport:
    tools_schema = _build_tools_schema()
    results = []
    
    for ex in examples:
        try:
            result = run_grpo_episode(model, tokenizer, sampling_params, lora_request, tools_schema, ex, max_steps)
        except Exception as e:
            result = GrpoEpisodeResult(
                db_id=ex.db_id, 
                success=False, 
                steps_taken=0,
                tool_call_count=0, 
                invalid_tool_call_count=0,
                transcript=f"[Episode crashed: {e!r}]",
            )
        results.append(result)

    n = len(results)
    if n == 0:
        return GrpoEvalReport(0, 0.0, 0.0, 0.0, {}, results)

    successes = [r for r in results if r.success]
    total_calls = sum(r.tool_call_count for r in results)
    total_invalid = sum(r.invalid_tool_call_count for r in results)

    by_difficulty: dict[str, float] = {}
    for tier in DIFFICULTIES:
        tier_results = [r for ex, r in zip(examples, results) if ex.difficulty == tier]
        if tier_results:
            by_difficulty[tier] = sum(1 for r in tier_results if r.success) / len(tier_results)

    return GrpoEvalReport(
        n_examples=n,
        success_rate=len(successes) / n,
        avg_steps=sum(r.steps_taken for r in results) / n,
        invalid_tool_call_rate=(total_invalid / total_calls) if total_calls else 0.0,
        by_difficulty=by_difficulty,
        results=results,
    )


def _print_report(title: str, report: GrpoEvalReport) -> None:
    print(f"\n=== {title} (n={report.n_examples}) ===")
    print(f"success_rate:           {report.success_rate:.1%}")
    print(f"avg_steps:              {report.avg_steps:.2f}")
    print(f"invalid_tool_call_rate: {report.invalid_tool_call_rate:.1%}")
    
    if report.by_difficulty:
        print("by_difficulty:")
        for tier in DIFFICULTIES:
            if tier in report.by_difficulty:
                print(f"  {tier:8s} {report.by_difficulty[tier]:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--difficulty", choices=DIFFICULTIES, default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lora-path", default=None, help="omit to sanity-check the untrained base model")
    args = parser.parse_args()

    from unsloth import FastLanguageModel
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit",
        max_seq_length=8192,
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.7,
    )
    FastLanguageModel.for_inference(model)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
    # model.load_lora() isn't reliably available on a pure-inference load -
    # see the identical fix in env/policies.py's UnslothPolicy for why.
    lora_request = LoRARequest("trained_adapter", 1, args.lora_path) if args.lora_path else None

    tiers = [args.difficulty] if args.difficulty else list(DIFFICULTIES)
    
    for tier in tiers:
        examples = load_spider(split=args.split, difficulty=tier, limit=args.limit)
        if not examples:
            continue
            
        report = evaluate_grpo(model, tokenizer, sampling_params, lora_request, examples, max_steps=args.max_steps)
        _print_report(f"{args.split}/{tier}", report)


if __name__ == "__main__":
    main()