"""Agent loop shared by all 3 model variants (run_episode())

The baseline, SFT-trained, and GRPO-trained model versions all use this
exactt function to make sure nothing about episode mechanics is differnet
between them and ensure a fair comparison.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from env.environment import Environment, ToolCall
from env.verifier import verify_episode

Policy = Callable[[str], str]

_ACTION_RE = re.compile(
    r"Action:\s*(?P<tool>inspect_schema|run_sql|final_answer)"
    r"(?:\s*\n\s*Action Input:\s*(?P<input>.*))?",
    re.IGNORECASE | re.DOTALL,
)


def parse_action(text: str) -> ToolCall | None:
    """Parse a ReAct-style 'Action: <tool>\\nAction Input: <input>' block
    out of raw policy text.

    Action Input is optional in the regex on purpose. inspect_schema
    takes no input, and a model correctly omitting the line for it
    shouldn't count as malformed. Used to be stricter regex that
    required the Input line but it broke every inspect_schema call. agent
    couldn't see schema for the whole episode and in eval it made it look 
    like a model failure but was parser bug.

    Returns None only when no "Action: <known tool>" shows up at all.
    Callers treat that as a failed step. small models still drift from
    the format sometimes.
    """
    match = _ACTION_RE.search(text)
    if match is None:
        return None

    tool = match.group("tool").strip().lower()
    input_ = (match.group("input") or "").strip()
    # action input is free text. If policy kept outputting past it
    # cut at first blank line
    input_ = input_.split("\n\n")[0].strip()
    return ToolCall(tool=tool, input=input_)


@dataclass
class Turn:
    """One policy call within an episode.

    Needed for per-turn SFT examples: EpisodeResult's flattened
    `transcript` string can't be split back into (prompt, completion)
    pairs without re-deriving turn boundaries.
    """

    prompt: str
    completion: str


@dataclass
class EpisodeResult:
    transcript: str
    steps_taken: int
    final_answer: object
    success: bool
    gold_sql: str
    db_id: str
    invalid_sql_count: int
    run_sql_count: int
    turns: list[Turn]


def run_episode(
    policy: Policy,
    db_path: Path,
    db_id: str,
    question: str,
    gold_sql: str,
    max_steps: int = 10,
) -> EpisodeResult:
    """Drive one episode to completion and score it.

    Scoring note: final_answer's input is free text ("56", "Alice, Bob",
    whatever). Re-parsing that back into a row set to compare against
    gold is fragile, so instead score the result set of the agent's
    last successful run_sql call, which is already structured. Assumes
    the last query before final_answer is the one it means to use as final


    DB connection opens in sqlite's true read-only mode. 
    even a gap in tools.run_sql's own statement-type check still
    can't produce a write, since the OS-level connection refuses one.
    """

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        env = Environment(conn=conn, question=question, max_steps=max_steps)
        transcript = env.reset()
        turns: list[Turn] = []

        while not env.done:
            action_text = policy(transcript)
            turns.append(Turn(prompt=transcript, completion=action_text))
            transcript += f"\n{action_text}"
            tool_call = parse_action(action_text)
            observation, _done = env.step(tool_call)
            transcript += f"\nObservation: {observation}"

        success = False
        if env.final_value is not None and env.last_result is not None:
            success = verify_episode(conn, gold_sql, env.last_result)

        return EpisodeResult(
            transcript=transcript,
            steps_taken=env.steps_taken,
            final_answer=env.final_value,
            success=success,
            gold_sql=gold_sql,
            db_id=db_id,
            invalid_sql_count=env.invalid_sql_count,
            run_sql_count=env.run_sql_count,
            turns=turns,
        )
    finally:
        conn.close()