"""The per-episode state machine: wires the ReAct action text the policy
emits to the tools in tools.py, tracks when an episode is over.

Reward isn't computed here, environment only sees what the agent does, 
doesn't need to know the real answer. 
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from env.tools import ResultSet, final_answer, format_result, inspect_schema, run_sql

ACTION_FORMAT_HELP = (
    "Respond with exactly one action per turn, in this format:\n"
    "Action: <tool_name>\n"
    "Action Input: <input>\n\n"
    "Available tools:\n"
    "  inspect_schema            (no input needed)\n"
    "  run_sql                   (input: a single SELECT statement)\n"
    "  final_answer               (input: your answer)"
)


@dataclass
class ToolCall:
    tool: str
    input: str


@dataclass
class Environment:
    conn: sqlite3.Connection
    question: str
    max_steps: int = 10

    steps_taken: int = field(default=0, init=False)
    done: bool = field(default=False, init=False)
    final_value: object = field(default=None, init=False)
    last_result: ResultSet | None = field(default=None, init=False)
    invalid_sql_count: int = field(default=0, init=False)
    run_sql_count: int = field(default=0, init=False)

    def reset(self) -> str:
        self.steps_taken = 0
        self.done = False
        self.final_value = None
        self.last_result = None
        self.invalid_sql_count = 0
        self.run_sql_count = 0
        return (
            f"Question: {self.question}\n\n"
            f"{ACTION_FORMAT_HELP}"
        )

    def step(self, tool_call: ToolCall | None) -> tuple[str, bool]:
        """Dispatch one parsed action, return (observation, done).

        tool_call=None means the policy's output didn't parse into an
        action at all. Still consumes a step (so it can't stall forever
        spamming garbage) and returns guidance instead of crashing.
        """
        if self.done:
            raise RuntimeError("step() called after episode already finished")

        self.steps_taken += 1

        if tool_call is None:
            observation = f"Could not parse an action from your response.\n\n{ACTION_FORMAT_HELP}"
        elif tool_call.tool == "inspect_schema":
            observation = inspect_schema(self.conn)
        elif tool_call.tool == "run_sql":
            result = run_sql(self.conn, tool_call.input)
            self.run_sql_count += 1
            if isinstance(result, str):
                # error string (bad SQL, write attempt, timeout). counted
                # against run_sql_count (actual queries attempted) not
                # steps_taken, since evaluate.py reports this as the
                # invalid-query rate metric
                self.invalid_sql_count += 1
            else:
                # successful query becomes the scoring candidate.
                # we score off the agent's last successful run_sql result,
                # not whatever text it types into final_answer
                self.last_result = result
            observation = format_result(result)
        elif tool_call.tool == "final_answer":
            self.final_value = final_answer(tool_call.input)
            self.done = True
            observation = "Episode finished."
        else:
            observation = (
                f"Unknown tool {tool_call.tool!r}.\n\n{ACTION_FORMAT_HELP}"
            )

        if not self.done and self.steps_taken >= self.max_steps:
            self.done = True
            observation += "\n\n(Step limit reached - episode ending without a final_answer.)"

        return observation, self.done