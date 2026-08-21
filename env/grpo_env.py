"""GRPO-only environment. Wraps tools and verifier for TRL's 
`environment_factory` native tool-calling protocol (bypassing the ReAct loop).

The reward computation and underlying tool logic are identical to the baseline 
and SFT conditions. Only the calling protocol differs: the model invokes these 
methods directly rather than emitting text for `agent_loop.parse_action` to parse.

Follows TRL's environment_factory contract:
  - __init__(self): no arguments.
  - reset(self, **kwargs): called per rollout. Receives dataset row columns 
    as kwargs (e.g., db_path, gold_sql) to initialize the specific instance.
  - Public methods (except reset) with standard Args docstrings are 
    auto-discovered as tools.
  - Exceptions raised in tool methods act as environment feedback - the trainer 
    catches them and returns the error message to the model as an observation.
"""

from __future__ import annotations

import sqlite3

from env.tools import inspect_schema as _inspect_schema
from env.tools import run_sql as _run_sql
from env.verifier import verify_episode

MAX_STEPS = 10  # Matches the ReAct baseline budget (env/environment.py)


class SqlAgentGrpoEnv:
    def __init__(self):
        self.conn: sqlite3.Connection | None = None
        self.gold_sql: str | None = None
        self.last_result = None
        self.reward = 0.0
        self.done = False
        self.steps_taken = 0

    def reset(self, db_path: str, gold_sql: str, db_id: str | None = None, **kwargs) -> str | None:
        if self.conn is not None:
            self.conn.close()
            
        # Enforce read-only mode, matching the main agent loop
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.gold_sql = gold_sql
        self.last_result = None
        self.reward = 0.0
        self.done = False
        self.steps_taken = 0
        return None

    def _guard_step_budget(self) -> None:
        if self.done:
            raise ValueError("Episode already finished.")
        
        self.steps_taken += 1
        if self.steps_taken > MAX_STEPS:
            self.done = True
            raise ValueError("Max steps reached - episode over.")

    def inspect_schema(self) -> str:
        """
        List every table and its columns in the current database.

        Returns:
            A description of every table and its columns.
        """
        self._guard_step_budget()
        return _inspect_schema(self.conn)

    def run_sql(self, query: str) -> str:
        """
        Execute a single read-only SELECT statement against the database.

        Args:
            query: The SQL SELECT statement to run.

        Returns:
            The query's result rows, or an error message if the query was invalid.
        """
        self._guard_step_budget()
        result = _run_sql(self.conn, query)
        
        if not isinstance(result, str):
            # Cache the last successful result to score against (final_answer is unconstrained text)
            self.last_result = result
            
        return str(result)

    def final_answer(self, value: str) -> str:
        """
        End the episode with your answer.

        Args:
            value: Your answer to the question.

        Returns:
            A confirmation that the episode has ended.
        """
        self._guard_step_budget()
        self.done = True
        
        if self.last_result is not None:
            self.reward = 1.0 if verify_episode(self.conn, self.gold_sql, self.last_result) else 0.0
        else:
            self.reward = 0.0
            
        return "Episode finished."


def grpo_reward_func(environments: list[SqlAgentGrpoEnv], **kwargs) -> list[float]:
    return [env.reward for env in environments]