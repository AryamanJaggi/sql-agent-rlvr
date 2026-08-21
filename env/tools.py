"""agent's action space (tools available inside an episode).

Three tools total:
  - inspect_schema(conn): str, dumps table/column names for the current DB
  - run_sql(conn, query): rows, or an error string if bad query
  - final_answer(value): ends the episode, value gets scored

"""

from __future__ import annotations

import sqlite3
import time
from typing import Union

import sqlparse

ResultSet = list[tuple]

# An un-limited query against a large table can return a result set whose
# str() is megabytes long. Feeding that back into the transcript blows past
# the model's context window and crashes generation (ran into this during 
# data collection. unbounded SELECT produced a 5MB+ prompt against an 8192-token 
# model). Cap what the agent sees here. The
# untouched result set is still what scoring uses (env.environment keeps
# the raw rows separately for that), so truncation only affects what's fed
# back into the prompt instead of the reward
MAX_OBSERVATION_CHARS = 4000


def format_result(result: Union[ResultSet, str]) -> str:
    """Render a run_sql result (or its error string) as prompt-safe text."""
    text = str(result)
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text
    row_note = f" of {len(result)} rows" if not isinstance(result, str) else ""
    return (
        text[:MAX_OBSERVATION_CHARS]
        + f"\n... (truncated{row_note} - narrow your query, e.g. add LIMIT)"
    )


def inspect_schema(conn: sqlite3.Connection) -> str:
    """Return a human-readable summary of every table and its columns."""
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()

    lines = []
    for (table_name,) in tables:
        safe_name = table_name.replace('"', '""')
        columns = conn.execute(f'PRAGMA table_info("{safe_name}")').fetchall()
        # PRAGMA table_info row shape: (cid, name, type, notnull, dflt_value, pk)
        col_desc = ", ".join(f"{col[1]} ({col[2]})" for col in columns)
        lines.append(f"{table_name}: {col_desc}")

    return "\n".join(lines) if lines else "(no tables found)"


def _is_single_select(query: str) -> bool:
    statements = [s for s in sqlparse.parse(query) if s.token_first(skip_cm=True) is not None]
    if len(statements) != 1:
        return False
    return statements[0].get_type() == "SELECT"


def run_sql(conn: sqlite3.Connection, query: str, timeout_s: float = 5.0) -> Union[ResultSet, str]:
    """Execute a single read-only SELECT. Return rows or an error string
    describing what went wrong.

    Never raises. Bad query counts as failed step agent can recover from
    """
    if not query or not query.strip():
        return "Error: empty query."
    if not _is_single_select(query):
        return "Error: only a single SELECT statement is allowed."

    deadline = time.monotonic() + timeout_s

    def _abort_if_over_deadline() -> int:
        return 1 if time.monotonic() > deadline else 0

    # sqlite calls this every 1000 VM instructions. nonzero return aborts
    # the in-flight query. only way to bound wall-clock
    # time since sqlite3 has no native query timeout
    conn.set_progress_handler(_abort_if_over_deadline, 1000)
    try:
        cursor = conn.execute(query)
        return cursor.fetchall()
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            return f"Error: query timed out after {timeout_s}s."
        return f"Error: {e}"
    except sqlite3.Error as e:
        return f"Error: {e}"
    finally:
        conn.set_progress_handler(None, 0)


def final_answer(value: object) -> object:
    """Episode terminal action
    """
    return value