"""Execution-match scoring (EX). Reward signal for rest of project.

Given a SQLite DB, a gold SQL query, and the agent's predicted result set,
decide if the prediction is correct. Compare result sets instead of SQL
text so equivalent queries both score correct.

Verifier is kept simple to decrease exposure to reward hacking risk
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Sequence

Row = tuple
ResultSet = Sequence[Row]

# rows get rounded to 4 decimals before comparing floats.
# not real tolerance matching but spide results mostly int anyway
FLOAT_TOLERANCE_DECIMALS = 4


def _normalize_value(value: object) -> object:
    if isinstance(value, float):
        return round(value, FLOAT_TOLERANCE_DECIMALS)
    return value


def _normalize_row(row: Row) -> Row:
    return tuple(_normalize_value(v) for v in row)


def score(predicted_rows: ResultSet, gold_rows: ResultSet, ordered: bool = False) -> bool:
    """Compare two result sets for execution match.

    Column order isn't normalized. agent's SELECT has to project
    columns in the same order as gold. Might cause failures.
    Watch out during headroom calibration

    ordered=True does exact sequence match (use when gold has ORDER BY).
    ordered=False compares as a multiset. order doesn't matter but
    duplicate counts still have to line up.
    """
    if len(predicted_rows) != len(gold_rows):
        return False

    predicted_norm = [_normalize_row(r) for r in predicted_rows]
    gold_norm = [_normalize_row(r) for r in gold_rows]

    if ordered:
        return predicted_norm == gold_norm
    return Counter(predicted_norm) == Counter(gold_norm)


def run_query(conn: sqlite3.Connection, sql: str) -> list[Row]:
    """Run sql and return all rows. Exceptions propagate on purpose.
    If gold SQL itself is broken want hard failure, unlike
    agent-issued SQL which tools.run_sql handles defensively.
    """
    cursor = conn.execute(sql)
    return cursor.fetchall()


def verify_episode(conn: sqlite3.Connection, gold_sql: str, predicted_rows: ResultSet) -> bool:
    """Run gold SQL against conn. score predicted_rows against it."""
    gold_rows = run_query(conn, gold_sql)
    #come back to this. false positive if subquery has its own ORDER BY
    ordered = "order by" in gold_sql.lower()
    return score(predicted_rows, gold_rows, ordered=ordered)