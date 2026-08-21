import sqlite3

import pytest

from env.verifier import score, verify_episode


def test_exact_match():
    assert score([(1, "a"), (2, "b")], [(1, "a"), (2, "b")])


def test_float_tolerance_within_bound():
    assert score([(1.00001,)], [(1.0,)])


def test_float_tolerance_boundary_exceeded():
    assert not score([(1.001,)], [(1.0,)])


def test_row_order_insensitive_by_default():
    assert score([(2, "b"), (1, "a")], [(1, "a"), (2, "b")])


def test_ordered_requires_exact_sequence():
    predicted = [(2, "b"), (1, "a")]
    gold = [(1, "a"), (2, "b")]
    assert not score(predicted, gold, ordered=True)
    assert score(predicted, list(reversed(gold)), ordered=True)


def test_duplicate_rows_must_match_in_count():
    # Same set of distinct values, different multiplicities -> not a match.
    predicted = [(1, "a"), (1, "a"), (2, "b")]
    gold = [(1, "a"), (2, "b"), (2, "b")]
    assert not score(predicted, gold)


def test_duplicate_rows_matching_count():
    predicted = [(1, "a"), (1, "a"), (2, "b")]
    gold = [(1, "a"), (2, "b"), (1, "a")]
    assert score(predicted, gold)


def test_row_count_mismatch():
    assert not score([(1, "a")], [(1, "a"), (2, "b")])


def test_empty_result_sets_match():
    assert score([], [])


def test_column_order_mismatch_is_a_documented_failure():
    # (name, id) vs (id, name) for the "same" logical row -> not equal,
    # by design (see verifier.py module docstring).
    assert not score([("a", 1)], [(1, "a")])


def test_verify_episode_runs_gold_sql_against_connection():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()

    assert verify_episode(conn, "SELECT id, name FROM t", [(2, "b"), (1, "a")])
    assert not verify_episode(conn, "SELECT id, name FROM t", [(1, "a")])


def test_verify_episode_respects_order_by():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,)])
    conn.commit()

    assert verify_episode(conn, "SELECT id FROM t ORDER BY id", [(1,), (2,)])
    assert not verify_episode(conn, "SELECT id FROM t ORDER BY id", [(2,), (1,)])
