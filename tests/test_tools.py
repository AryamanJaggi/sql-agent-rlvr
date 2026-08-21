import sqlite3

import pytest

from env.tools import final_answer, inspect_schema, run_sql


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute('CREATE TABLE "singer" (id INTEGER PRIMARY KEY, name TEXT)')
    c.execute('CREATE TABLE "concert dates" (id INTEGER PRIMARY KEY, singer_id INTEGER)')
    c.executemany('INSERT INTO "singer" VALUES (?, ?)', [(1, "Alice"), (2, "Bob")])
    c.commit()
    yield c
    c.close()


def test_inspect_schema_lists_tables_and_columns(conn):
    schema = inspect_schema(conn)
    assert "singer: id (INTEGER), name (TEXT)" in schema
    assert "concert dates: id (INTEGER), singer_id (INTEGER)" in schema


def test_inspect_schema_empty_db():
    empty_conn = sqlite3.connect(":memory:")
    assert inspect_schema(empty_conn) == "(no tables found)"


def test_run_sql_happy_path(conn):
    rows = run_sql(conn, "SELECT id, name FROM singer ORDER BY id")
    assert rows == [(1, "Alice"), (2, "Bob")]


def test_run_sql_rejects_write_statements(conn):
    for bad in [
        "INSERT INTO singer VALUES (3, 'Eve')",
        "UPDATE singer SET name = 'x'",
        "DELETE FROM singer",
        "DROP TABLE singer",
        "ALTER TABLE singer ADD COLUMN age INTEGER",
        "PRAGMA writable_schema = 1",
    ]:
        result = run_sql(conn, bad)
        assert isinstance(result, str) and result.startswith("Error:")

    # The rejected statements must not have actually run.
    assert run_sql(conn, "SELECT COUNT(*) FROM singer") == [(2,)]


def test_run_sql_rejects_stacked_statements(conn):
    result = run_sql(conn, "SELECT * FROM singer; DROP TABLE singer;")
    assert isinstance(result, str) and result.startswith("Error:")
    assert run_sql(conn, "SELECT COUNT(*) FROM singer") == [(2,)]


def test_run_sql_returns_error_string_on_bad_syntax(conn):
    result = run_sql(conn, "SELECT FROM WHERE")
    assert isinstance(result, str) and result.startswith("Error:")


def test_run_sql_rejects_empty_query(conn):
    assert run_sql(conn, "").startswith("Error:")
    assert run_sql(conn, "   ").startswith("Error:")


def test_run_sql_times_out_on_slow_query(conn):
    # A recursive CTE that generates a lot of rows, given an
    # unreasonably tight deadline - forces the progress handler to fire.
    slow_query = (
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 50000000) "
        "SELECT COUNT(*) FROM cnt"
    )
    result = run_sql(conn, slow_query, timeout_s=0.001)
    assert isinstance(result, str) and "timed out" in result


def test_run_sql_recovers_after_timeout(conn):
    slow_query = (
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM cnt WHERE x < 50000000) "
        "SELECT COUNT(*) FROM cnt"
    )
    run_sql(conn, slow_query, timeout_s=0.001)
    # The progress handler must be cleared afterwards, or every later
    # query on this connection would spuriously "time out" too.
    assert run_sql(conn, "SELECT COUNT(*) FROM singer") == [(2,)]


def test_final_answer_passes_value_through():
    assert final_answer(42) == 42
    assert final_answer("Alice") == "Alice"
