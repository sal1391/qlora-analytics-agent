"""Unit tests for SQL safety guardrails (no GPU required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.safety import (
    enforce_limit,
    find_write_operations,
    is_read_only_sql,
    safety_check,
    validate_tables,
)


def test_select_is_read_only():
    assert is_read_only_sql("SELECT * FROM FACT_SALES")
    assert is_read_only_sql("WITH t AS (SELECT 1) SELECT * FROM t")
    assert is_read_only_sql("SELECT SUM(revenue) FROM FACT_SALES;")


def test_write_ops_blocked():
    for bad in [
        "DELETE FROM FACT_SALES",
        "UPDATE FACT_SALES SET revenue = 0",
        "INSERT INTO FACT_SALES VALUES (1)",
        "DROP TABLE FACT_SALES",
        "ALTER TABLE FACT_SALES ADD COLUMN x INT",
        "TRUNCATE TABLE DIM_CUSTOMER",
        "MERGE INTO FACT_FORECAST USING x ON true",
        "CREATE TABLE t AS SELECT 1",
        "GRANT SELECT ON db TO role",
    ]:
        assert not is_read_only_sql(bad), bad
        assert find_write_operations(bad)


def test_stacked_statement_blocked():
    assert not is_read_only_sql("SELECT 1; DROP TABLE FACT_SALES")


def test_comment_hidden_write_blocked():
    sql = "SELECT 1 /* DELETE FROM FACT_SALES */"
    # DELETE is inside a comment, so it should be safe.
    assert is_read_only_sql(sql)
    # But an actual delete after a comment is not.
    assert not is_read_only_sql("-- ok\nDELETE FROM FACT_SALES")


def test_validate_tables():
    ok, unknown = validate_tables(
        "SELECT * FROM FACT_SALES JOIN DIM_PRODUCT ON true",
        allowed=["FACT_SALES", "DIM_PRODUCT"],
    )
    assert ok and not unknown

    ok, unknown = validate_tables(
        "SELECT * FROM SECRET_TABLE", allowed=["FACT_SALES"]
    )
    assert not ok and "SECRET_TABLE" in unknown


def test_enforce_limit():
    limited = enforce_limit("SELECT * FROM FACT_SALES", default_limit=100)
    assert "LIMIT 100" in limited
    # Aggregates untouched.
    agg = enforce_limit("SELECT SUM(revenue) FROM FACT_SALES")
    assert "LIMIT" not in agg
    grouped = enforce_limit("SELECT year, SUM(revenue) FROM FACT_SALES GROUP BY year")
    assert "LIMIT" not in grouped


def test_safety_check_bundle():
    rep = safety_check("SELECT * FROM FACT_SALES", allowed_tables=["FACT_SALES"])
    assert rep["safe"] is True
    rep = safety_check("DELETE FROM FACT_SALES", allowed_tables=["FACT_SALES"])
    assert rep["safe"] is False
    assert "DELETE" in rep["write_ops"]
