"""Unit tests for metrics + heuristic router (no GPU required)."""

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    exact_sql_match,
    execution_match,
    json_is_valid,
    normalize_sql,
    routing_accuracy,
    safety_refusal_accuracy,
    sql_execution_accuracy,
)
from src.skill_router import baseline_predict, route_skill


def _mem_con():
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE FACT_SALES AS SELECT * FROM (VALUES "
                "(1, 100.0, 40.0), (2, 200.0, 80.0)) AS t(sales_id, revenue, cost)")
    return con


def test_normalize_sql_keywords():
    n = normalize_sql("select sum(revenue) from FACT_SALES")
    assert "SELECT" in n and "FROM" in n


def test_exact_sql_match():
    assert exact_sql_match("SELECT SUM(revenue) FROM FACT_SALES",
                           "select sum(revenue) from FACT_SALES")
    assert not exact_sql_match("SELECT 1", "SELECT 2")
    assert exact_sql_match(None, None)


def test_execution_match():
    con = _mem_con()
    assert execution_match(con, "SELECT SUM(revenue) FROM FACT_SALES",
                           "SELECT SUM(revenue) FROM FACT_SALES")
    # Different order, same rows -> still matches.
    assert execution_match(
        con,
        "SELECT sales_id, revenue FROM FACT_SALES ORDER BY sales_id DESC",
        "SELECT sales_id, revenue FROM FACT_SALES ORDER BY sales_id ASC",
    )
    # Wrong query.
    assert not execution_match(con, "SELECT COUNT(*) FROM FACT_SALES",
                               "SELECT SUM(revenue) FROM FACT_SALES")
    # Broken prediction returns False, not an exception.
    assert not execution_match(con, "SELECT bogus FROM nope",
                               "SELECT SUM(revenue) FROM FACT_SALES")


def test_json_validity():
    assert json_is_valid('{"skill": "SQL_ANALYST", "safety_status": "safe"}')
    assert not json_is_valid("not json")
    assert not json_is_valid('{"foo": 1}')


def test_router_unsafe_and_clarify():
    assert route_skill("Delete all sales records")[0] == "REFUSE_UNSAFE"
    assert route_skill("How are we doing?")[0] == "NEEDS_CLARIFICATION"
    assert route_skill("What is a star schema?")[0] == "GENERAL_QA"
    assert route_skill("What was our gross margin percentage last quarter?")[0] == "FINANCE_ANALYST"


def test_baseline_predict_contract():
    pred = baseline_predict("What is the total revenue across all sales?")
    assert pred["skill"] == "SQL_ANALYST"
    assert pred["sql"] is not None
    refuse = baseline_predict("Drop the FACT_SALES table.")
    assert refuse["safety_status"] == "unsafe" and refuse["action"] == "refuse"


def test_aggregate_metrics():
    preds = [{"skill": "SQL_ANALYST", "safety_status": "safe", "sql": "SELECT 1"},
             {"skill": "REFUSE_UNSAFE", "safety_status": "unsafe", "sql": None}]
    golds = [{"skill": "SQL_ANALYST", "safety_status": "safe", "gold_sql": "SELECT 1"},
             {"skill": "REFUSE_UNSAFE", "safety_status": "unsafe", "gold_sql": None}]
    assert routing_accuracy(preds, golds) == 1.0
    assert safety_refusal_accuracy(preds, golds) == 1.0
    con = _mem_con()
    assert sql_execution_accuracy(con, preds, golds) == 1.0
