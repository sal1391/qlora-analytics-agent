"""Heuristic skill router — the offline ``template_baseline``.

This provides a rule-based router and a template SQL generator so the full
evaluation pipeline can run without any GPU or trained adapter. It mirrors the
production ``orchestrator -> skill router`` pattern.
"""

from __future__ import annotations

import re

from .safety import find_write_operations

# Keyword cues per skill (lower-cased, checked as substrings / word matches).
_UNSAFE_CUES = [
    "delete", "update", "insert", "drop", "truncate", "alter",
    "merge", "wipe", "remove the", "grant", "revoke",
]
_CLARIFY_CUES = [
    "last period", "how are we doing", "the top ones", "good customers",
    "compare it to before", "which products are best", "recent data",
    "how much did we make", "the trend", "what changed",
]
_OUT_OF_SCOPE_CUES = [
    "stock price", "employees", "weather", "headcount", "marketing spend",
    "support tickets", "travel schedule", "parking",
]
_DOC_CUES = ["playbook", "policy document", "knowledge base", "onboarding guide",
             "contract clause", "section of", "document say"]
_GENERAL_CUES = ["what is a", "explain the difference", "what does yoy",
                 "overview of", "purpose of a"]
_FINANCE_CUES = ["gross margin", "margin", "year-over-year", "yoy", "cost of goods",
                 "forecast attainment", "cogs", "profitability"]
_SALES_CUES = ["accounts", "trending up", "growth customers", "declining orders",
               "win pattern", "territories", "rank sales"]


def _has_any(text: str, cues: list[str]) -> bool:
    return any(c in text for c in cues)


def route_skill(question: str) -> tuple[str, str]:
    """Return ``(skill, safety_status)`` for a natural-language question."""
    q = question.lower().strip()

    if _has_any(q, _UNSAFE_CUES) or find_write_operations(question):
        return "REFUSE_UNSAFE", "unsafe"
    if _has_any(q, _OUT_OF_SCOPE_CUES):
        return "GENERAL_QA", "out_of_scope"
    # Vague/short questions -> clarification.
    if _has_any(q, _CLARIFY_CUES) or (len(q.split()) <= 5 and "?" not in q and not _has_any(q, _FINANCE_CUES)):
        return "NEEDS_CLARIFICATION", "needs_clarification"
    if _has_any(q, _DOC_CUES):
        return "DOCUMENT_SEARCH", "safe"
    if _has_any(q, _GENERAL_CUES):
        return "GENERAL_QA", "safe"
    if _has_any(q, _FINANCE_CUES):
        return "FINANCE_ANALYST", "safe"
    if _has_any(q, _SALES_CUES):
        return "SALES_INTELLIGENCE", "safe"
    return "SQL_ANALYST", "safe"


# ---- Template SQL generation for the baseline ----------------------------

_TEMPLATE_SQL: list[tuple[re.Pattern, str]] = [
    (re.compile(r"total revenue.*all|overall .*revenue|total revenue$", re.I),
     "SELECT SUM(revenue) AS total_revenue FROM FACT_SALES"),
    (re.compile(r"total units|units.*sold", re.I),
     "SELECT SUM(quantity) AS units_sold FROM FACT_SALES"),
    (re.compile(r"average discount|avg discount", re.I),
     "SELECT AVG(discount_pct) AS avg_discount FROM FACT_SALES"),
    (re.compile(r"gross margin percentage|gross margin pct", re.I),
     "SELECT (SUM(revenue) - SUM(cost)) / NULLIF(SUM(revenue), 0) * 100 AS gross_margin_pct FROM FACT_SALES"),
    (re.compile(r"gross margin", re.I),
     "SELECT SUM(revenue) - SUM(cost) AS gross_margin FROM FACT_SALES"),
    (re.compile(r"top .*product.*revenue", re.I),
     "SELECT p.product_name, SUM(s.revenue) AS total_revenue FROM FACT_SALES s "
     "JOIN DIM_PRODUCT p ON s.product_id = p.product_id GROUP BY p.product_name "
     "ORDER BY total_revenue DESC LIMIT 10"),
    (re.compile(r"customers.*most revenue|top .*customers", re.I),
     "SELECT c.customer_name, SUM(s.revenue) AS total_revenue FROM FACT_SALES s "
     "JOIN DIM_CUSTOMER c ON s.customer_id = c.customer_id GROUP BY c.customer_name "
     "ORDER BY total_revenue DESC LIMIT 5"),
    (re.compile(r"revenue by (product )?category|category", re.I),
     "SELECT p.category, SUM(s.revenue) AS total_revenue FROM FACT_SALES s "
     "JOIN DIM_PRODUCT p ON s.product_id = p.product_id GROUP BY p.category "
     "ORDER BY total_revenue DESC"),
    (re.compile(r"revenue by region|by region", re.I),
     "SELECT r.region_name, SUM(s.revenue) AS total_revenue FROM FACT_SALES s "
     "JOIN DIM_REGION r ON s.region_id = r.region_id GROUP BY r.region_name "
     "ORDER BY total_revenue DESC"),
    (re.compile(r"revenue by year", re.I),
     "SELECT d.year, SUM(s.revenue) AS total_revenue FROM FACT_SALES s "
     "JOIN DIM_DATE d ON s.date_id = d.date_id GROUP BY d.year ORDER BY d.year"),
    (re.compile(r"on-time delivery|on time delivery", re.I),
     "SELECT AVG(CASE WHEN on_time_flag THEN 1.0 ELSE 0.0 END) * 100 AS on_time_rate FROM FACT_SHIPMENT"),
]


def template_sql(question: str) -> str | None:
    """Best-effort template SQL for common analytics questions."""
    for pat, sql in _TEMPLATE_SQL:
        if pat.search(question):
            return sql
    return None


def baseline_predict(question: str) -> dict:
    """Full heuristic prediction contract for one question."""
    skill, safety_status = route_skill(question)
    needs_clarification = safety_status == "needs_clarification"
    sql = None
    action = "answer"
    if skill == "REFUSE_UNSAFE":
        action = "refuse"
    elif needs_clarification:
        action = "clarify"
    elif safety_status == "out_of_scope":
        action = "answer"
    elif skill in ("SQL_ANALYST", "FINANCE_ANALYST"):
        sql = template_sql(question)
        action = "run_sql" if sql else "answer"
    return {
        "skill": skill,
        "action": action,
        "safety_status": safety_status,
        "needs_clarification": needs_clarification,
        "sql": sql,
    }
