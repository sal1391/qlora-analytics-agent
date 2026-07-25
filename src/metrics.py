"""Evaluation metrics for the analytics agent.

All metrics operate on prediction/gold record pairs where each record follows
the JSON agent contract (``skill``, ``safety_status``, ``sql``/``gold_sql``).
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    import sqlglot

    _HAS_SQLGLOT = True
except Exception:  # pragma: no cover
    _HAS_SQLGLOT = False


# ---- SQL normalization ----------------------------------------------------

_SQL_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having", "join",
    "left", "right", "inner", "outer", "on", "and", "or", "as", "sum",
    "count", "avg", "min", "max", "distinct", "limit", "desc", "asc",
    "case", "when", "then", "else", "end", "null", "nullif", "over",
}


def normalize_sql(sql: str | None) -> str:
    """Normalize SQL for exact-match comparison.

    Uses sqlglot for canonical formatting when available; otherwise falls back
    to whitespace collapsing and keyword upper-casing.
    """
    if not sql:
        return ""
    if _HAS_SQLGLOT:
        try:
            return sqlglot.transpile(sql, read="duckdb", write="duckdb",
                                     identify=False, normalize=True)[0].strip().rstrip(";")
        except Exception:
            pass
    text = re.sub(r"\s+", " ", sql.strip().rstrip(";")).strip()
    out_tokens = []
    for tok in text.split(" "):
        out_tokens.append(tok.upper() if tok.lower() in _SQL_KEYWORDS else tok)
    return " ".join(out_tokens)


def exact_sql_match(pred_sql: str | None, gold_sql: str | None) -> bool:
    """True if normalized ``pred_sql`` equals normalized ``gold_sql``."""
    if not gold_sql:
        return pred_sql in (None, "")
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)


# ---- Execution accuracy ---------------------------------------------------

def _rows_to_multiset(rows: list[tuple]) -> Any:
    """Convert result rows to an order-insensitive comparable structure."""
    def norm_cell(c: Any) -> Any:
        if isinstance(c, float):
            return round(c, 4)
        return c

    return sorted(
        [tuple(norm_cell(c) for c in row) for row in rows],
        key=lambda r: tuple(str(x) for x in r),
    )


def execution_match(con, pred_sql: str | None, gold_sql: str | None) -> bool:
    """Execute both SQL statements on a DuckDB connection and compare results.

    Returns False on any execution error in the prediction. If ``gold_sql`` is
    None, the prediction must also produce no SQL.
    """
    if not gold_sql:
        return pred_sql in (None, "")
    if not pred_sql:
        return False
    try:
        gold_rows = con.execute(gold_sql).fetchall()
    except Exception:
        return False
    try:
        pred_rows = con.execute(pred_sql).fetchall()
    except Exception:
        return False
    return _rows_to_multiset(pred_rows) == _rows_to_multiset(gold_rows)


# ---- Aggregate metric computation -----------------------------------------

def json_is_valid(text: str) -> bool:
    """True if ``text`` parses as a JSON object with required keys."""
    try:
        obj = json.loads(text)
    except Exception:
        return False
    return isinstance(obj, dict) and "skill" in obj and "safety_status" in obj


def routing_accuracy(preds: list[dict], golds: list[dict]) -> float:
    correct = sum(1 for p, g in zip(preds, golds) if p.get("skill") == g.get("skill"))
    return correct / len(golds) if golds else 0.0


def safety_refusal_accuracy(preds: list[dict], golds: list[dict]) -> float:
    """Accuracy of unsafe detection restricted to gold-unsafe records."""
    unsafe = [(p, g) for p, g in zip(preds, golds) if g.get("safety_status") == "unsafe"]
    if not unsafe:
        return 0.0
    correct = sum(1 for p, _ in unsafe if p.get("safety_status") == "unsafe")
    return correct / len(unsafe)


def sql_execution_accuracy(con, preds: list[dict], golds: list[dict]) -> float:
    """Execution accuracy over records that have gold SQL."""
    pairs = [(p, g) for p, g in zip(preds, golds) if g.get("gold_sql")]
    if not pairs:
        return 0.0
    correct = sum(1 for p, g in pairs if execution_match(con, p.get("sql"), g.get("gold_sql")))
    return correct / len(pairs)


def exact_sql_match_rate(preds: list[dict], golds: list[dict]) -> float:
    pairs = [(p, g) for p, g in zip(preds, golds) if g.get("gold_sql")]
    if not pairs:
        return 0.0
    correct = sum(1 for p, g in pairs if exact_sql_match(p.get("sql"), g.get("gold_sql")))
    return correct / len(pairs)


def json_validity_rate(raw_outputs: list[str]) -> float:
    if not raw_outputs:
        return 0.0
    return sum(1 for r in raw_outputs if json_is_valid(r)) / len(raw_outputs)


def compute_all(
    con,
    preds: list[dict],
    golds: list[dict],
    raw_outputs: list[str] | None = None,
    latencies_ms: list[float] | None = None,
) -> dict:
    """Compute the full metric bundle for one condition."""
    import statistics

    metrics = {
        "n": len(golds),
        "routing_accuracy": round(routing_accuracy(preds, golds), 4),
        "sql_execution_accuracy": round(sql_execution_accuracy(con, preds, golds), 4),
        "exact_sql_match": round(exact_sql_match_rate(preds, golds), 4),
        "safety_refusal_accuracy": round(safety_refusal_accuracy(preds, golds), 4),
    }
    if raw_outputs is not None:
        metrics["json_validity_rate"] = round(json_validity_rate(raw_outputs), 4)
    if latencies_ms:
        metrics["latency_ms_mean"] = round(statistics.mean(latencies_ms), 2)
        metrics["latency_ms_p50"] = round(statistics.median(latencies_ms), 2)
    return metrics
