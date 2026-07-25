"""Unit tests for preprocessing (no GPU required)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import (
    build_assistant_target,
    deduplicate,
    preprocess_records,
    stratified_split,
    to_chat_sample,
)
from src.schema import load_data_dictionary, load_schema, render_business_rules


def _raw(task_type, question, skill, safety, gold_sql=None, idx=0):
    return {
        "id": f"enterprise_sales_v1-{idx:05d}",
        "task_type": task_type,
        "schema_id": "enterprise_sales_v1",
        "question": question,
        "skill": skill,
        "safety_status": safety,
        "gold_sql": gold_sql,
        "complexity_tags": [],
    }


def test_assistant_target_actions():
    t = build_assistant_target(_raw("refuse_unsafe", "Delete all", "REFUSE_UNSAFE", "unsafe"))
    assert t["action"] == "refuse" and t["sql"] is None

    t = build_assistant_target(_raw("needs_clarification", "How are we?", "NEEDS_CLARIFICATION",
                                    "needs_clarification"))
    assert t["action"] == "clarify" and t["needs_clarification"] is True

    t = build_assistant_target(_raw("text_to_sql", "Total revenue?", "SQL_ANALYST", "safe",
                                    "SELECT SUM(revenue) FROM FACT_SALES"))
    assert t["action"] == "run_sql" and t["sql"]


def test_chat_sample_structure():
    schema = load_schema()
    rules = render_business_rules(load_data_dictionary())
    sample = to_chat_sample(
        _raw("text_to_sql", "Total revenue?", "SQL_ANALYST", "safe",
             "SELECT SUM(revenue) FROM FACT_SALES"),
        schema, rules,
    )
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["system", "user", "assistant"]
    # Assistant content must be valid JSON with the contract keys.
    obj = json.loads(sample["messages"][2]["content"])
    assert set(["skill", "action", "safety_status", "needs_clarification", "sql"]).issubset(obj)
    assert "Question:" in sample["messages"][1]["content"]


def test_deduplicate():
    recs = [
        _raw("text_to_sql", "Total revenue?", "SQL_ANALYST", "safe", "SELECT 1", 0),
        _raw("text_to_sql", "Total revenue?", "SQL_ANALYST", "safe", "SELECT 1", 1),
        _raw("text_to_sql", "Units sold?", "SQL_ANALYST", "safe", "SELECT 2", 2),
    ]
    out = deduplicate(recs)
    assert len(out) == 2


def test_stratified_split_proportions():
    recs = []
    idx = 0
    for _ in range(50):
        recs.append(_raw("text_to_sql", f"q{idx}", "SQL_ANALYST", "safe", "SELECT 1", idx)); idx += 1
    for _ in range(20):
        recs.append(_raw("refuse_unsafe", f"del{idx}", "REFUSE_UNSAFE", "unsafe", None, idx)); idx += 1
    # Make chat samples (split operates on any dict with task_type).
    train, val, test = stratified_split(recs, seed=1)
    total = len(train) + len(val) + len(test)
    assert total == len(recs)
    # Each split should contain both task types.
    assert any(r["task_type"] == "refuse_unsafe" for r in train)


def test_full_preprocess_smoke():
    raw = []
    idx = 0
    for _ in range(20):
        raw.append(_raw("text_to_sql", f"Total revenue {idx}?", "SQL_ANALYST", "safe",
                        "SELECT SUM(revenue) FROM FACT_SALES", idx)); idx += 1
    for _ in range(10):
        raw.append(_raw("refuse_unsafe", f"Delete {idx}", "REFUSE_UNSAFE", "unsafe", None, idx)); idx += 1
    train, val, test, stats = preprocess_records(raw)
    assert stats["splits"]["train"] > 0
    assert stats["after_validation"] <= stats["raw_count"]
    # Invalid-table gold SQL should be dropped.
    bad = _raw("text_to_sql", "bad", "SQL_ANALYST", "safe", "SELECT * FROM NOPE_TABLE", 999)
    _, _, _, stats2 = preprocess_records([bad])
    assert stats2["after_validation"] == 0
