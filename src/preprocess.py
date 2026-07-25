"""Preprocessing: normalize, dedup, validate, format to chat, split.

Converts raw instruction records into SFT-ready chat message samples with the
JSON agent contract as the assistant target.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict

from .metrics import normalize_sql
from .schema import (
    Schema,
    load_data_dictionary,
    load_schema,
    render_business_rules,
    render_schema_context,
)

SYSTEM_PROMPT = (
    "You are an enterprise analytics routing and SQL assistant for GlobalTrade "
    "Analytics. Classify each request to a skill, decide the action, and enforce "
    "a read-only contract. You must respond with a SINGLE JSON object only, no "
    "prose. Never generate write operations (INSERT/UPDATE/DELETE/DROP/ALTER/"
    "TRUNCATE/MERGE/CREATE/GRANT). If a request is a data write, refuse. If it is "
    "ambiguous, ask for clarification. If it is out of scope for the schema, say so. "
    "JSON schema: {\"skill\": str, \"action\": str, \"safety_status\": str, "
    "\"needs_clarification\": bool, \"sql\": str|null}."
)


def build_assistant_target(rec: dict) -> dict:
    """Construct the JSON assistant target for a raw record."""
    task = rec["task_type"]
    skill = rec["skill"]
    safety = rec["safety_status"]
    sql = rec.get("gold_sql")

    if task == "refuse_unsafe":
        action = "refuse"
    elif task == "needs_clarification":
        action = "clarify"
    elif task == "insufficient_schema":
        action = "answer"
    elif task == "skill_routing":
        action = "route"
    else:  # text_to_sql
        action = "run_sql" if sql else "answer"

    return {
        "skill": skill,
        "action": action,
        "safety_status": safety,
        "needs_clarification": task == "needs_clarification",
        "sql": normalize_sql(sql) if sql else None,
    }


def build_user_prompt(rec: dict, schema: Schema, business_rules: str) -> str:
    """Assemble the user turn: schema + rules + question."""
    schema_ctx = render_schema_context(schema, compact=True)
    return (
        f"Schema ({schema.schema_id}):\n{schema_ctx}\n\n"
        f"{business_rules}\n\n"
        f"Question: {rec['question']}"
    )


def to_chat_sample(rec: dict, schema: Schema, business_rules: str) -> dict:
    """Convert a raw record into a chat-format SFT sample."""
    target = build_assistant_target(rec)
    user = build_user_prompt(rec, schema, business_rules)
    return {
        "id": rec["id"],
        "task_type": rec["task_type"],
        "schema_id": rec["schema_id"],
        "skill": rec["skill"],
        "safety_status": rec["safety_status"],
        "gold_sql": target["sql"],
        "complexity_tags": rec.get("complexity_tags", []),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
    }


def _dedup_key(rec: dict) -> str:
    payload = f"{rec['task_type']}|{rec['question'].strip().lower()}|{normalize_sql(rec.get('gold_sql'))}"
    return hashlib.md5(payload.encode()).hexdigest()


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove exact-question+sql duplicates, keeping first occurrence."""
    seen: set[str] = set()
    out: list[dict] = []
    for rec in records:
        k = _dedup_key(rec)
        if k not in seen:
            seen.add(k)
            out.append(rec)
    return out


def validate_schema_references(rec: dict, schema: Schema) -> bool:
    """Ensure any gold SQL references only known tables."""
    sql = rec.get("gold_sql")
    if not sql:
        return True
    refs = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", sql, flags=re.IGNORECASE)
    known = {t.upper() for t in schema.table_names}
    return all(r.upper() in known for r in refs)


def stratified_split(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records stratified by ``task_type`` (schema_id is constant here).

    We stratify by task_type so each split has proportional representation of
    text-to-SQL, routing, refusal, clarification and out-of-scope examples.
    """
    import random

    rnd = random.Random(seed)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_type"]].append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for task, recs in by_task.items():
        recs = recs[:]
        rnd.shuffle(recs)
        n = len(recs)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train += recs[:n_train]
        val += recs[n_train : n_train + n_val]
        test += recs[n_train + n_val :]
    rnd.shuffle(train)
    rnd.shuffle(val)
    rnd.shuffle(test)
    return train, val, test


def compute_stats(records: list[dict]) -> dict:
    """Distribution stats over task_type / skill / safety / complexity."""
    def counter(key):
        c: dict[str, int] = defaultdict(int)
        for r in records:
            c[r.get(key, "?")] += 1
        return dict(c)

    tag_counts: dict[str, int] = defaultdict(int)
    for r in records:
        for t in r.get("complexity_tags", []):
            tag_counts[t] += 1

    return {
        "total": len(records),
        "by_task_type": counter("task_type"),
        "by_skill": counter("skill"),
        "by_safety_status": counter("safety_status"),
        "by_complexity_tag": dict(tag_counts),
    }


def preprocess_records(
    raw_records: list[dict],
    schema_id: str = "enterprise_sales_v1",
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Full pipeline: dedup -> validate -> chat-format -> split -> stats."""
    schema = load_schema(schema_id)
    business_rules = render_business_rules(load_data_dictionary())

    deduped = deduplicate(raw_records)
    valid = [r for r in deduped if validate_schema_references(r, schema)]

    samples = [to_chat_sample(r, schema, business_rules) for r in valid]
    train, val, test = stratified_split(samples)

    stats = {
        "raw_count": len(raw_records),
        "after_dedup": len(deduped),
        "after_validation": len(valid),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "train_distribution": compute_stats(train),
        "val_distribution": compute_stats(val),
        "test_distribution": compute_stats(test),
    }
    return train, val, test, stats
