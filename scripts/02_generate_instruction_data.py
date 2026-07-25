#!/usr/bin/env python
"""Stage 02: generate the instruction dataset (JSONL) and validate gold SQL.

Every gold SQL is executed against the DuckDB database; records whose SQL fails
are dropped and reported. Adds ``schema_context`` and ``expected_result`` to
each record.

Output: data/raw/instructions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_gen import SEED, build_instruction_records  # noqa: E402
from src.schema import load_schema, render_schema_context  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw"
DB_PATH = RAW_DIR / "enterprise.duckdb"
OUT_PATH = RAW_DIR / "instructions.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate instruction dataset")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--text_to_sql", type=int, default=1000)
    parser.add_argument("--skill_routing", type=int, default=400)
    parser.add_argument("--refuse_unsafe", type=int, default=150)
    parser.add_argument("--needs_clarification", type=int, default=100)
    parser.add_argument("--insufficient_schema", type=int, default=50)
    args = parser.parse_args()

    if not DB_PATH.exists():
        raise SystemExit("[02] DuckDB not found. Run scripts/01_generate_synthetic_db.py first.")

    counts = {
        "text_to_sql": args.text_to_sql,
        "skill_routing": args.skill_routing,
        "refuse_unsafe": args.refuse_unsafe,
        "needs_clarification": args.needs_clarification,
        "insufficient_schema": args.insufficient_schema,
    }
    print(f"[02] Building instruction records (seed={args.seed}): {counts}")
    records = build_instruction_records(counts=counts, seed=args.seed)

    schema = load_schema()
    schema_ctx = render_schema_context(schema, compact=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    kept: list[dict] = []
    dropped = 0
    for rec in records:
        rec["schema_context"] = schema_ctx
        gold = rec.get("gold_sql")
        if gold:
            try:
                rows = con.execute(gold).fetchall()
                # Store a compact expected_result preview for reproducibility.
                rec["expected_result"] = rows[:5]
            except Exception as exc:
                dropped += 1
                print(f"[02]   DROP invalid SQL ({rec['id']}): {str(exc)[:80]}")
                continue
        else:
            rec["expected_result"] = None
        kept.append(rec)
    con.close()

    with OUT_PATH.open("w") as f:
        for rec in kept:
            f.write(json.dumps(rec, default=str) + "\n")

    # Distribution summary.
    from collections import Counter

    by_task = Counter(r["task_type"] for r in kept)
    by_skill = Counter(r["skill"] for r in kept)
    print(f"[02] Kept {len(kept)} records ({dropped} dropped for invalid SQL).")
    print(f"[02] By task_type: {dict(by_task)}")
    print(f"[02] By skill:     {dict(by_skill)}")
    print(f"[02] Wrote -> {OUT_PATH}")


if __name__ == "__main__":
    main()
