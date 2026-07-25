#!/usr/bin/env python
"""Stage 03: preprocess raw instructions into SFT chat splits.

Reads data/raw/instructions.jsonl, normalizes/dedups/validates, formats to chat
messages, and writes stratified splits.

Output:
    data/processed/train.jsonl
    data/processed/val.jsonl
    data/processed/test.jsonl
    data/processed/stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocess import preprocess_records  # noqa: E402

RAW_PATH = REPO_ROOT / "data" / "raw" / "instructions.jsonl"
PROC_DIR = REPO_ROOT / "data" / "processed"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess instruction data")
    parser.add_argument("--input", type=str, default=str(RAW_PATH))
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit("[03] instructions.jsonl not found. Run scripts/02 first.")

    raw = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    print(f"[03] Loaded {len(raw)} raw records.")

    train, val, test, stats = preprocess_records(raw)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(PROC_DIR / "train.jsonl", train)
    _write_jsonl(PROC_DIR / "val.jsonl", val)
    _write_jsonl(PROC_DIR / "test.jsonl", test)
    (PROC_DIR / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"[03] Splits -> train={len(train)} val={len(val)} test={len(test)}")
    print(f"[03] Dedup: {stats['raw_count']} -> {stats['after_dedup']} "
          f"-> validated {stats['after_validation']}")
    print(f"[03] Wrote processed splits + stats.json -> {PROC_DIR}")


if __name__ == "__main__":
    main()
