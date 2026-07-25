#!/usr/bin/env python
"""Stage 05: evaluate conditions and write metrics + predictions.

Conditions default to all of: template_baseline, base_zero_shot, base_few_shot,
qlora_r8, qlora_r16, qlora_r32. GPU conditions are skipped gracefully if no CUDA
GPU or adapter is present, so the pipeline is always testable offline via
template_baseline.

Output:
    results/metrics.json
    results/predictions_<condition>.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import evaluate  # noqa: E402

ALL_CONDITIONS = [
    "template_baseline",
    "base_zero_shot",
    "base_few_shot",
    "qlora_r8",
    "qlora_r16",
    "qlora_r32",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agent conditions")
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS)
    parser.add_argument("--condition", type=str, default=None,
                        help="Shortcut to evaluate a single condition")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapters_dir", type=str, default="results/adapters")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else args.conditions
    print(f"[05] Evaluating conditions: {conditions}")

    metrics = evaluate(
        conditions=conditions,
        model_name=args.model,
        adapters_dir=args.adapters_dir,
        out_dir=args.out_dir,
        max_samples=args.max_samples,
    )

    print("\n[05] === Summary ===")
    for cond, m in metrics.items():
        if m.get("skipped"):
            print(f"  {cond:18s} SKIPPED ({m['reason']})")
        else:
            print(f"  {cond:18s} routing={m.get('routing_accuracy')} "
                  f"sql_exec={m.get('sql_execution_accuracy')} "
                  f"safety={m.get('safety_refusal_accuracy')} "
                  f"json={m.get('json_validity_rate')}")


if __name__ == "__main__":
    main()
