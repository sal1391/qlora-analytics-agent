#!/usr/bin/env python
"""End-to-end pipeline orchestrator.

Runs stages in order. GPU training is skipped gracefully if unavailable, so the
default ``--stage all`` still produces a full offline result set via the
template baseline.

Usage
-----
    python scripts/run_pipeline.py --stage all
    python scripts/run_pipeline.py --stage generate
    python scripts/run_pipeline.py --stage all --smoke
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
PY = sys.executable


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def stage_generate(smoke: bool) -> None:
    if smoke:
        run([PY, str(SCRIPTS / "01_generate_synthetic_db.py"),
             "--sales", "500", "--customers", "60", "--products", "20"])
        run([PY, str(SCRIPTS / "02_generate_instruction_data.py"),
             "--text_to_sql", "80", "--skill_routing", "40",
             "--refuse_unsafe", "20", "--needs_clarification", "15",
             "--insufficient_schema", "10"])
    else:
        run([PY, str(SCRIPTS / "01_generate_synthetic_db.py")])
        run([PY, str(SCRIPTS / "02_generate_instruction_data.py")])


def stage_preprocess() -> None:
    run([PY, str(SCRIPTS / "03_preprocess.py")])


def stage_train(smoke: bool) -> None:
    cmd = [PY, str(SCRIPTS / "04_train_qlora.py")]
    if smoke:
        cmd.append("--smoke")
    code = run(cmd)
    if code != 0:
        print("[pipeline] Training stage skipped/failed (likely no GPU). "
              "Offline evaluation will still run.")


def stage_evaluate(smoke: bool) -> None:
    cmd = [PY, str(SCRIPTS / "05_evaluate.py")]
    if smoke:
        cmd += ["--max_samples", "50"]
    run(cmd)


def stage_plot() -> None:
    run([PY, str(SCRIPTS / "06_plot_results.py")])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QLoRA analytics pipeline")
    parser.add_argument("--stage", default="all",
                        choices=["all", "generate", "preprocess", "train", "evaluate", "plot"])
    parser.add_argument("--smoke", action="store_true",
                        help="Small subset for fast end-to-end testing")
    args = parser.parse_args()

    if args.stage in ("all", "generate"):
        stage_generate(args.smoke)
    if args.stage in ("all", "preprocess"):
        stage_preprocess()
    if args.stage in ("all", "train"):
        stage_train(args.smoke)
    if args.stage in ("all", "evaluate"):
        stage_evaluate(args.smoke)
    if args.stage in ("all", "plot"):
        stage_plot()
    print("\n[pipeline] Complete.")


if __name__ == "__main__":
    main()
