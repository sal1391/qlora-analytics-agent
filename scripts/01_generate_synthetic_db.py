#!/usr/bin/env python
"""Stage 01: generate the synthetic GlobalTrade Analytics DuckDB + CSVs.

Deterministic (seed=42). Produces:
    data/raw/enterprise.duckdb
    data/raw/*.csv  (one per table)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_gen import SEED, generate_dimensions_and_facts  # noqa: E402

RAW_DIR = REPO_ROOT / "data" / "raw"
DB_PATH = RAW_DIR / "enterprise.duckdb"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic enterprise DB")
    parser.add_argument("--customers", type=int, default=200)
    parser.add_argument("--products", type=int, default=50)
    parser.add_argument("--sales", type=int, default=3000)
    parser.add_argument("--months", type=int, default=36)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[01] Generating synthetic data (seed={args.seed})...")
    tables = generate_dimensions_and_facts(
        n_customers=args.customers,
        n_products=args.products,
        n_sales=args.sales,
        n_months=args.months,
        seed=args.seed,
    )

    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    for name, df in tables.items():
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
        csv_path = RAW_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[01]   {name:16s} rows={len(df):6d}  -> {csv_path.name}")

    # Sanity check: total revenue.
    total_rev = con.execute("SELECT SUM(revenue) FROM FACT_SALES").fetchone()[0]
    print(f"[01] FACT_SALES total revenue = {total_rev:,.2f}")
    con.close()
    print(f"[01] Wrote DuckDB database -> {DB_PATH}")


if __name__ == "__main__":
    main()
