# QLoRA Analytics Agent

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22117133.svg)](https://doi.org/10.5281/zenodo.22117133)

**QLoRA fine-tuning of open LLMs for enterprise analytics-agent skill routing and
safe SQL generation.**

Paper: [Efficient Domain Adaptation of Open Language Models for Enterprise
Analytics Agents Using LoRA and QLoRA](https://doi.org/10.5281/zenodo.22117133)

This is a self-contained, reproducible research project. It fine-tunes
**Qwen2.5-3B-Instruct** with **QLoRA (4-bit NF4)** to act as an enterprise BI
analytics agent that:

1. **Routes** a natural-language request to the right *skill*
   (`SQL_ANALYST`, `FINANCE_ANALYST`, `SALES_INTELLIGENCE`, `DOCUMENT_SEARCH`,
   `GENERAL_QA`, `NEEDS_CLARIFICATION`, `REFUSE_UNSAFE`).
2. **Generates read-only SQL** over a star-schema warehouse.
3. **Enforces guardrails** — refuses writes (INSERT/UPDATE/DELETE/DROP/…),
   off-topic, and unsafe requests.
4. **Emits a structured JSON contract** for downstream agent orchestration.

The reference schema is a *fictional* company, **GlobalTrade Analytics**. All
data is synthetic — no proprietary or real customer data is used anywhere.

Architecture pattern mirrored:
`orchestrator → skill router → SQL generation → guardrails → warehouse`.

---

## Repository layout

```
qlora-analytics-agent/
  configs/       training + evaluation YAML configs
  data/
    schema/      synthetic star-schema + business data dictionary
    raw/         generated DuckDB + CSVs + instructions.jsonl
    processed/   train/val/test SFT splits + stats.json
  scripts/       01..06 pipeline stages + run_pipeline.py
  src/           library modules (schema, safety, data_gen, preprocess,
                 metrics, skill_router, train, evaluate, inference,
                 snowflake_adapter)
  capture/       guide + SQL + template for capturing real data safely
  notebooks/     Colab + local RTX 5090 training notebooks
  results/       metrics.json, predictions_*.jsonl, adapters/, train_metrics.json
  figures/       publication-quality PNGs (300 dpi)
  tests/         unit tests (run without a GPU)
```

---

## Quick start (offline, no GPU)

The full pipeline runs offline using a rule-based `template_baseline` so you can
verify everything before touching a GPU:

```bash
pip install -r requirements.txt   # or: conda env create -f environment.yml
python scripts/run_pipeline.py --stage all     # train stage auto-skips if no GPU
pytest -q
```

This produces `results/metrics.json`, prediction logs, and figures.

---

## A) Local RTX 5090 setup (primary target, 32GB VRAM)

```bash
conda env create -f environment.yml
conda activate qlora-analytics-agent

# 1) Data
python scripts/01_generate_synthetic_db.py
python scripts/02_generate_instruction_data.py
python scripts/03_preprocess.py

# 2) Train QLoRA (auto-detects CUDA, picks batch size for 32GB)
python scripts/04_train_qlora.py --config configs/train_qlora_3b.yaml --lora_r 16

# 3) Ablations
python scripts/04_train_qlora.py --lora_r 8
python scripts/04_train_qlora.py --lora_r 32

# 4) Evaluate + plot
python scripts/05_evaluate.py
python scripts/06_plot_results.py
```

Batch size auto-adapts: 32GB → `bs=4, grad_accum=4`. Gradient checkpointing +
sequence packing keep memory well within 32GB for the 3B model in 4-bit.

**Smoke test** (fast end-to-end):
```bash
python scripts/run_pipeline.py --stage all --smoke
```

---

## B) Google Colab setup (fallback, < $100)

Open `notebooks/Colab_QLoRA_Training.ipynb` in Colab and run top to bottom:
install → upload/generate data → train → evaluate → download adapters.
Use `configs/train_qlora_colab.yaml` (2 epochs, T4/A100-friendly). On an A100 a
2–3 epoch 3B QLoRA run finishes in ~15–40 min — a few dollars of compute units.
On a free/low-tier T4 it is slower but still under budget.

---

## C) Connecting Snowflake at work

`src/snowflake_adapter.py` is a **read-only** production adapter. It exports
schema metadata and captures anonymized training candidates — it never runs
writes (only `SELECT` / `SHOW` / `DESCRIBE` / `EXPLAIN`).

Set env vars (never hard-code secrets):

```bash
export SNOWFLAKE_ACCOUNT=xy12345.us-east-1
export SNOWFLAKE_USER=...
export SNOWFLAKE_PASSWORD=...          # or SNOWFLAKE_PRIVATE_KEY_PATH=...
export SNOWFLAKE_WAREHOUSE=...
export SNOWFLAKE_DATABASE=...
export SNOWFLAKE_SCHEMA=...
export SNOWFLAKE_ROLE=...
pip install "snowflake-connector-python[pandas]>=3.11.0"
```

```python
from src.snowflake_adapter import SnowflakeConnector, SnowflakeConfig
with SnowflakeConnector(SnowflakeConfig()) as sf:
    ctx = sf.export_schema_context(tables=["FACT_SALES", "DIM_CUSTOMER"])
    report = sf.dry_run_sql("SELECT SUM(revenue) FROM FACT_SALES")  # EXPLAIN only
```

See `capture/DATA_CAPTURE_GUIDE.md` for exactly what to log and how to stay
privacy-safe (no PII, metadata-only schema export, synthetic paraphrasing).

---

## D) Reproducing paper results

Everything is seeded (`seed=42`). To reproduce:

```bash
python scripts/01_generate_synthetic_db.py     # deterministic DB + CSVs
python scripts/02_generate_instruction_data.py # gold SQL validated on DuckDB
python scripts/03_preprocess.py                # stratified 70/15/15 splits
python scripts/04_train_qlora.py --lora_r 8
python scripts/04_train_qlora.py --lora_r 16
python scripts/04_train_qlora.py --lora_r 32
python scripts/05_evaluate.py                  # all conditions
python scripts/06_plot_results.py              # figures @ 300 dpi
```

**Conditions compared:** `base_zero_shot`, `base_few_shot` (3-shot),
`qlora_r8/16/32`, plus a GPU-free `template_baseline`.
**Metrics:** routing accuracy, SQL execution accuracy (result comparison on
DuckDB), exact SQL match (normalized), safety refusal accuracy, JSON validity,
mean latency, trainable params, peak VRAM.

---

## Results (SEAAD test set, n=166; 60 gold-SQL cases)

| Condition         | Routing | SQL exec. | Safety | JSON |
|-------------------|--------:|----------:|-------:|-----:|
| Template baseline |  92.8%  |   23.3%   |  100%  | 100% |
| Base zero-shot    |   0.0%  |   45.0%   |  56.5% | 98.8%|
| Base 3-shot       |  36.8%  |   56.7%   |  100%  | 80.1%|
| QLoRA r=8         |  80.1%  |   73.3%   |  100%  | 100% |
| QLoRA r=16        |  92.2%  |   86.7%   |  100%  | 100% |
| **QLoRA r=32**    | **95.8%** | **100%** | **100%** | **100%** |

Trained on a single RTX 5090 laptop GPU (24 GB), ~5.1 GB peak VRAM, ~30 min
per rank. Gold assistant turns are stripped from evaluation prompts
(regression-tested in `tests/test_eval_no_leakage.py`).

---

## E) Budget note

- **Local RTX 5090:** free (your own hardware). Full 3B QLoRA at 2–3 epochs fits
  in 32GB with 4-bit NF4 + gradient checkpointing + packing.
- **Colab A100:** a full 2–3 epoch run is typically **a few dollars** of compute
  units — comfortably **well under $100**, even including all three rank
  ablations. Colab T4 works too (slower) and remains under budget.

---

## Safety contract

The agent must return a single JSON object:

```json
{"skill": "SQL_ANALYST", "action": "run_sql", "safety_status": "safe",
 "needs_clarification": false, "sql": "SELECT SUM(revenue) FROM FACT_SALES"}
```

`src/safety.py` enforces read-only SQL at inference time regardless of model
output, blocking `INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/MERGE/CREATE/GRANT`,
stacked statements, and unknown tables.

---

*The `latex/` directory contains the IEEE-format paper source (`main.tex`)
and figures.*

---

## Citation

```bibtex
@misc{salgado2026seaad,
  author       = {Salgado, Carlos},
  title        = {Efficient Domain Adaptation of Open Language Models for
                  Enterprise Analytics Agents Using LoRA and QLoRA},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22117133},
  url          = {https://doi.org/10.5281/zenodo.22117133}
}
```

Salgado, C. (2026). *Efficient Domain Adaptation of Open Language Models for
Enterprise Analytics Agents Using LoRA and QLoRA.* Zenodo.
https://doi.org/10.5281/zenodo.22117133

## License

Code is MIT licensed (see [LICENSE](LICENSE)). The paper and Zenodo record are
released under CC-BY-4.0.
