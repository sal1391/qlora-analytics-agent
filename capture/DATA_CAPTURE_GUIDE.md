# Data Capture Guide — Replicating Training WITHOUT Sensitive Data

This guide describes exactly what to log from your **production analytics agent
at work** so you can (a) evaluate the QLoRA model on realistic traffic and
(b) build additional training data — **without exposing real customer names,
row-level values, or proprietary schema names in the public paper.**

> Golden rule: capture *structure and behavior*, never *sensitive values*.
> Everything published must survive the Privacy Checklist at the bottom.

---

## 1. Fields to log per production interaction

Log one JSON record per agent turn. Recommended fields (see
`example_capture_template.jsonl` for the exact shape and `capture_schema.sql`
for a warehouse-side logging table):

| Field | Description | Sensitivity handling |
|-------|-------------|----------------------|
| `anonymized_question_template` | User question with PII, quoted literals, and numbers replaced by `<EMAIL>`, `<VALUE>`, `<N>`. Use `src.snowflake_adapter.anonymize_question`. | Strip PII **before** storage |
| `skill_selected` | Skill the agent routed to (e.g. `SQL_ANALYST`) | Safe |
| `skill_correct` | Ground-truth skill from a human reviewer (nullable until labeled) | Safe |
| `sql_generated` | Model's raw SQL | Replace literal values with `?` placeholders |
| `sql_final` | SQL actually executed after guardrails/edits | Same placeholder rule |
| `execution_success` | Boolean — did the SQL run without error | Safe |
| `safety_flags` | List of guardrail hits (e.g. `["write_blocked"]`) | Safe |
| `schema_version` | Version id of the schema context used (e.g. `enterprise_sales_v3`) | Safe (version tag only) |
| `latency_ms` | End-to-end response latency | Safe |
| `user_feedback` | Thumbs up/down or rating | Safe |
| `clarification_needed` | Boolean — did the agent ask a follow-up | Safe |
| `question_hash` | SHA-256 (first 16 chars) of the raw question for dedup | Safe (one-way) |

**Never log:** raw customer names, account numbers, emails, exact monetary
values, employee names, or the raw un-anonymized question.

---

## 2. Export schema metadata ONLY (not row data)

Use the read-only Snowflake adapter. It queries `INFORMATION_SCHEMA` and never
reads table rows:

```python
from src.snowflake_adapter import SnowflakeConnector, SnowflakeConfig

with SnowflakeConnector(SnowflakeConfig()) as sf:
    # Structural metadata only — column names + types + comments.
    schema_json = sf.convert_snowflake_schema_to_training_format(
        schema_id="enterprise_sales_v1",
        tables=["FACT_SALES", "DIM_CUSTOMER", "DIM_PRODUCT"],
    )
```

For the **public paper**, further redact real table/column names if they are
proprietary: map them to neutral names (`FACT_SALES`, `DIM_CUSTOMER`, ...) and
publish only the neutralized schema. Keep the mapping private.

---

## 3. Build synthetic paraphrases from templates

Do **not** publish raw production questions. Instead:

1. Convert each captured question to an `anonymized_question_template`.
2. Group templates by intent (aggregation, ranking, KPI, etc.).
3. Generate **synthetic paraphrases** by re-filling placeholders with
   *fictional* values drawn from the synthetic schema
   (`src.data_gen.generate_dimensions_and_facts`).
4. Validate every synthetic gold SQL by executing it against the synthetic
   DuckDB — exactly as `scripts/02_generate_instruction_data.py` does.

This yields realistic distribution coverage while guaranteeing that no real
value ever appears in the released dataset or paper.

```python
from src.snowflake_adapter import SnowflakeConnector
# interactions: list of {"question": ..., "skill": ..., "sql": ...}
candidates = SnowflakeConnector().capture_training_candidates(
    interactions, schema_id="enterprise_sales_v1",
)  # returns records in the SAME JSONL schema as synthetic data
```

---

## 4. Labeling workflow

1. Sample ~200–500 anonymized templates stratified by `skill_selected`.
2. Have a reviewer set `skill_correct` and (for SQL tasks) a corrected
   `sql_final` template.
3. Feed labeled templates back through the synthetic paraphrase step to expand.

---

## 5. Privacy Checklist (must pass before anything leaves your environment)

- [ ] No raw customer / account / person names anywhere.
- [ ] No emails, phone numbers, addresses, or IDs (regex + manual spot check).
- [ ] All numeric literals in questions replaced with `<N>` / `<VALUE>`.
- [ ] All SQL literals replaced with `?` placeholders.
- [ ] Only `INFORMATION_SCHEMA` metadata exported — zero table rows.
- [ ] Proprietary table/column names mapped to neutral names for publication.
- [ ] `schema_version` is a tag, not a data dump.
- [ ] Aggregate metrics reported; no per-record sensitive fields published.
- [ ] Reviewed by a second person before release.

Only data passing every box above may be used in the public paper or shared
dataset.
