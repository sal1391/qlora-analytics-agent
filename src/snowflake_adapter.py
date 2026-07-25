"""Production-ready, READ-ONLY Snowflake adapter.

Lets the user connect the trained agent to their real Snowflake warehouse *at
work* to (a) export schema metadata into the training schema-context format and
(b) capture anonymized training candidates. It NEVER executes writes — only
``SELECT`` / ``SHOW`` / ``DESCRIBE``.

Environment variables (never hard-code secrets):
    SNOWFLAKE_ACCOUNT     e.g. xy12345.us-east-1
    SNOWFLAKE_USER
    SNOWFLAKE_PASSWORD    (or use SNOWFLAKE_PRIVATE_KEY_PATH for key-pair auth)
    SNOWFLAKE_PRIVATE_KEY_PATH   path to PEM private key (preferred)
    SNOWFLAKE_WAREHOUSE
    SNOWFLAKE_DATABASE
    SNOWFLAKE_SCHEMA
    SNOWFLAKE_ROLE

The ``snowflake-connector-python`` package is optional; import is deferred.
Install with:  pip install "snowflake-connector-python[pandas]>=3.11.0"
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

from .safety import is_read_only_sql, safety_check

# Only these statement prefixes are ever sent to Snowflake.
_READ_PREFIXES = ("SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN")


@dataclass
class SnowflakeConfig:
    account: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_ACCOUNT", ""))
    user: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_USER", ""))
    password: str | None = field(default_factory=lambda: os.getenv("SNOWFLAKE_PASSWORD"))
    private_key_path: str | None = field(
        default_factory=lambda: os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    )
    warehouse: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_WAREHOUSE", ""))
    database: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_DATABASE", ""))
    schema: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_SCHEMA", ""))
    role: str = field(default_factory=lambda: os.getenv("SNOWFLAKE_ROLE", ""))


class SnowflakeConnector:
    """Read-only Snowflake client for schema export and safe dry-runs."""

    def __init__(self, config: SnowflakeConfig | None = None) -> None:
        self.config = config or SnowflakeConfig()
        self._conn = None

    # -- connection lifecycle ------------------------------------------------
    def connect(self):
        """Open a Snowflake connection using password or key-pair auth."""
        try:
            import snowflake.connector as sf
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "snowflake-connector-python is not installed. Install with: "
                'pip install "snowflake-connector-python[pandas]>=3.11.0"'
            ) from exc

        kwargs = dict(
            account=self.config.account,
            user=self.config.user,
            warehouse=self.config.warehouse,
            database=self.config.database,
            schema=self.config.schema,
            role=self.config.role,
        )
        if self.config.private_key_path:
            kwargs["private_key"] = self._load_private_key(self.config.private_key_path)
        elif self.config.password:
            kwargs["password"] = self.config.password
        else:
            raise RuntimeError(
                "No credentials found. Set SNOWFLAKE_PASSWORD or "
                "SNOWFLAKE_PRIVATE_KEY_PATH."
            )
        self._conn = sf.connect(**kwargs)
        return self._conn

    @staticmethod
    def _load_private_key(path: str) -> bytes:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import serialization

        with open(path, "rb") as f:
            p_key = serialization.load_pem_private_key(
                f.read(),
                password=os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "").encode() or None,
                backend=default_backend(),
            )
        return p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # -- safe execution ------------------------------------------------------
    def _assert_read_only(self, sql: str) -> None:
        upper = sql.strip().upper()
        if not upper.startswith(_READ_PREFIXES):
            raise PermissionError(f"Refusing non-read statement: {sql[:60]}...")
        if not is_read_only_sql(sql) and not upper.startswith(("SHOW", "DESCRIBE", "DESC")):
            raise PermissionError(f"SQL failed read-only guardrail: {sql[:60]}...")

    def _fetch(self, sql: str) -> list[tuple]:
        self._assert_read_only(sql)
        if self._conn is None:
            self.connect()
        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            return cur.fetchall()
        finally:
            cur.close()

    # -- schema metadata -----------------------------------------------------
    def get_schema_metadata(
        self, database: str | None = None, schema: str | None = None,
        tables: list[str] | None = None,
    ) -> dict:
        """Return table/column metadata via INFORMATION_SCHEMA (read-only).

        No row-level data is read — only structural metadata.
        """
        database = database or self.config.database
        schema = schema or self.config.schema
        table_filter = ""
        if tables:
            names = ", ".join(f"'{t.upper()}'" for t in tables)
            table_filter = f"AND TABLE_NAME IN ({names})"
        sql = (
            "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COMMENT "
            f"FROM {database}.INFORMATION_SCHEMA.COLUMNS "
            f"WHERE TABLE_SCHEMA = '{schema.upper()}' {table_filter} "
            "ORDER BY TABLE_NAME, ORDINAL_POSITION"
        )
        rows = self._fetch(sql)
        meta: dict[str, dict] = {}
        for table_name, col, dtype, nullable, comment in rows:
            meta.setdefault(table_name, {"columns": []})
            meta[table_name]["columns"].append(
                {"name": col, "type": dtype, "nullable": nullable == "YES",
                 "description": comment or ""}
            )
        return {"database": database, "schema": schema, "tables": meta}

    def export_schema_context(
        self, database: str | None = None, schema: str | None = None,
        tables: list[str] | None = None, compact: bool = True,
    ) -> str:
        """Render a schema-context string in the same format used for training."""
        meta = self.get_schema_metadata(database, schema, tables)
        lines: list[str] = []
        for table_name, info in meta["tables"].items():
            if compact:
                cols = ", ".join(f"{c['name']} {c['type']}" for c in info["columns"])
                lines.append(f"{table_name}({cols})")
            else:
                lines.append(f"-- {table_name}")
                for c in info["columns"]:
                    lines.append(f"  {c['name']} {c['type']}  -- {c['description']}")
        return "\n".join(lines)

    def convert_snowflake_schema_to_training_format(
        self, schema_id: str, database: str | None = None, schema: str | None = None,
        tables: list[str] | None = None,
    ) -> dict:
        """Produce a schema JSON matching ``data/schema/*.json`` structure."""
        meta = self.get_schema_metadata(database, schema, tables)
        out_tables = []
        for table_name, info in meta["tables"].items():
            out_tables.append(
                {
                    "name": table_name,
                    "type": "fact" if table_name.upper().startswith("FACT") else "dimension",
                    "primary_key": info["columns"][0]["name"] if info["columns"] else "",
                    "description": "",
                    "columns": [
                        {"name": c["name"], "type": c["type"], "description": c["description"]}
                        for c in info["columns"]
                    ],
                }
            )
        return {
            "schema_id": schema_id,
            "company": "REDACTED",
            "description": "Schema metadata exported from Snowflake (structure only).",
            "dialect": "snowflake",
            "tables": out_tables,
        }

    def dry_run_sql(self, sql: str) -> dict:
        """Validate SQL against guardrails and Snowflake's planner WITHOUT running it.

        Uses ``EXPLAIN`` so no data is scanned/returned. Returns a report dict.
        """
        report = safety_check(sql)
        if not report["safe"]:
            report["dry_run"] = "blocked_by_guardrail"
            return report
        try:
            self._fetch(f"EXPLAIN {sql}")
            report["dry_run"] = "valid"
        except Exception as exc:  # planner rejected it
            report["dry_run"] = "invalid"
            report["error"] = str(exc)[:300]
        return report

    # -- training candidate capture -----------------------------------------
    def capture_training_candidates(
        self, interactions: list[dict], schema_id: str,
        schema_context: str | None = None,
    ) -> list[dict]:
        """Format captured production interactions into the training JSONL schema.

        Each ``interaction`` should contain at minimum ``question`` and may
        contain ``skill``, ``safety_status``, ``sql``. Questions are anonymized
        (numbers and quoted literals templated) before formatting. NO row data
        and NO raw PII is retained.
        """
        schema_context = schema_context or self.export_schema_context()
        out: list[dict] = []
        for i, item in enumerate(interactions):
            q_template = anonymize_question(item["question"])
            out.append(
                {
                    "id": f"{schema_id}-capture-{i:05d}",
                    "task_type": item.get("task_type", "text_to_sql"),
                    "schema_id": schema_id,
                    "question": q_template,
                    "schema_context": schema_context,
                    "skill": item.get("skill", "SQL_ANALYST"),
                    "safety_status": item.get("safety_status", "safe"),
                    "gold_sql": item.get("sql"),
                    "complexity_tags": item.get("complexity_tags", []),
                    "source": "production_capture",
                    "question_hash": hashlib.sha256(item["question"].encode()).hexdigest()[:16],
                }
            )
        return out


# ---- Anonymization helpers -------------------------------------------------

_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_NUM_RE = re.compile(r"\b\d[\d,\.]*\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def anonymize_question(question: str) -> str:
    """Strip PII-ish tokens and template literals for a public-safe question.

    Replaces emails, quoted literals and numeric values with placeholders so a
    question can be published without exposing real values.
    """
    q = _EMAIL_RE.sub("<EMAIL>", question)
    q = _QUOTED_RE.sub("<VALUE>", q)
    q = _NUM_RE.sub("<N>", q)
    return q.strip()
