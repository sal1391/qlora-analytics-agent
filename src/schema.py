"""Schema loading and rendering utilities.

Loads the synthetic ``enterprise_sales_v1`` star schema and the business
data dictionary, and renders compact schema context strings that are fed to
the model as part of the prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Repository root (two levels up from this file: src/schema.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = REPO_ROOT / "data" / "schema"
DEFAULT_SCHEMA_ID = "enterprise_sales_v1"


@dataclass
class Column:
    name: str
    type: str
    description: str = ""


@dataclass
class Table:
    name: str
    type: str
    primary_key: str
    description: str
    columns: list[Column] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


@dataclass
class Schema:
    schema_id: str
    company: str
    description: str
    dialect: str
    tables: list[Table]

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def get_table(self, name: str) -> Table | None:
        for t in self.tables:
            if t.name.upper() == name.upper():
                return t
        return None

    def all_columns(self) -> set[str]:
        cols: set[str] = set()
        for t in self.tables:
            cols.update(c.name.lower() for c in t.columns)
        return cols


def _schema_path(schema_id: str) -> Path:
    return SCHEMA_DIR / f"{schema_id}.json"


@lru_cache(maxsize=8)
def load_schema(schema_id: str = DEFAULT_SCHEMA_ID) -> Schema:
    """Load a schema definition from ``data/schema/<schema_id>.json``."""
    data = json.loads(_schema_path(schema_id).read_text())
    tables = [
        Table(
            name=t["name"],
            type=t["type"],
            primary_key=t["primary_key"],
            description=t["description"],
            columns=[Column(**c) for c in t["columns"]],
        )
        for t in data["tables"]
    ]
    return Schema(
        schema_id=data["schema_id"],
        company=data["company"],
        description=data["description"],
        dialect=data["dialect"],
        tables=tables,
    )


@lru_cache(maxsize=2)
def load_data_dictionary() -> dict:
    """Load business rules, metric definitions and synonyms."""
    return json.loads((SCHEMA_DIR / "data_dictionary.json").read_text())


def render_schema_context(schema: Schema, compact: bool = True) -> str:
    """Render a DDL-like schema string for prompting.

    Parameters
    ----------
    schema:
        The loaded :class:`Schema`.
    compact:
        If True, render one line per table (``TABLE(col type, ...)``).
        Otherwise render multi-line blocks with descriptions.
    """
    lines: list[str] = []
    if compact:
        for t in schema.tables:
            cols = ", ".join(f"{c.name} {c.type}" for c in t.columns)
            lines.append(f"{t.name}({cols})")
        return "\n".join(lines)

    for t in schema.tables:
        lines.append(f"-- {t.name} [{t.type}]: {t.description}")
        for c in t.columns:
            lines.append(f"  {c.name} {c.type}  -- {c.description}")
        lines.append("")
    return "\n".join(lines).strip()


def render_business_rules(dictionary: dict | None = None, max_metrics: int = 8) -> str:
    """Render business rules and metric formulas for prompting."""
    d = dictionary or load_data_dictionary()
    parts: list[str] = ["Business rules:"]
    parts += [f"- {r}" for r in d.get("business_rules", [])]
    parts.append("Metric definitions:")
    for i, (name, meta) in enumerate(d.get("metrics", {}).items()):
        if i >= max_metrics:
            break
        parts.append(f"- {name}: {meta['definition']} ({meta['formula']})")
    return "\n".join(parts)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    s = load_schema()
    print(f"Loaded {s.schema_id} for {s.company} with {len(s.tables)} tables")
    print(render_schema_context(s))
