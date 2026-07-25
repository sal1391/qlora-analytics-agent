"""QLoRA Analytics Agent — enterprise skill routing and safe SQL generation.

Fictional company: GlobalTrade Analytics. All data is synthetic.
"""

__version__ = "0.1.0"

SKILLS = [
    "SQL_ANALYST",
    "FINANCE_ANALYST",
    "SALES_INTELLIGENCE",
    "DOCUMENT_SEARCH",
    "GENERAL_QA",
    "NEEDS_CLARIFICATION",
    "REFUSE_UNSAFE",
]

SAFETY_STATUSES = ["safe", "unsafe", "needs_clarification", "out_of_scope"]

TASK_TYPES = [
    "text_to_sql",
    "skill_routing",
    "refuse_unsafe",
    "needs_clarification",
    "insufficient_schema",
]
