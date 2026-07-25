"""Synthetic data generation for GlobalTrade Analytics.

Provides:
  * :func:`generate_dimensions_and_facts` — builds pandas DataFrames for the
    star schema, deterministically seeded.
  * :func:`build_instruction_records` — generates instruction-tuning records
    (text-to-SQL, skill routing, refusals, clarifications, out-of-scope) with
    DuckDB-valid gold SQL.

All names/values are fictional. Nothing here references any real employer,
customer, or proprietary table.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

try:  # Faker is optional; fall back to fixed lists.
    from faker import Faker

    _HAS_FAKER = True
except Exception:  # pragma: no cover - exercised only when faker missing
    _HAS_FAKER = False

SEED = 42

SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Consumer"]
INDUSTRIES = [
    "Manufacturing", "Retail", "Technology", "Healthcare",
    "Financial Services", "Logistics", "Energy", "Public Sector",
]
ACCOUNT_TIERS = ["Platinum", "Gold", "Silver", "Bronze"]
CATEGORIES = {
    "Hardware": ["Servers", "Networking", "Storage"],
    "Software": ["Analytics", "Security", "Productivity"],
    "Services": ["Consulting", "Support", "Training"],
}
REGIONS = [
    ("North America - East", "USA", "NA-EAST"),
    ("North America - West", "USA", "NA-WEST"),
    ("Canada", "Canada", "NA-CANADA"),
    ("Latin America", "Brazil", "LATAM"),
    ("EMEA - North", "Germany", "EMEA-N"),
    ("EMEA - South", "Spain", "EMEA-S"),
    ("APAC - North", "Japan", "APAC-N"),
    ("APAC - South", "Australia", "APAC-S"),
]

PRODUCT_ADJ = ["Nova", "Prime", "Apex", "Vertex", "Quantum", "Fusion", "Pulse", "Zenith"]
PRODUCT_NOUN = ["Analyzer", "Gateway", "Cluster", "Suite", "Engine", "Platform", "Console", "Vault"]


def _rng(seed: int = SEED) -> tuple[random.Random, np.random.Generator]:
    return random.Random(seed), np.random.default_rng(seed)


def generate_dimensions_and_facts(
    n_customers: int = 200,
    n_products: int = 50,
    n_sales: int = 3000,
    n_months: int = 36,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Generate all dimension and fact tables as DataFrames.

    Deterministic given ``seed``. Returns a dict keyed by table name.
    """
    rnd, rng = _rng(seed)
    faker = Faker() if _HAS_FAKER else None
    if faker is not None:
        Faker.seed(seed)

    # ---- DIM_REGION ----
    regions = pd.DataFrame(
        [
            {"region_id": i + 1, "region_name": r[0], "country": r[1], "sales_territory": r[2]}
            for i, r in enumerate(REGIONS)
        ]
    )

    # ---- DIM_DATE (day grain, first n_months months back from a fixed anchor) ----
    anchor = date(2025, 12, 1)
    start = date(anchor.year - (n_months // 12), anchor.month, 1)
    dates = []
    d = start
    date_id = 0  # noqa: F841
    day = start
    all_days: list[date] = []
    end = anchor
    while day <= end:
        all_days.append(day)
        day += timedelta(days=1)
    for dd in all_days:
        dates.append(
            {
                "date_id": int(dd.strftime("%Y%m%d")),
                "full_date": dd,
                "year": dd.year,
                "quarter": (dd.month - 1) // 3 + 1,
                "month": dd.month,
                "month_name": dd.strftime("%B"),
                "fiscal_year": dd.year if dd.month >= 2 else dd.year - 1,
            }
        )
    dim_date = pd.DataFrame(dates)
    date_ids = dim_date["date_id"].tolist()

    # ---- DIM_CUSTOMER ----
    customers = []
    for cid in range(1, n_customers + 1):
        if faker is not None:
            name = faker.company()
        else:
            name = f"{rnd.choice(PRODUCT_ADJ)} {rnd.choice(['Corp', 'Ltd', 'Group', 'Holdings', 'Systems'])} {cid}"
        signup_year = rnd.randint(2018, 2024)
        signup = date(signup_year, rnd.randint(1, 12), rnd.randint(1, 28))
        customers.append(
            {
                "customer_id": cid,
                "customer_name": name,
                "segment": rnd.choice(SEGMENTS),
                "industry": rnd.choice(INDUSTRIES),
                "region_id": rnd.randint(1, len(REGIONS)),
                "account_tier": rnd.choice(ACCOUNT_TIERS),
                "signup_date": signup,
            }
        )
    dim_customer = pd.DataFrame(customers)

    # ---- DIM_PRODUCT ----
    products = []
    cat_list = list(CATEGORIES.items())
    for pid in range(1, n_products + 1):
        cat, subs = rnd.choice(cat_list)
        sub = rnd.choice(subs)
        name = f"{rnd.choice(PRODUCT_ADJ)} {rnd.choice(PRODUCT_NOUN)} {pid}"
        unit_cost = round(rnd.uniform(20, 900), 2)
        list_price = round(unit_cost * rnd.uniform(1.3, 2.5), 2)
        launch = date(rnd.randint(2016, 2023), rnd.randint(1, 12), rnd.randint(1, 28))
        products.append(
            {
                "product_id": pid,
                "product_name": name,
                "category": cat,
                "subcategory": sub,
                "unit_cost": unit_cost,
                "list_price": list_price,
                "launch_date": launch,
            }
        )
    dim_product = pd.DataFrame(products)
    cost_lookup = dict(zip(dim_product["product_id"], dim_product["unit_cost"]))
    price_lookup = dict(zip(dim_product["product_id"], dim_product["list_price"]))

    # ---- FACT_SALES ----
    sales = []
    for sid in range(1, n_sales + 1):
        pid = rnd.randint(1, n_products)
        cid = rnd.randint(1, n_customers)
        region_id = int(dim_customer.loc[cid - 1, "region_id"])
        did = rnd.choice(date_ids)
        qty = rnd.randint(1, 60)
        base_price = price_lookup[pid]
        unit_price = round(base_price * rnd.uniform(0.85, 1.05), 2)
        discount = round(rnd.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2]), 4)
        revenue = round(qty * unit_price * (1 - discount), 2)
        cost = round(qty * cost_lookup[pid], 2)
        sales.append(
            {
                "sales_id": sid,
                "date_id": did,
                "customer_id": cid,
                "product_id": pid,
                "region_id": region_id,
                "quantity": qty,
                "unit_price": unit_price,
                "discount_pct": discount,
                "revenue": revenue,
                "cost": cost,
            }
        )
    fact_sales = pd.DataFrame(sales)

    # Month-first date ids for monthly facts.
    month_first = (
        dim_date[dim_date["full_date"].map(lambda x: x.day == 1)]["date_id"].tolist()
    )

    # ---- FACT_FORECAST (monthly by product x region, sampled) ----
    forecasts = []
    fid = 1
    for did in month_first:
        for pid in rng.choice(range(1, n_products + 1), size=min(12, n_products), replace=False):
            region_id = int(rng.integers(1, len(REGIONS) + 1))
            fq = int(rng.integers(50, 500))
            fr = round(fq * price_lookup[int(pid)] * float(rng.uniform(0.8, 1.1)), 2)
            forecasts.append(
                {
                    "forecast_id": fid,
                    "date_id": int(did),
                    "product_id": int(pid),
                    "region_id": region_id,
                    "forecast_revenue": fr,
                    "forecast_quantity": fq,
                }
            )
            fid += 1
    fact_forecast = pd.DataFrame(forecasts)

    # ---- FACT_INVENTORY (month-end snapshot, sampled) ----
    inventory = []
    iid = 1
    for did in month_first:
        for pid in rng.choice(range(1, n_products + 1), size=min(15, n_products), replace=False):
            region_id = int(rng.integers(1, len(REGIONS) + 1))
            uoh = int(rng.integers(0, 1000))
            reorder = int(rng.integers(50, 300))
            inventory.append(
                {
                    "inventory_id": iid,
                    "date_id": int(did),
                    "product_id": int(pid),
                    "region_id": region_id,
                    "units_on_hand": uoh,
                    "reorder_point": reorder,
                }
            )
            iid += 1
    fact_inventory = pd.DataFrame(inventory)

    # ---- FACT_SHIPMENT (one per ~80% of sales) ----
    shipments = []
    ship_id = 1
    for _, row in fact_sales.iterrows():
        if rnd.random() < 0.8:
            delivery_days = rnd.randint(1, 14)
            shipments.append(
                {
                    "shipment_id": ship_id,
                    "sales_id": int(row["sales_id"]),
                    "date_id": int(row["date_id"]),
                    "region_id": int(row["region_id"]),
                    "ship_quantity": int(row["quantity"]),
                    "ship_cost": round(rnd.uniform(5, 120), 2),
                    "delivery_days": delivery_days,
                    "on_time_flag": delivery_days <= 7,
                }
            )
            ship_id += 1
    fact_shipment = pd.DataFrame(shipments)

    return {
        "DIM_REGION": regions,
        "DIM_DATE": dim_date,
        "DIM_CUSTOMER": dim_customer,
        "DIM_PRODUCT": dim_product,
        "FACT_SALES": fact_sales,
        "FACT_FORECAST": fact_forecast,
        "FACT_INVENTORY": fact_inventory,
        "FACT_SHIPMENT": fact_shipment,
    }


# ---------------------------------------------------------------------------
# Instruction-tuning record generation
# ---------------------------------------------------------------------------

# Skill routing question banks. Each maps to a target skill.
_ROUTING_BANK: dict[str, list[str]] = {
    "SQL_ANALYST": [
        "How many distinct customers placed orders last year?",
        "List the top 10 products by units sold.",
        "Count the number of sales transactions per region.",
        "What is the average order quantity by product category?",
        "Show total revenue grouped by customer segment.",
    ],
    "FINANCE_ANALYST": [
        "What was our gross margin percentage last quarter?",
        "Calculate year-over-year revenue growth for the technology category.",
        "Break down gross margin by region for fiscal year 2024.",
        "What is forecast attainment for the current year?",
        "Show cost of goods sold trend over the last 12 months.",
    ],
    "SALES_INTELLIGENCE": [
        "Which accounts are trending up in revenue this quarter?",
        "Identify our top 5 growth customers by segment.",
        "Which platinum-tier accounts have declining orders?",
        "What is the win pattern across sales territories?",
        "Rank sales territories by revenue and highlight leaders.",
    ],
    "DOCUMENT_SEARCH": [
        "Find the section of the sales playbook about discount approvals.",
        "What does the pricing policy document say about enterprise deals?",
        "Search the knowledge base for the returns policy.",
        "Where in the onboarding guide is the SLA definition?",
        "Pull the relevant contract clause about delivery penalties.",
    ],
    "GENERAL_QA": [
        "What is a star schema in data warehousing?",
        "Explain the difference between gross margin and net margin.",
        "What does YoY growth mean?",
        "Give me a quick overview of what BI dashboards are for.",
        "What is the purpose of a fact table?",
    ],
}

_CLARIFY_BANK = [
    "Show me the numbers for last period.",
    "How are we doing?",
    "Give me the top ones.",
    "What about the good customers?",
    "Compare it to before.",
    "Which products are best?",
    "Pull the recent data.",
    "How much did we make?",
    "Show the trend.",
    "What changed?",
]

_OUT_OF_SCOPE_BANK = [
    "What is the current stock price of our competitor?",
    "How many employees does the company have?",
    "What is the weather in the North America - West region?",
    "Show me the HR headcount by department.",
    "What is the marketing spend on social media ads?",
    "List all open support tickets from customers.",
    "What is the CEO's travel schedule?",
    "How many parking spaces are at headquarters?",
]

_UNSAFE_TEMPLATES = [
    "Delete all sales records for customer {cid}.",
    "Update the unit price of product {pid} to zero.",
    "Insert {n} fake orders into FACT_SALES.",
    "Drop the FACT_SALES table right now (request {n}).",
    "Truncate the DIM_CUSTOMER table (ticket {n}).",
    "Alter FACT_SALES to add a bogus column col_{n}.",
    "Remove the region with id {rid} from DIM_REGION.",
    "Grant admin access on the sales database to user {n}.",
    "Merge test batch {n} into FACT_FORECAST.",
    "Wipe the inventory table clean for run {n}.",
]


# Value banks used to diversify parametrized SQL questions.
_SEG_VALUES = SEGMENTS
_CAT_VALUES = list(CATEGORIES.keys())
_TIER_VALUES = ACCOUNT_TIERS
_YEAR_VALUES = [2023, 2024, 2025]
_TOPN_VALUES = [3, 5, 10, 15, 20]
_DISC_VALUES = [0.05, 0.10, 0.15, 0.20]


def _parametrized_sql_records(rnd: random.Random) -> list[dict]:
    """Generate a large, DEDUP-FRIENDLY set of parametrized text-to-SQL records.

    Each family varies a literal (segment / category / year / top-N / etc.) so
    that questions and gold SQL are distinct after deduplication.
    """
    out: list[dict] = []

    # Revenue / margin by segment.
    for seg in _SEG_VALUES:
        out.append({
            "task_type": "text_to_sql",
            "question": f"Show total revenue for the {seg} segment.",
            "skill": "SQL_ANALYST", "safety_status": "safe",
            "gold_sql": (
                "SELECT SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_CUSTOMER c ON s.customer_id = c.customer_id "
                f"WHERE c.segment = '{seg}'"),
            "complexity_tags": ["filter", "join", "aggregation"]})
        out.append({
            "task_type": "text_to_sql",
            "question": f"What is the gross margin for the {seg} segment?",
            "skill": "FINANCE_ANALYST", "safety_status": "safe",
            "gold_sql": (
                "SELECT SUM(s.revenue) - SUM(s.cost) AS gross_margin "
                "FROM FACT_SALES s JOIN DIM_CUSTOMER c ON s.customer_id = c.customer_id "
                f"WHERE c.segment = '{seg}'"),
            "complexity_tags": ["calculated_kpi", "filter", "join", "aggregation"]})

    # Revenue / margin % by category.
    for cat in _CAT_VALUES:
        out.append({
            "task_type": "text_to_sql",
            "question": f"Total revenue for the {cat} product category.",
            "skill": "SQL_ANALYST", "safety_status": "safe",
            "gold_sql": (
                "SELECT SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_PRODUCT p ON s.product_id = p.product_id "
                f"WHERE p.category = '{cat}'"),
            "complexity_tags": ["filter", "join", "aggregation"]})
        out.append({
            "task_type": "text_to_sql",
            "question": f"What is the gross margin percentage for the {cat} category?",
            "skill": "FINANCE_ANALYST", "safety_status": "safe",
            "gold_sql": (
                "SELECT (SUM(s.revenue) - SUM(s.cost)) / NULLIF(SUM(s.revenue), 0) * 100 "
                "AS gross_margin_pct "
                "FROM FACT_SALES s JOIN DIM_PRODUCT p ON s.product_id = p.product_id "
                f"WHERE p.category = '{cat}'"),
            "complexity_tags": ["calculated_kpi", "filter", "join", "aggregation"]})

    # Revenue by year, and units by year.
    for yr in _YEAR_VALUES:
        out.append({
            "task_type": "text_to_sql",
            "question": f"What was the total revenue in {yr}?",
            "skill": "SQL_ANALYST", "safety_status": "safe",
            "gold_sql": (
                "SELECT SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_DATE d ON s.date_id = d.date_id "
                f"WHERE d.year = {yr}"),
            "complexity_tags": ["filter", "time_comparison", "join", "aggregation"]})
        out.append({
            "task_type": "text_to_sql",
            "question": f"How many units were sold in {yr}?",
            "skill": "SQL_ANALYST", "safety_status": "safe",
            "gold_sql": (
                "SELECT SUM(s.quantity) AS units_sold "
                "FROM FACT_SALES s JOIN DIM_DATE d ON s.date_id = d.date_id "
                f"WHERE d.year = {yr}"),
            "complexity_tags": ["filter", "time_comparison", "join", "aggregation"]})
        out.append({
            "task_type": "text_to_sql",
            "question": f"Monthly revenue trend for {yr}.",
            "skill": "SQL_ANALYST", "safety_status": "safe",
            "gold_sql": (
                "SELECT d.month, SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_DATE d ON s.date_id = d.date_id "
                f"WHERE d.year = {yr} GROUP BY d.month ORDER BY d.month"),
            "complexity_tags": ["trend", "time_comparison", "join", "aggregation"]})

    # Top-N products and customers.
    for k in _TOPN_VALUES:
        out.append({
            "task_type": "text_to_sql",
            "question": f"List the top {k} products by total revenue.",
            "skill": "SQL_ANALYST", "safety_status": "safe",
            "gold_sql": (
                "SELECT p.product_name, SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_PRODUCT p ON s.product_id = p.product_id "
                f"GROUP BY p.product_name ORDER BY total_revenue DESC LIMIT {k}"),
            "complexity_tags": ["ranking", "join", "aggregation"]})
        out.append({
            "task_type": "text_to_sql",
            "question": f"Which {k} customers generated the most revenue?",
            "skill": "SALES_INTELLIGENCE", "safety_status": "safe",
            "gold_sql": (
                "SELECT c.customer_name, SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_CUSTOMER c ON s.customer_id = c.customer_id "
                f"GROUP BY c.customer_name ORDER BY total_revenue DESC LIMIT {k}"),
            "complexity_tags": ["ranking", "join", "aggregation"]})

    # Discount filters.
    for disc in _DISC_VALUES:
        pct = int(disc * 100)
        out.append({
            "task_type": "text_to_sql",
            "question": f"How many sales had a discount greater than {pct} percent?",
            "skill": "SQL_ANALYST", "safety_status": "safe",
            "gold_sql": f"SELECT COUNT(*) AS n FROM FACT_SALES WHERE discount_pct > {disc}",
            "complexity_tags": ["filter", "aggregation"]})

    # Account-tier revenue (sales intelligence).
    for tier in _TIER_VALUES:
        out.append({
            "task_type": "text_to_sql",
            "question": f"Total revenue from {tier}-tier accounts.",
            "skill": "SALES_INTELLIGENCE", "safety_status": "safe",
            "gold_sql": (
                "SELECT SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_CUSTOMER c ON s.customer_id = c.customer_id "
                f"WHERE c.account_tier = '{tier}'"),
            "complexity_tags": ["filter", "join", "aggregation"]})

    # Region-scoped and dimension breakdowns (fixed, distinct questions).
    fixed = [
        ("What is the total revenue across all sales?",
         "SELECT SUM(revenue) AS total_revenue FROM FACT_SALES",
         ["aggregation"], "SQL_ANALYST"),
        ("How many total units were sold?",
         "SELECT SUM(quantity) AS units_sold FROM FACT_SALES",
         ["aggregation"], "SQL_ANALYST"),
        ("What is the average discount applied on sales?",
         "SELECT AVG(discount_pct) AS avg_discount FROM FACT_SALES",
         ["aggregation"], "SQL_ANALYST"),
        ("What is the overall gross margin?",
         "SELECT SUM(revenue) - SUM(cost) AS gross_margin FROM FACT_SALES",
         ["calculated_kpi", "aggregation"], "FINANCE_ANALYST"),
        ("What is the overall gross margin percentage?",
         "SELECT (SUM(revenue) - SUM(cost)) / NULLIF(SUM(revenue), 0) * 100 "
         "AS gross_margin_pct FROM FACT_SALES",
         ["calculated_kpi", "aggregation"], "FINANCE_ANALYST"),
        ("Total revenue by product category.",
         "SELECT p.category, SUM(s.revenue) AS total_revenue FROM FACT_SALES s "
         "JOIN DIM_PRODUCT p ON s.product_id = p.product_id GROUP BY p.category "
         "ORDER BY total_revenue DESC",
         ["join", "aggregation"], "SQL_ANALYST"),
        ("Revenue by region name.",
         "SELECT r.region_name, SUM(s.revenue) AS total_revenue FROM FACT_SALES s "
         "JOIN DIM_REGION r ON s.region_id = r.region_id GROUP BY r.region_name "
         "ORDER BY total_revenue DESC",
         ["join", "aggregation"], "SQL_ANALYST"),
        ("Show total revenue by year.",
         "SELECT d.year, SUM(s.revenue) AS total_revenue FROM FACT_SALES s "
         "JOIN DIM_DATE d ON s.date_id = d.date_id GROUP BY d.year ORDER BY d.year",
         ["time_comparison", "join", "aggregation"], "SQL_ANALYST"),
        ("Gross margin percentage by product category.",
         "SELECT p.category, (SUM(s.revenue) - SUM(s.cost)) / NULLIF(SUM(s.revenue), 0) * 100 "
         "AS gross_margin_pct FROM FACT_SALES s JOIN DIM_PRODUCT p ON s.product_id = p.product_id "
         "GROUP BY p.category ORDER BY gross_margin_pct DESC",
         ["calculated_kpi", "join", "aggregation"], "FINANCE_ANALYST"),
        ("On-time delivery rate across shipments.",
         "SELECT AVG(CASE WHEN on_time_flag THEN 1.0 ELSE 0.0 END) * 100 "
         "AS on_time_rate FROM FACT_SHIPMENT",
         ["calculated_kpi", "aggregation"], "SQL_ANALYST"),
        ("Revenue by customer segment.",
         "SELECT c.segment, SUM(s.revenue) AS total_revenue FROM FACT_SALES s "
         "JOIN DIM_CUSTOMER c ON s.customer_id = c.customer_id GROUP BY c.segment "
         "ORDER BY total_revenue DESC",
         ["join", "aggregation"], "SQL_ANALYST"),
        ("Average delivery days by region.",
         "SELECT r.region_name, AVG(sh.delivery_days) AS avg_days FROM FACT_SHIPMENT sh "
         "JOIN DIM_REGION r ON sh.region_id = r.region_id GROUP BY r.region_name "
         "ORDER BY avg_days",
         ["join", "aggregation"], "SQL_ANALYST"),
        ("Which product categories have the highest units sold?",
         "SELECT p.category, SUM(s.quantity) AS units FROM FACT_SALES s "
         "JOIN DIM_PRODUCT p ON s.product_id = p.product_id GROUP BY p.category "
         "ORDER BY units DESC",
         ["ranking", "join", "aggregation"], "SQL_ANALYST"),
        ("Total forecast revenue by region.",
         "SELECT r.region_name, SUM(f.forecast_revenue) AS forecast_rev FROM FACT_FORECAST f "
         "JOIN DIM_REGION r ON f.region_id = r.region_id GROUP BY r.region_name "
         "ORDER BY forecast_rev DESC",
         ["join", "aggregation"], "FINANCE_ANALYST"),
        ("How many customers are in each industry?",
         "SELECT industry, COUNT(*) AS n FROM DIM_CUSTOMER GROUP BY industry "
         "ORDER BY n DESC",
         ["aggregation"], "SQL_ANALYST"),
    ]
    for q, sql, tags, skill in fixed:
        out.append({"task_type": "text_to_sql", "question": q, "skill": skill,
                    "safety_status": "safe", "gold_sql": sql, "complexity_tags": tags})

    return out


def _sql_records(rnd: random.Random, n: int) -> list[dict]:
    """Generate ``n`` text-to-SQL records.

    Draws from a diverse parametrized pool; if ``n`` exceeds the unique pool
    size, paraphrase prefixes are added to reach the count while keeping
    questions distinct.
    """
    pool = _parametrized_sql_records(rnd)
    rnd.shuffle(pool)
    records: list[dict] = [dict(r) for r in pool[:n]]

    # If we still need more, create distinct paraphrases of pool items.
    paraphrase_prefixes = [
        "Can you tell me", "I need to know", "Please report",
        "For the dashboard,", "Quick question:", "As of the latest data,",
    ]
    pi = 0
    while len(records) < n:
        base = pool[pi % len(pool)]
        prefix = paraphrase_prefixes[(pi // len(pool)) % len(paraphrase_prefixes)]
        q = base["question"]
        q_low = q[0].lower() + q[1:]
        rec = dict(base)
        rec["question"] = f"{prefix} {q_low}"
        records.append(rec)
        pi += 1
    return records


def _sql_records_legacy(rnd: random.Random, n: int) -> list[dict]:
    """Deprecated: original small template set (kept for reference/tests)."""
    records: list[dict] = []

    # Each template: (question builder, sql builder, complexity tags, skill)
    templates = [
        # aggregation
        (
            lambda: "What is the total revenue across all sales?",
            lambda: "SELECT SUM(revenue) AS total_revenue FROM FACT_SALES",
            ["aggregation"], "SQL_ANALYST",
        ),
        (
            lambda: "How many total units were sold?",
            lambda: "SELECT SUM(quantity) AS units_sold FROM FACT_SALES",
            ["aggregation"], "SQL_ANALYST",
        ),
        (
            lambda: "What is the average discount applied on sales?",
            lambda: "SELECT AVG(discount_pct) AS avg_discount FROM FACT_SALES",
            ["aggregation"], "SQL_ANALYST",
        ),
        # ranking / top-n
        (
            lambda: f"List the top {(k := rnd.choice([5, 10]))} products by total revenue.",
            lambda k=None: (
                "SELECT p.product_name, SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_PRODUCT p ON s.product_id = p.product_id "
                "GROUP BY p.product_name ORDER BY total_revenue DESC LIMIT 10"
            ),
            ["ranking", "join", "aggregation"], "SQL_ANALYST",
        ),
        (
            lambda: "Which 5 customers generated the most revenue?",
            lambda: (
                "SELECT c.customer_name, SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_CUSTOMER c ON s.customer_id = c.customer_id "
                "GROUP BY c.customer_name ORDER BY total_revenue DESC LIMIT 5"
            ),
            ["ranking", "join", "aggregation"], "SQL_ANALYST",
        ),
        # filter
        (
            lambda: "Show total revenue for the Enterprise segment.",
            lambda: (
                "SELECT SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_CUSTOMER c ON s.customer_id = c.customer_id "
                "WHERE c.segment = 'Enterprise'"
            ),
            ["filter", "join", "aggregation"], "SQL_ANALYST",
        ),
        (
            lambda: "How many sales had a discount greater than 10 percent?",
            lambda: "SELECT COUNT(*) AS n FROM FACT_SALES WHERE discount_pct > 0.10",
            ["filter", "aggregation"], "SQL_ANALYST",
        ),
        # join by dimension
        (
            lambda: "Total revenue by product category.",
            lambda: (
                "SELECT p.category, SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_PRODUCT p ON s.product_id = p.product_id "
                "GROUP BY p.category ORDER BY total_revenue DESC"
            ),
            ["join", "aggregation"], "SQL_ANALYST",
        ),
        (
            lambda: "Revenue by region name.",
            lambda: (
                "SELECT r.region_name, SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_REGION r ON s.region_id = r.region_id "
                "GROUP BY r.region_name ORDER BY total_revenue DESC"
            ),
            ["join", "aggregation"], "SQL_ANALYST",
        ),
        # time comparison / trend
        (
            lambda: "Show total revenue by year.",
            lambda: (
                "SELECT d.year, SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_DATE d ON s.date_id = d.date_id "
                "GROUP BY d.year ORDER BY d.year"
            ),
            ["time_comparison", "join", "aggregation"], "SQL_ANALYST",
        ),
        (
            lambda: "Monthly revenue trend for 2024.",
            lambda: (
                "SELECT d.month, SUM(s.revenue) AS total_revenue "
                "FROM FACT_SALES s JOIN DIM_DATE d ON s.date_id = d.date_id "
                "WHERE d.year = 2024 GROUP BY d.month ORDER BY d.month"
            ),
            ["trend", "time_comparison", "join", "aggregation"], "SQL_ANALYST",
        ),
        # calculated KPI: gross margin
        (
            lambda: "What is the overall gross margin?",
            lambda: "SELECT SUM(revenue) - SUM(cost) AS gross_margin FROM FACT_SALES",
            ["calculated_kpi", "aggregation"], "FINANCE_ANALYST",
        ),
        (
            lambda: "What is the gross margin percentage?",
            lambda: (
                "SELECT (SUM(revenue) - SUM(cost)) / NULLIF(SUM(revenue), 0) * 100 "
                "AS gross_margin_pct FROM FACT_SALES"
            ),
            ["calculated_kpi", "aggregation"], "FINANCE_ANALYST",
        ),
        (
            lambda: "Gross margin percentage by product category.",
            lambda: (
                "SELECT p.category, "
                "(SUM(s.revenue) - SUM(s.cost)) / NULLIF(SUM(s.revenue), 0) * 100 AS gross_margin_pct "
                "FROM FACT_SALES s JOIN DIM_PRODUCT p ON s.product_id = p.product_id "
                "GROUP BY p.category ORDER BY gross_margin_pct DESC"
            ),
            ["calculated_kpi", "join", "aggregation"], "FINANCE_ANALYST",
        ),
        (
            lambda: "On-time delivery rate across shipments.",
            lambda: (
                "SELECT AVG(CASE WHEN on_time_flag THEN 1.0 ELSE 0.0 END) * 100 "
                "AS on_time_rate FROM FACT_SHIPMENT"
            ),
            ["calculated_kpi", "aggregation"], "SQL_ANALYST",
        ),
    ]

    i = 0
    while len(records) < n:
        q_fn, sql_fn, tags, skill = templates[i % len(templates)]
        i += 1
        question = q_fn()
        sql = sql_fn()
        records.append(
            {
                "task_type": "text_to_sql",
                "question": question,
                "skill": skill,
                "safety_status": "safe",
                "gold_sql": sql,
                "complexity_tags": tags,
            }
        )
    return records


def build_instruction_records(
    counts: dict[str, int] | None = None,
    schema_id: str = "enterprise_sales_v1",
    seed: int = SEED,
) -> list[dict]:
    """Build the full instruction dataset (list of dicts, pre-schema-context).

    ``counts`` overrides per-task-type counts. Records include ``id`` and
    ``schema_id`` but NOT ``schema_context`` (added in preprocessing so the
    schema can be swapped).
    """
    rnd = random.Random(seed)
    counts = counts or {
        "text_to_sql": 1000,
        "skill_routing": 400,
        "refuse_unsafe": 150,
        "needs_clarification": 100,
        "insufficient_schema": 50,
    }
    records: list[dict] = []

    # text_to_sql
    records += _sql_records(rnd, counts["text_to_sql"])

    # skill_routing -- expand each base question with distinct paraphrases.
    routing: list[dict] = []
    routing_prefixes = [
        "", "Hey, ", "Please help: ", "For my report, ", "I was wondering, ",
        "Analytics team asks: ", "Could you tell me: ", "Quick one -- ",
        "For the exec deck, ", "When you have a sec, ", "Priority request: ",
        "Follow-up: ", "On today's call: ", "Can we get ", "Help me understand: ",
        "For Q review, ", "Ping: ", "Kindly ",
    ]
    routing_pool: list[tuple[str, str]] = []
    for skill, questions in _ROUTING_BANK.items():
        for q in questions:
            for pref in routing_prefixes:
                if pref:
                    text = pref + (q[0].lower() + q[1:])
                else:
                    text = q
                routing_pool.append((skill, text))
    rnd.shuffle(routing_pool)
    for skill, q in routing_pool:
        if len(routing) >= counts["skill_routing"]:
            break
        routing.append(
            {
                "task_type": "skill_routing",
                "question": q,
                "skill": skill,
                "safety_status": "safe",
                "gold_sql": None,
                "complexity_tags": ["routing"],
            }
        )
    records += routing

    # refuse_unsafe -- vary the injected ids so each is distinct. Cycle the
    # template by iteration count (not by len(unsafe)) so no single template
    # with a small value range can trap the loop.
    unsafe: list[dict] = []
    used_unsafe: set[str] = set()
    _u_it = 0
    while len(unsafe) < counts["refuse_unsafe"]:
        tmpl = _UNSAFE_TEMPLATES[_u_it % len(_UNSAFE_TEMPLATES)]
        _u_it += 1
        q = tmpl.format(cid=rnd.randint(1, 200), pid=rnd.randint(1, 50),
                        rid=rnd.randint(1, 8), n=rnd.randint(1, 99999))
        if q in used_unsafe:
            continue
        used_unsafe.add(q)
        unsafe.append(
            {
                "task_type": "refuse_unsafe",
                "question": q,
                "skill": "REFUSE_UNSAFE",
                "safety_status": "unsafe",
                "gold_sql": None,
                "complexity_tags": ["write_operation", "unsafe"],
            }
        )
    records += unsafe

    # needs_clarification -- paraphrase to reach the count with distinct text.
    clarify: list[dict] = []
    clarify_pool: list[str] = []
    clarify_prefixes = ["", "So, ", "Um, ", "OK ", "Hey ", "Just ", "Also ", "Quick: ", "Btw ", "Real quick "]
    for base in _CLARIFY_BANK:
        for pref in clarify_prefixes:
            clarify_pool.append((pref + (base[0].lower() + base[1:])) if pref else base)
    rnd.shuffle(clarify_pool)
    for q in clarify_pool:
        if len(clarify) >= counts["needs_clarification"]:
            break
        clarify.append(
            {
                "task_type": "needs_clarification",
                "question": q,
                "skill": "NEEDS_CLARIFICATION",
                "safety_status": "needs_clarification",
                "gold_sql": None,
                "complexity_tags": ["ambiguous"],
            }
        )
    records += clarify

    # insufficient_schema / out of scope -- paraphrase for distinct text.
    oos: list[dict] = []
    oos_pool: list[str] = []
    oos_prefixes = ["", "Can you find ", "I want ", "Show me ", "Tell me ", "Look up ", "Get me "]
    for base in _OUT_OF_SCOPE_BANK:
        for pref in oos_prefixes:
            oos_pool.append((pref + (base[0].lower() + base[1:])) if pref else base)
    rnd.shuffle(oos_pool)
    for q in oos_pool:
        if len(oos) >= counts["insufficient_schema"]:
            break
        oos.append(
            {
                "task_type": "insufficient_schema",
                "question": q,
                "skill": "GENERAL_QA",
                "safety_status": "out_of_scope",
                "gold_sql": None,
                "complexity_tags": ["out_of_scope"],
            }
        )
    records += oos

    # Assign stable ids and schema_id.
    rnd.shuffle(records)
    for idx, rec in enumerate(records):
        rec["id"] = f"{schema_id}-{idx:05d}"
        rec["schema_id"] = schema_id
    return records
