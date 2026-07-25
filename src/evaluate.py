"""Evaluation driver.

Runs one or more prediction conditions over the held-out test set and computes
metrics against DuckDB execution. Designed to run fully offline via the
``template_baseline`` condition, and to add GPU conditions when adapters exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from .metrics import compute_all
from .schema import load_schema

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "raw" / "enterprise.duckdb"
DEFAULT_TEST = REPO_ROOT / "data" / "processed" / "test.jsonl"

# Three-shot demonstrations used by the base_few_shot condition.
FEW_SHOT_EXAMPLES = [
    (
        "Schema (enterprise_sales_v1): ...\nQuestion: What is the total revenue across all sales?",
        json.dumps({
            "skill": "SQL_ANALYST", "action": "run_sql", "safety_status": "safe",
            "needs_clarification": False,
            "sql": "SELECT SUM(revenue) AS total_revenue FROM FACT_SALES",
        }),
    ),
    (
        "Schema (enterprise_sales_v1): ...\nQuestion: Delete all sales records for customer 5.",
        json.dumps({
            "skill": "REFUSE_UNSAFE", "action": "refuse", "safety_status": "unsafe",
            "needs_clarification": False, "sql": None,
        }),
    ),
    (
        "Schema (enterprise_sales_v1): ...\nQuestion: How are we doing?",
        json.dumps({
            "skill": "NEEDS_CLARIFICATION", "action": "clarify",
            "safety_status": "needs_clarification", "needs_clarification": True, "sql": None,
        }),
    ),
]


def load_test_set(path: str | Path = DEFAULT_TEST) -> list[dict]:
    """Load JSONL test samples."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _gold_from_sample(sample: dict) -> dict:
    return {
        "skill": sample["skill"],
        "safety_status": sample["safety_status"],
        "gold_sql": sample.get("gold_sql"),
    }


def run_condition(predictor, test_samples: list[dict], con) -> dict:
    """Run a predictor over the test set and compute metrics.

    Returns ``{"metrics": {...}, "predictions": [...]}``.
    """
    preds: list[dict] = []
    golds: list[dict] = []
    raw_outputs: list[str] = []
    latencies: list[float] = []
    predictions_log: list[dict] = []

    for sample in test_samples:
        # Never forward the gold assistant turn: test samples share the
        # training-file layout, so ``messages`` ends with the reference
        # answer. Leaking it lets the model copy its own prompt.
        prompt_messages = [m for m in sample["messages"] if m["role"] != "assistant"]
        contract, raw, latency = predictor.predict(prompt_messages)
        pred = {
            "skill": contract.get("skill"),
            "safety_status": contract.get("safety_status"),
            "sql": contract.get("sql"),
        }
        preds.append(pred)
        golds.append(_gold_from_sample(sample))
        raw_outputs.append(raw)
        latencies.append(latency)
        predictions_log.append(
            {
                "id": sample["id"],
                "task_type": sample["task_type"],
                "question": sample["messages"][1]["content"].split("Question:")[-1].strip()[:200],
                "gold_skill": sample["skill"],
                "pred_skill": pred["skill"],
                "gold_sql": sample.get("gold_sql"),
                "pred_sql": pred["sql"],
                "gold_safety": sample["safety_status"],
                "pred_safety": pred["safety_status"],
                "raw": raw[:500],
            }
        )

    metrics = compute_all(con, preds, golds, raw_outputs=raw_outputs, latencies_ms=latencies)
    return {"metrics": metrics, "predictions": predictions_log}


def evaluate(
    conditions: list[str],
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    adapters_dir: str = "results/adapters",
    db_path: str | Path = DEFAULT_DB,
    test_path: str | Path = DEFAULT_TEST,
    out_dir: str | Path = "results",
    max_samples: int | None = None,
) -> dict:
    """Evaluate the requested conditions and write metrics/predictions.

    Supported conditions: ``template_baseline``, ``base_zero_shot``,
    ``base_few_shot``, ``qlora_r8``, ``qlora_r16``, ``qlora_r32``.
    GPU conditions are skipped gracefully if unavailable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_samples = load_test_set(test_path)
    if max_samples:
        test_samples = test_samples[:max_samples]

    con = duckdb.connect(str(db_path), read_only=True)
    schema = load_schema()  # noqa: F841 (loaded to ensure schema is present)

    all_metrics: dict = {}
    for cond in conditions:
        predictor, note = _make_predictor(cond, model_name, adapters_dir)
        if predictor is None:
            print(f"[eval] skipping '{cond}': {note}")
            all_metrics[cond] = {"skipped": True, "reason": note}
            continue
        print(f"[eval] running condition '{cond}' over {len(test_samples)} samples...")
        result = run_condition(predictor, test_samples, con)
        # Attach param/vram info if we can find it.
        result["metrics"].update(_augment_resource_metrics(cond, adapters_dir))
        all_metrics[cond] = result["metrics"]
        (out_dir / f"predictions_{cond}.jsonl").write_text(
            "\n".join(json.dumps(p) for p in result["predictions"])
        )

    con.close()
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"[eval] wrote {metrics_path}")
    return all_metrics


def _make_predictor(cond: str, model_name: str, adapters_dir: str):
    """Return ``(predictor_or_None, note)`` for a condition."""
    from .inference import TemplateBaselinePredictor

    if cond == "template_baseline":
        return TemplateBaselinePredictor(), "ok"

    # GPU-backed conditions.
    try:
        import torch

        if not torch.cuda.is_available():
            return None, "no CUDA GPU available"
    except Exception:
        return None, "transformers/torch not installed"

    from .inference import HFPredictor

    if cond == "base_zero_shot":
        return HFPredictor(model_name=model_name), "ok"
    if cond == "base_few_shot":
        return HFPredictor(model_name=model_name, few_shot=FEW_SHOT_EXAMPLES), "ok"
    if cond.startswith("qlora_r"):
        r = cond.split("qlora_r")[-1]
        adapter_dir = Path(adapters_dir) / f"qlora_r{r}"
        if not adapter_dir.exists():
            return None, f"adapter not found at {adapter_dir}"
        return HFPredictor(model_name=model_name, adapter_dir=str(adapter_dir)), "ok"
    return None, f"unknown condition '{cond}'"


def _augment_resource_metrics(cond: str, adapters_dir: str) -> dict:
    """Pull trainable_params / peak_vram from train_metrics.json if present."""
    out: dict = {}
    tm_path = REPO_ROOT / "results" / "train_metrics.json"
    if cond.startswith("qlora_r") and tm_path.exists():
        try:
            tm = json.loads(tm_path.read_text()).get(cond, {})
            if "trainable_params" in tm:
                out["trainable_params"] = tm["trainable_params"]
            if "peak_vram_gb" in tm:
                out["peak_vram_gb"] = tm["peak_vram_gb"]
        except Exception:
            pass
    return out
