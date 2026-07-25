#!/usr/bin/env python
"""Stage 06: publication-quality figures from results/metrics.json.

Generates IEEE-friendly figures (readable in grayscale, distinct hatches +
color) at 300 dpi:
    figures/routing_accuracy.png
    figures/sql_exec_accuracy.png
    figures/safety_json_metrics.png
    figures/training_loss.png            (if results/train_metrics.json exists)
    figures/resource_comparison.png      (if trainable_params/vram present)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"

# IEEE-friendly academic style.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Distinct color + hatch pairs so bars remain distinguishable in grayscale.
_HATCHES = ["", "//", "\\\\", "xx", "..", "++"]
_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

_COND_LABELS = {
    "template_baseline": "Template\nbaseline",
    "base_zero_shot": "Base\nzero-shot",
    "base_few_shot": "Base\nfew-shot",
    "qlora_r8": "QLoRA r=8",
    "qlora_r16": "QLoRA r=16",
    "qlora_r32": "QLoRA r=32",
}


def _load_metrics() -> dict:
    path = RESULTS / "metrics.json"
    if not path.exists():
        raise SystemExit("[06] results/metrics.json not found. Run scripts/05 first.")
    return json.loads(path.read_text())


def _active_conditions(metrics: dict) -> list[str]:
    return [c for c, m in metrics.items() if not m.get("skipped")]


def _bar_metric(metrics: dict, key: str, title: str, ylabel: str, out: Path,
                percent: bool = True) -> None:
    conds = _active_conditions(metrics)
    values = [metrics[c].get(key, 0) or 0 for c in conds]
    if percent:
        values = [v * 100 for v in values]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = range(len(conds))
    for i, (xi, v) in enumerate(zip(x, values)):
        ax.bar(xi, v, color=_COLORS[i % len(_COLORS)],
               hatch=_HATCHES[i % len(_HATCHES)], edgecolor="black", linewidth=0.6)
        ax.text(xi, v + (1 if percent else 0.01), f"{v:.1f}", ha="center",
                va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([_COND_LABELS.get(c, c) for c in conds])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.set_ylim(0, 105)
    fig.savefig(out)
    plt.close(fig)
    print(f"[06] wrote {out}")


def plot_routing(metrics: dict) -> None:
    _bar_metric(metrics, "routing_accuracy", "Skill Routing Accuracy",
                "Accuracy (%)", FIGURES / "routing_accuracy.png")


def plot_sql_exec(metrics: dict) -> None:
    _bar_metric(metrics, "sql_execution_accuracy", "SQL Execution Accuracy",
                "Execution accuracy (%)", FIGURES / "sql_exec_accuracy.png")


def plot_safety_json(metrics: dict) -> None:
    conds = _active_conditions(metrics)
    safety = [(metrics[c].get("safety_refusal_accuracy", 0) or 0) * 100 for c in conds]
    jsonv = [(metrics[c].get("json_validity_rate", 0) or 0) * 100 for c in conds]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = range(len(conds))
    w = 0.38
    ax.bar([i - w / 2 for i in x], safety, width=w, label="Safety refusal",
           color=_COLORS[0], hatch="//", edgecolor="black", linewidth=0.6)
    ax.bar([i + w / 2 for i in x], jsonv, width=w, label="JSON validity",
           color=_COLORS[2], hatch="..", edgecolor="black", linewidth=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels([_COND_LABELS.get(c, c) for c in conds])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Safety Refusal & JSON Validity")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "safety_json_metrics.png")
    plt.close(fig)
    print(f"[06] wrote {FIGURES / 'safety_json_metrics.png'}")


def plot_training_loss() -> None:
    path = RESULTS / "train_metrics.json"
    if not path.exists():
        print("[06] (skip training_loss: no train_metrics.json)")
        return
    tm = json.loads(path.read_text())
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    plotted = False
    for i, (run, summary) in enumerate(tm.items()):
        if not isinstance(summary, dict):
            # Legacy pilot runs wrote flat scalars at the top level.
            continue
        hist = summary.get("loss_history", [])
        if not hist:
            continue
        steps = [h["step"] for h in hist]
        losses = [h["loss"] for h in hist]
        ax.plot(steps, losses, marker="o", markersize=2.5, linewidth=1.2,
                color=_COLORS[i % len(_COLORS)], label=run)
        plotted = True
    if not plotted:
        print("[06] (skip training_loss: no loss history)")
        plt.close(fig)
        return
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("QLoRA Training Loss")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "training_loss.png")
    plt.close(fig)
    print(f"[06] wrote {FIGURES / 'training_loss.png'}")


def plot_resource_comparison(metrics: dict) -> None:
    conds = [c for c in _active_conditions(metrics)
             if metrics[c].get("trainable_params") or metrics[c].get("peak_vram_gb")]
    if not conds:
        print("[06] (skip resource_comparison: no resource metrics)")
        return
    params_m = [(metrics[c].get("trainable_params", 0) or 0) / 1e6 for c in conds]
    vram = [metrics[c].get("peak_vram_gb", 0) or 0 for c in conds]

    fig, ax1 = plt.subplots(figsize=(6.4, 3.6))
    x = range(len(conds))
    w = 0.38
    b1 = ax1.bar([i - w / 2 for i in x], params_m, width=w, color=_COLORS[0],
                 hatch="//", edgecolor="black", linewidth=0.6, label="Trainable params (M)")
    ax1.set_ylabel("Trainable params (M)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([_COND_LABELS.get(c, c) for c in conds])
    ax2 = ax1.twinx()
    b2 = ax2.bar([i + w / 2 for i in x], vram, width=w, color=_COLORS[3],
                 hatch="..", edgecolor="black", linewidth=0.6, label="Peak VRAM (GB)")
    ax2.set_ylabel("Peak VRAM (GB)")
    ax1.set_title("Resource Footprint by Adapter Rank")
    ax1.legend(handles=[b1, b2], frameon=False, fontsize=8, loc="upper left")
    fig.savefig(FIGURES / "resource_comparison.png")
    plt.close(fig)
    print(f"[06] wrote {FIGURES / 'resource_comparison.png'}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics()
    plot_routing(metrics)
    plot_sql_exec(metrics)
    plot_safety_json(metrics)
    plot_training_loss()
    plot_resource_comparison(metrics)
    print("[06] Done.")


if __name__ == "__main__":
    main()
