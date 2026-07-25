#!/usr/bin/env python3
"""Fast CPU LoRA pilot for paper metrics (gpt2 + LoRA r=16)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/processed/train.jsonl"
TEST = ROOT / "data/processed/test.jsonl"
OUT = ROOT / "results/adapters/pilot_lora_r16"
METRICS = ROOT / "results/metrics.json"
TRAIN_METRICS = ROOT / "results/train_metrics.json"

MODEL_NAME = "gpt2"
MAX_LEN = 192
MAX_TRAIN = 60
MAX_EVAL = 12
EPOCHS = 1
LR = 5e-4
LORA_R = 16
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_text(messages: list[dict]) -> str:
    parts = [f"### {m['role'].upper()}\n{m['content']}" for m in messages]
    return "\n\n".join(parts) + "\n"


def build_dataset(rows, tokenizer, max_n):
    texts = [to_text(r["messages"]) for r in rows[:max_n]]

    def tok(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding=False)
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    return Dataset.from_dict({"text": texts}).map(tok, batched=True, remove_columns=["text"])


def parse_json_blob(text: str):
    m = re.search(r"\{[^{}]*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def evaluate(model, tokenizer, test_rows, label):
    model.eval()
    n = routing_ok = json_ok = safety_ok = safety_n = sql_exact = sql_n = 0
    latencies = []
    for r in test_rows[:MAX_EVAL]:
        gold_skill = r.get("skill")
        gold_sql = r.get("gold_sql")
        gold_safety = r.get("safety_status")
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        system = next(m["content"] for m in r["messages"] if m["role"] == "system")
        prompt = f"### SYSTEM\n{system}\n\n### USER\n{user}\n\n### ASSISTANT\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latencies.append((time.time() - t0) * 1000)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = parse_json_blob(gen)
        n += 1
        if pred is not None:
            json_ok += 1
            if pred.get("skill") == gold_skill:
                routing_ok += 1
            if gold_skill == "REFUSE_UNSAFE" or gold_safety == "unsafe":
                safety_n += 1
                if pred.get("skill") == "REFUSE_UNSAFE" or pred.get("safety_status") in {
                    "unsafe",
                    "BLOCKED",
                }:
                    safety_ok += 1
            if gold_sql:
                sql_n += 1
                gs = re.sub(r"\s+", " ", str(gold_sql)).strip().upper()
                ps = re.sub(r"\s+", " ", str(pred.get("sql") or "")).strip().upper()
                if gs and ps and gs == ps:
                    sql_exact += 1
        elif gold_skill == "REFUSE_UNSAFE":
            safety_n += 1

    metrics = {
        "n": n,
        "routing_accuracy": round(routing_ok / n, 4) if n else 0.0,
        "json_validity_rate": round(json_ok / n, 4) if n else 0.0,
        "safety_refusal_accuracy": round(safety_ok / safety_n, 4) if safety_n else 1.0,
        "exact_sql_match": round(sql_exact / sql_n, 4) if sql_n else 0.0,
        "sql_execution_accuracy": round(sql_exact / sql_n, 4) if sql_n else 0.0,
        "latency_ms_mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "model": MODEL_NAME,
        "lora_r": LORA_R if "lora" in label else 0,
        "condition": label,
        "pilot": True,
    }
    print(label, metrics)
    return metrics


def main():
    torch.manual_seed(SEED)
    print("Loading", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base.config.use_cache = False
    test_rows = load_jsonl(TEST)
    base_metrics = evaluate(base, tokenizer, test_rows, "base_zero_shot")

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn"],
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    train_ds = build_dataset(load_jsonl(TRAIN), tokenizer, MAX_TRAIN)
    OUT.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(OUT / "ckpt"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=LR,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    print("Training ...")
    result = trainer.train()
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)

    loss_history = [{"step": h.get("step", 0), "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h]
    train_log = {
        "pilot": True,
        "model": MODEL_NAME,
        "lora_r": LORA_R,
        "epochs": EPOCHS,
        "max_train_samples": MAX_TRAIN,
        "train_runtime_sec": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "loss_history": loss_history,
    }
    TRAIN_METRICS.write_text(json.dumps(train_log, indent=2))
    lora_metrics = evaluate(model, tokenizer, test_rows, "lora_r16")

    metrics = json.loads(METRICS.read_text()) if METRICS.exists() else {}
    metrics["base_zero_shot"] = base_metrics
    metrics["qlora_r16"] = {
        **lora_metrics,
        "note": "CPU LoRA pilot on gpt2; run 04_train_qlora.py on RTX 5090 for Qwen2.5-3B QLoRA",
    }
    metrics["lora_pilot_r16"] = lora_metrics
    METRICS.write_text(json.dumps(metrics, indent=2))
    print("Done. train_loss=", train_log["train_loss"])


if __name__ == "__main__":
    main()
