"""QLoRA training core.

Wraps transformers + peft + trl + bitsandbytes into a single entry point.
Imports of heavy GPU libraries are deferred so that this module can be imported
(and unit-tested) on machines without CUDA or the training stack installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """Training hyper-parameters. Mirrors the YAML configs in ``configs/``."""

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/val.jsonl"
    output_dir: str = "results/adapters/qlora_r16"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Optimization
    epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_len: int = 1024
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_strategy: str = "epoch"
    packing: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    # Smoke testing
    max_samples: int | None = None


def detect_hardware() -> dict:
    """Return a dict describing available compute (safe if torch missing)."""
    info = {"cuda": False, "device_name": "cpu", "vram_gb": 0.0, "bf16": False}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(
                cuda=True,
                device_name=props.name,
                vram_gb=round(props.total_memory / 1e9, 1),
                bf16=torch.cuda.is_bf16_supported(),
            )
    except Exception:
        pass
    return info


def auto_batch_size(vram_gb: float) -> tuple[int, int]:
    """Pick (per_device_batch_size, grad_accum) for a target VRAM budget.

    Tuned for a 3B model with 4-bit QLoRA at seq_len 1024.
    """
    if vram_gb >= 40:  # A100 40/80GB
        return 8, 2
    if vram_gb >= 24:  # RTX 5090 (32GB) / 4090 (24GB)
        return 4, 4
    if vram_gb >= 14:  # T4 16GB / Colab
        return 2, 8
    return 1, 16  # tiny / safety


def set_seed_everywhere(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def train_qlora(cfg: TrainConfig) -> dict:
    """Run QLoRA fine-tuning. Returns a metrics summary dict.

    Raises a clear RuntimeError (rather than crashing on import) if the GPU
    training stack is unavailable.
    """
    hw = detect_hardware()
    set_seed_everywhere(cfg.seed)

    if not hw["cuda"]:
        msg = (
            "No CUDA GPU detected. QLoRA 4-bit training with bitsandbytes "
            "requires a CUDA GPU (target: RTX 5090 32GB, or Colab T4/A100). "
            "Use scripts/05_evaluate.py --condition template_baseline for an "
            "offline, CPU-only pipeline sanity check."
        )
        raise RuntimeError(msg)

    # ---- Heavy imports (only reached with CUDA) ----
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    # Adaptive batch sizing.
    bs, ga = auto_batch_size(hw["vram_gb"])
    per_device_bs = cfg.per_device_batch_size or bs
    grad_accum = cfg.gradient_accumulation_steps or ga
    compute_dtype = torch.bfloat16 if hw["bf16"] else torch.float16

    print(f"[train] Hardware: {hw}")
    print(f"[train] batch_size={per_device_bs} grad_accum={grad_accum} dtype={compute_dtype}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4-bit NF4 quantization ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = _count_params(model)
    print(f"[train] trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ---- Data ----
    data_files = {"train": cfg.train_file, "validation": cfg.val_file}
    ds = load_dataset("json", data_files=data_files)
    if cfg.max_samples:
        ds["train"] = ds["train"].select(range(min(cfg.max_samples, len(ds["train"]))))
        n_val = min(max(cfg.max_samples // 5, 1), len(ds["validation"]))
        ds["validation"] = ds["validation"].select(range(n_val))

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(format_chat, remove_columns=ds["train"].column_names)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # trl renamed several kwargs across major versions (max_seq_length ->
    # max_length, SFTTrainer tokenizer -> processing_class); pick whichever
    # the installed version accepts.
    import inspect

    sft_params = inspect.signature(SFTConfig.__init__).parameters
    sft_kwargs = dict(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy="epoch",
        bf16=hw["bf16"],
        fp16=not hw["bf16"],
        packing=cfg.packing,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim="paged_adamw_8bit",
        seed=cfg.seed,
        report_to="none",
    )
    seq_len_key = "max_seq_length" if "max_seq_length" in sft_params else "max_length"
    sft_kwargs[seq_len_key] = cfg.max_seq_len
    if "dataset_text_field" in sft_params:
        sft_kwargs["dataset_text_field"] = "text"
    sft_config = SFTConfig(**sft_kwargs)

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    tok_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        **{tok_key: tokenizer},
    )

    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # ---- Metrics ----
    peak_vram = 0.0
    try:
        peak_vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        pass

    loss_history = [
        {"step": h.get("step"), "loss": h.get("loss")}
        for h in trainer.state.log_history
        if "loss" in h
    ]
    summary = {
        "model_name": cfg.model_name,
        "lora_r": cfg.lora_r,
        "epochs": cfg.epochs,
        "trainable_params": trainable,
        "total_params": total,
        "final_train_loss": train_result.training_loss,
        "peak_vram_gb": peak_vram,
        "loss_history": loss_history,
        "output_dir": cfg.output_dir,
        "hardware": hw,
    }

    metrics_path = Path("results") / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
    existing[f"qlora_r{cfg.lora_r}"] = summary
    metrics_path.write_text(json.dumps(existing, indent=2))
    print(f"[train] wrote metrics to {metrics_path}")
    return summary


def _count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def load_config_from_yaml(path: str) -> TrainConfig:
    """Build a :class:`TrainConfig` from a YAML file (partial override)."""
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in TrainConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in known}
    return TrainConfig(**filtered)
