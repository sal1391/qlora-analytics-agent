#!/usr/bin/env python
"""Stage 04: QLoRA fine-tuning entry point.

Reads a YAML config (configs/train_qlora_3b.yaml by default) and applies CLI
overrides. Auto-detects CUDA and adapts batch size. Fails gracefully with a
helpful message on CPU-only machines.

Examples
--------
    python scripts/04_train_qlora.py --config configs/train_qlora_3b.yaml
    python scripts/04_train_qlora.py --lora_r 8 --epochs 2
    python scripts/04_train_qlora.py --smoke   # 50 samples, 1 epoch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.train import TrainConfig, detect_hardware, load_config_from_yaml, train_qlora  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning")
    parser.add_argument("--config", type=str, default="configs/train_qlora_3b.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--lora_r", type=int, default=None, choices=[8, 16, 32])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training samples (smoke tests)")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run: 50 samples, 1 epoch")
    args = parser.parse_args()

    cfg_path = REPO_ROOT / args.config
    cfg = load_config_from_yaml(str(cfg_path)) if cfg_path.exists() else TrainConfig()

    if args.model:
        cfg.model_name = args.model
    if args.lora_r:
        cfg.lora_r = args.lora_r
        cfg.lora_alpha = args.lora_r * 2
        cfg.output_dir = f"results/adapters/qlora_r{args.lora_r}"
    if args.epochs:
        cfg.epochs = args.epochs
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_samples:
        cfg.max_samples = args.max_samples
    if args.smoke:
        cfg.max_samples = 50
        cfg.epochs = 1
        cfg.logging_steps = 1

    print(f"[04] Hardware: {detect_hardware()}")
    print(f"[04] Config: model={cfg.model_name} lora_r={cfg.lora_r} epochs={cfg.epochs} "
          f"output={cfg.output_dir} max_samples={cfg.max_samples}")

    try:
        summary = train_qlora(cfg)
    except RuntimeError as exc:
        print(f"\n[04] Training unavailable: {exc}")
        sys.exit(2)

    print(f"[04] Done. Final train loss={summary['final_train_loss']:.4f} "
          f"trainable={summary['trainable_params']:,} peak_vram={summary['peak_vram_gb']}GB")


if __name__ == "__main__":
    main()
