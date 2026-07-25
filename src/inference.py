"""Inference wrappers for evaluation.

Provides:
  * :class:`TemplateBaselinePredictor` — GPU-free heuristic predictor.
  * :class:`HFPredictor` — loads a base model (optionally with a QLoRA adapter)
    and generates JSON agent contracts. Heavy imports are deferred.

Both expose ``predict(messages) -> (contract_dict, raw_text, latency_ms)``.
"""

from __future__ import annotations

import json
import re
import time

from .skill_router import baseline_predict


def _extract_question(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "user":
            match = re.search(r"Question:\s*(.*)\s*$", m["content"], flags=re.DOTALL)
            if match:
                return match.group(1).strip()
            return m["content"]
    return ""


def parse_contract(text: str) -> dict:
    """Extract the first JSON object from model text; tolerant of extra prose."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


class TemplateBaselinePredictor:
    """Heuristic, GPU-free predictor implementing the agent contract."""

    name = "template_baseline"

    def predict(self, messages: list[dict]) -> tuple[dict, str, float]:
        question = _extract_question(messages)
        t0 = time.perf_counter()
        contract = baseline_predict(question)
        latency = (time.perf_counter() - t0) * 1000.0
        raw = json.dumps(contract, ensure_ascii=False)
        return contract, raw, latency


class HFPredictor:
    """Hugging Face causal-LM predictor with optional QLoRA adapter.

    Parameters
    ----------
    model_name:
        Base model id (e.g. ``Qwen/Qwen2.5-3B-Instruct``).
    adapter_dir:
        Optional path to a trained PEFT adapter.
    few_shot:
        Optional list of ``(user, assistant)`` example pairs prepended as
        in-context demonstrations.
    load_in_4bit:
        Load base model in 4-bit (recommended for the 3B model on 32GB).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_dir: str | None = None,
        few_shot: list[tuple[str, str]] | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.name = f"qlora_adapter" if adapter_dir else "base_model"
        self.max_new_tokens = max_new_tokens
        self.few_shot = few_shot or []

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant = None
        if load_in_4bit and torch.cuda.is_available():
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        if adapter_dir:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()
        self._torch = torch

    def _build_messages(self, messages: list[dict]) -> list[dict]:
        if not self.few_shot:
            return messages
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        shots: list[dict] = []
        for u, a in self.few_shot:
            shots.append({"role": "user", "content": u})
            shots.append({"role": "assistant", "content": a})
        return system + shots + rest

    def predict(self, messages: list[dict]) -> tuple[dict, str, float]:
        torch = self._torch
        msgs = self._build_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = (time.perf_counter() - t0) * 1000.0
        gen = out[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return parse_contract(raw), raw, latency
