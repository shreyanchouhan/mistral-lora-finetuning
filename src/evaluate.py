"""Evaluate base vs fine-tuned model on the held-out eval set.

Metric: simple instruction-following score = average token-overlap
(F1-style unigram overlap) between generated response and reference.

This is a lightweight proxy — for a real run, swap in a benchmark like
MT-Bench / IFEval. The script structure (base vs LoRA-adapter delta)
is the part that matters.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import get_formatter, read_jsonl  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


def overlap_f1(pred: str, ref: str) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    n = sum(common.values())
    if n == 0:
        return 0.0
    precision = n / len(p)
    recall = n / len(r)
    return 2 * precision * recall / (precision + recall)


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def load_base(model_name: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def attach_adapter(base, adapter_path: str):
    return PeftModel.from_pretrained(base, adapter_path)


def evaluate(model, tokenizer, records: list[dict], model_name: str, limit: int) -> float:
    fmt = get_formatter(model_name)
    scores: list[float] = []
    for rec in records[:limit]:
        prompt = fmt(rec["instruction"], rec.get("input", ""), output=None)
        gen = generate(model, tokenizer, prompt)
        scores.append(overlap_f1(gen, rec["output"]))
    return sum(scores) / max(len(scores), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--adapter", default="outputs/lora-adapter")
    p.add_argument("--eval_path", default="data/eval.jsonl")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--out", default="outputs/eval_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.eval_path)
    print(f"[eval] Loaded {len(records)} eval records; using first {args.limit}")

    print("[eval] Loading base model ...")
    base, tok = load_base(args.model)
    base_score = evaluate(base, tok, records, args.model, args.limit)
    print(f"[eval] BASE  token-overlap F1: {base_score:.4f}")

    print("[eval] Attaching LoRA adapter ...")
    tuned = attach_adapter(base, args.adapter)
    tuned.eval()
    tuned_score = evaluate(tuned, tok, records, args.model, args.limit)
    print(f"[eval] TUNED token-overlap F1: {tuned_score:.4f}")

    delta = (tuned_score - base_score) / max(base_score, 1e-9) * 100
    print(f"[eval] Relative improvement: {delta:+.2f}%")

    report = {
        "model": args.model,
        "adapter": args.adapter,
        "n_eval": args.limit,
        "base_f1": base_score,
        "tuned_f1": tuned_score,
        "relative_improvement_pct": delta,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] Wrote report -> {args.out}")


if __name__ == "__main__":
    main()
