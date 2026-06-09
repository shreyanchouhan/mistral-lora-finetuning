"""LoRA / QLoRA fine-tuning for causal LMs (Mistral-7B, TinyLlama, etc.).

Uses transformers `Trainer` + PEFT directly (no TRL dependency, for
maximum cross-version stability).

Features
--------
* LoRA adapters on attention projections via `peft`
* Optional 4-bit NF4 quantization (QLoRA) when `--use_4bit` is set and
  `bitsandbytes` is importable
* Mixed-precision training (bf16 when available, else fp16)
* Gradient checkpointing for activation-memory savings
* Chat-template formatting per model family

Usage
-----
    # Local smoke test on CPU / small GPU
    python src/train.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max_steps 50

    # Mistral-7B with QLoRA (needs bitsandbytes, ≥12GB VRAM)
    python src/train.py --model mistralai/Mistral-7B-Instruct-v0.2 --use_4bit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import get_default_target_modules, get_formatter, load_yaml  # noqa: E402


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    p.add_argument("--model", default=None)
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--train_path", default=None)
    p.add_argument("--eval_path", default=None)
    p.add_argument("--max_seq_length", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=None)
    p.add_argument("--target_modules", nargs="+", default=None,
                   help="Override LoRA target modules (auto-detected from model name otherwise)")
    return p.parse_args()


# ----------------------------- Model -----------------------------

def load_model_and_tokenizer(cfg: dict, model_name: str, use_4bit: bool):
    quantization_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg["model"].get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=getattr(
                    torch, cfg["model"].get("bnb_4bit_compute_dtype", "bfloat16")
                ),
                bnb_4bit_use_double_quant=True,
            )
            print("[train] QLoRA 4-bit NF4 enabled.")
        except Exception as e:
            print(f"[train] 4-bit unavailable ({e}); using fp16/bf16.")
            quantization_config = None

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = dict(low_cpu_mem_usage=True)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = dtype
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    elif cfg["training"].get("gradient_checkpointing", True) and torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    return model, tokenizer


# ----------------------------- Data -----------------------------

def build_dataset(train_path, eval_path, model_name, tokenizer, max_seq_length):
    fmt = get_formatter(model_name)

    def to_text(rec):
        return {"text": fmt(rec["instruction"], rec.get("input", ""), rec["output"])}

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        return out

    ds = load_dataset("json", data_files={"train": train_path, "eval": eval_path})
    ds = ds.map(to_text, remove_columns=ds["train"].column_names)
    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds


# ----------------------------- Main -----------------------------

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    model_name = args.model or cfg["model"]["name"]
    use_4bit = args.use_4bit or cfg["model"].get("use_4bit", False)
    output_dir = args.output_dir or cfg["training"]["output_dir"]
    train_path = args.train_path or cfg["data"]["train_path"]
    eval_path = args.eval_path or cfg["data"]["eval_path"]
    max_seq_length = args.max_seq_length or cfg["training"]["max_seq_length"]
    batch_size = args.per_device_train_batch_size or cfg["training"]["per_device_train_batch_size"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"[train] model={model_name}  4bit={use_4bit}  out={output_dir}")
    print(f"[train] cuda={torch.cuda.is_available()}  "
          f"bf16={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")

    model, tokenizer = load_model_and_tokenizer(cfg, model_name, use_4bit)

    target_modules = args.target_modules or get_default_target_modules(model_name)
    print(f"[train] LoRA target modules: {target_modules}")
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=target_modules,
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = build_dataset(train_path, eval_path, model_name, tokenizer, max_seq_length)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    ta_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"] and torch.cuda.is_available(),
        optim=("paged_adamw_8bit" if use_4bit else "adamw_torch"),
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        report_to=cfg["training"]["report_to"],
        seed=cfg["training"]["seed"],
        remove_unused_columns=False,
    )
    if args.max_steps:
        ta_kwargs["max_steps"] = args.max_steps

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        data_collator=collator,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train] Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
