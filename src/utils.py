"""Shared helpers."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def format_mistral(instruction: str, inp: str, output: str | None = None) -> str:
    """Mistral-Instruct chat format.

    [INST] {prompt} [/INST] {response}</s>
    """
    prompt = instruction if not inp else f"{instruction}\n\n{inp}"
    if output is None:
        return f"<s>[INST] {prompt} [/INST]"
    return f"<s>[INST] {prompt} [/INST] {output}</s>"


def format_tinyllama(instruction: str, inp: str, output: str | None = None) -> str:
    """TinyLlama-Chat ChatML-ish format."""
    user = instruction if not inp else f"{instruction}\n\n{inp}"
    sys_msg = "You are a helpful technical assistant."
    if output is None:
        return (
            f"<|system|>\n{sys_msg}</s>\n"
            f"<|user|>\n{user}</s>\n"
            f"<|assistant|>\n"
        )
    return (
        f"<|system|>\n{sys_msg}</s>\n"
        f"<|user|>\n{user}</s>\n"
        f"<|assistant|>\n{output}</s>"
    )


def format_plain(instruction: str, inp: str, output: str | None = None) -> str:
    """Generic instruction format for non-chat models (e.g. GPT-2)."""
    user = instruction if not inp else f"{instruction}\n\n{inp}"
    if output is None:
        return f"### Instruction:\n{user}\n\n### Response:\n"
    return f"### Instruction:\n{user}\n\n### Response:\n{output}"


def get_formatter(model_name: str):
    n = model_name.lower()
    if "tinyllama" in n:
        return format_tinyllama
    if "mistral" in n or "llama" in n:
        return format_mistral
    return format_plain


def get_default_target_modules(model_name: str) -> list[str]:
    """Pick reasonable LoRA target modules per model family."""
    n = model_name.lower()
    if "gpt2" in n or "distilgpt" in n:
        return ["c_attn"]
    if "pythia" in n or "gpt-neox" in n:
        return ["query_key_value"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]
