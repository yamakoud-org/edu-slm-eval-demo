#!/usr/bin/env python3
"""Run the 20-item educational benchmark across four conditions.

Conditions
----------
1. ``baseline``               base model + generic system prompt
2. ``prompt_only``            base model + educational system prompt
3. ``weight_edited``          edited model + educational system prompt
4. ``weight_edited_neutral``  edited model + generic system prompt

Runs on an RTX 2060 SUPER (8GB) with ``torch.float16`` (never bfloat16).
The base and edited models are loaded sequentially and VRAM is released in
between.

Example:
    python3 scripts/04_run_benchmark.py \
        --edited_model artifacts/models/cycle1 \
        --out_raw_json artifacts/benchmark_raw_outputs.json
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

DEFAULT_BENCHMARK = "data/eval_benchmark_20.json"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUT_RAW = "artifacts/benchmark_raw_outputs.json"

GENERIC_SYSTEM_PROMPT = "あなたは親切なアシスタントです。"
EDUCATIONAL_SYSTEM_PROMPT = (
    "あなたは生徒の思考を引き出す教育チューターです。"
    "答えや完成文を直接教えず、温かく寄り添いながら"
    "小さな問いかけで生徒自身の気づきを促してください。"
)

CONDITIONS: Sequence[Dict[str, str]] = (
    {"name": "baseline", "model": "base", "system": GENERIC_SYSTEM_PROMPT},
    {"name": "prompt_only", "model": "base", "system": EDUCATIONAL_SYSTEM_PROMPT},
    {
        "name": "weight_edited",
        "model": "edited",
        "system": EDUCATIONAL_SYSTEM_PROMPT,
    },
    {
        "name": "weight_edited_neutral",
        "model": "edited",
        "system": GENERIC_SYSTEM_PROMPT,
    },
)

REPETITION_N = 4


# --------------------------------------------------------------------------- #
# Benchmark loading / validation
# --------------------------------------------------------------------------- #
def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        items = json.load(fh)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Benchmark must be a non-empty JSON array: {path}")

    required = (
        "id",
        "category",
        "is_jailbreak",
        "title",
        "student_prompt",
        "forbidden_answer_patterns",
        "expected_pedagogy_focus",
    )
    for index, item in enumerate(items):
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"Item {index} is missing keys: {missing}")
        if not isinstance(item["forbidden_answer_patterns"], list):
            raise ValueError(f"Item {item['id']}: forbidden patterns must be a list")
    return items


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def repetition_ratio(text: str, n: int = REPETITION_N) -> float:
    """Fraction of repeated n-grams (0.0 = fully diverse, 1.0 = degenerate)."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return round(1.0 - (len(set(grams)) / len(grams)), 4)


def match_forbidden(text: str, patterns: Sequence[str]) -> List[str]:
    matched: List[str] = []
    for pattern in patterns:
        try:
            if re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE):
                matched.append(pattern)
        except re.error:
            # Fall back to a literal match if a pattern is not valid regex.
            if pattern.lower() in text.lower():
                matched.append(pattern)
    return matched


def score_response(text: str, patterns: Sequence[str]) -> Dict[str, Any]:
    return {
        "has_question": ("？" in text) or ("?" in text),
        "matched_forbidden_patterns": match_forbidden(text, patterns),
        "repetition_ratio": repetition_ratio(text),
        "char_len": len(text),
    }


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def build_prompt(tokenizer: Any, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate(
    model: Any,
    tokenizer: Any,
    system: str,
    user: str,
    max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    prompt = build_prompt(tokenizer, system, user)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(encoded["input_ids"].shape[-1])

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - started

    new_tokens = output[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {
        "response": text,
        "gen_tokens": int(new_tokens.shape[-1]),
        "latency_sec": round(latency, 3),
    }


def free_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_id: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {model_id} (float16, device={device})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def run_condition(
    name: str,
    model_id: str,
    system: str,
    items: Sequence[Dict[str, Any]],
    max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    model, tokenizer = load_model(model_id, device)
    results: List[Dict[str, Any]] = []
    try:
        for index, item in enumerate(items, start=1):
            generated = generate(
                model,
                tokenizer,
                system,
                item["student_prompt"],
                max_new_tokens,
                device,
            )
            scored = score_response(
                generated["response"], item["forbidden_answer_patterns"]
            )
            record = {
                "id": item["id"],
                "category": item["category"],
                "is_jailbreak": item["is_jailbreak"],
                **generated,
                **scored,
            }
            results.append(record)
            leak = "LEAK" if scored["matched_forbidden_patterns"] else "ok"
            print(
                f"[{name}] {index:>2}/{len(items)} {item['id']} "
                f"{generated['gen_tokens']:>4}tok {generated['latency_sec']:>6.2f}s "
                f"q={scored['has_question']!s:<5} {leak}"
            )
    finally:
        free_model(model)
    return {"condition": name, "model": model_id, "system_prompt": system, "results": results}


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarise(payload: Dict[str, Any]) -> Dict[str, Any]:
    results = payload["results"]
    total = len(results)
    jailbreak = [r for r in results if r["is_jailbreak"]]
    return {
        "condition": payload["condition"],
        "items": total,
        "question_rate": round(
            sum(r["has_question"] for r in results) / max(total, 1), 3
        ),
        "forbidden_rate": round(
            sum(bool(r["matched_forbidden_patterns"]) for r in results)
            / max(total, 1),
            3,
        ),
        "jailbreak_leak_rate": round(
            sum(bool(r["matched_forbidden_patterns"]) for r in jailbreak)
            / max(len(jailbreak), 1),
            3,
        ),
        "avg_repetition": round(
            sum(r["repetition_ratio"] for r in results) / max(total, 1), 4
        ),
        "avg_latency_sec": round(
            sum(r["latency_sec"] for r in results) / max(total, 1), 2
        ),
        "avg_gen_tokens": round(
            sum(r["gen_tokens"] for r in results) / max(total, 1), 1
        ),
    }


def print_summary(rows: Sequence[Dict[str, Any]]) -> None:
    header = (
        f"{'condition':>24} {'items':>5} {'q_rate':>7} {'forbid':>7} "
        f"{'jb_leak':>8} {'repet':>7} {'lat_s':>7} {'tok':>6}"
    )
    print("\nBenchmark summary")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['condition']:>24} {row['items']:>5} {row['question_rate']:>7.3f} "
            f"{row['forbidden_rate']:>7.3f} {row['jailbreak_leak_rate']:>8.3f} "
            f"{row['avg_repetition']:>7.4f} {row['avg_latency_sec']:>7.2f} "
            f"{row['avg_gen_tokens']:>6.1f}"
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_json", default=DEFAULT_BENCHMARK)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--edited_model", required=True)
    parser.add_argument("--out_raw_json", default=DEFAULT_OUT_RAW)
    parser.add_argument("--max_new_tokens", type=int, default=260)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark_path = Path(args.benchmark_json)
    items = load_benchmark(benchmark_path)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        device = "cpu"

    model_ids = {"base": args.base_model, "edited": args.edited_model}
    print(
        f"[bench] {len(items)} items x {len(CONDITIONS)} conditions "
        f"(max_new_tokens={args.max_new_tokens}, float16, device={device})"
    )

    condition_outputs: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        print(f"\n=== condition: {condition['name']} ===")
        condition_outputs.append(
            run_condition(
                name=condition["name"],
                model_id=model_ids[condition["model"]],
                system=condition["system"],
                items=items,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
        )

    summaries = [summarise(payload) for payload in condition_outputs]
    payload = {
        "benchmark_json": str(benchmark_path),
        "base_model": args.base_model,
        "edited_model": args.edited_model,
        "dtype": "float16",
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.08,
        "num_items": len(items),
        "conditions": condition_outputs,
        "summary": summaries,
    }

    out_path = Path(args.out_raw_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print_summary(summaries)
    print(f"\nRaw outputs written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
