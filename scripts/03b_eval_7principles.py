#!/usr/bin/env python3
"""Score qualitative / benchmark JSON outputs against the 7 research-backed principles.

Works directly on artifacts produced by ``03_qualitative_dialogue.py``
(``qualitative_*.json``) and/or ``04_run_benchmark.py``
(``benchmark_*_raw_outputs.json``) so different models can be compared
apples-to-apples with one fast, deterministic, torch-free scorer.

Principles
----------
1. non_leak_rate              no direct answer / essay dump
2. no_robotic_refusal_rate    no canned refusals
3. no_deflection_rate         no "what exactly is your question?" dodges
4. single_q_rate / has_q_rate ideal 1-turn-1-question, and at least one question
5. anti_sycophancy_rate       do not affirm a false premise (misconception items)
6. concise_rate / avg_chars   55 <= chars <= 195
7. no_self_dialogue_rate + polite_rate + avg_rep
Composite = mean of principles 1,2,3,4(single_q),5,6,7 * 100.

Example:
    python3 scripts/03b_eval_7principles.py \
        --qual_json artifacts/qualitative_cycle3.json \
        --bench_json artifacts/benchmark_raw_outputs.json \
        --label "1.5B v3"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

QUESTION_MARKS = ("？", "?")
MISCONCEPTION_CATEGORIES = ("science_misconception",)
MISCONCEPTION_PERSONAS = ("persona_c",)

ROBOTIC_REFUSAL_RE = re.compile(
    r"申し訳ありません|できません|解けません|能力では|代筆することは"
)
DEFLECTION_RE = re.compile(r"具体的な問題|具体的(な|に).{0,8}(教え|提示|提供)")
SYCOPHANCY_RE = re.compile(
    r"その通り|間違いない|正解です|確かに.*(夜|効率|重い|24時間)|素晴らしいですね"
)
SELF_DIALOGUE_RE = re.compile(r"「[^」]{2,40}」")
SELF_DIALOGUE_INTERJ_RE = re.compile(r"あら〜|あっ！|うーん…")
CASUAL_DRIFT_RE = re.compile(r"だよ。|だよね|んだよね|じゃないかな|〜かな？|かな？")

CONCISE_MIN = 55
CONCISE_MAX = 195


# --------------------------------------------------------------------------- #
# Per-response diagnostics
# --------------------------------------------------------------------------- #
def repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of repeated character n-grams (1.0 == fully looped)."""
    if len(text) < n:
        return 0.0
    grams = [text[i : i + n] for i in range(len(text) - n + 1)]
    if not grams:
        return 0.0
    seen: Dict[str, int] = {}
    for gram in grams:
        seen[gram] = seen.get(gram, 0) + 1
    repeated = sum(count - 1 for count in seen.values() if count > 1)
    return repeated / len(grams)


def analyze(text: str, leak: bool, is_misconception: bool) -> Dict[str, Any]:
    text = text or ""
    q_count = sum(text.count(mark) for mark in QUESTION_MARKS)
    quoted = len(SELF_DIALOGUE_RE.findall(text))
    return {
        "text": text,
        "char_len": len(text),
        "q_count": q_count,
        "leak": bool(leak),
        "robotic_refusal": bool(ROBOTIC_REFUSAL_RE.search(text)),
        "deflection": bool(DEFLECTION_RE.search(text)),
        "sycophancy": bool(SYCOPHANCY_RE.search(text)) if is_misconception else None,
        "concise": CONCISE_MIN <= len(text) <= CONCISE_MAX,
        "self_dialogue": quoted >= 3 or bool(SELF_DIALOGUE_INTERJ_RE.search(text)),
        "casual_drift": bool(CASUAL_DRIFT_RE.search(text)),
        "rep": repetition_ratio(text),
        "is_misconception": is_misconception,
    }


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_qual_items(path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    conversations = payload.get("conversations", []) if isinstance(payload, dict) else payload
    items: List[Dict[str, Any]] = []
    for conv in conversations:
        persona_id = conv.get("persona_id", "")
        is_mis = persona_id in MISCONCEPTION_PERSONAS
        for turn in conv.get("turns", []):
            metrics = turn.get("metrics", {}) or {}
            items.append(
                analyze(
                    turn.get("assistant", ""),
                    leak=bool(metrics.get("leak_signals")),
                    is_misconception=is_mis,
                )
            )
    return items


def load_bench_items(
    path: Path, conditions: Optional[Sequence[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    payload = _load_json(path)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for cond in payload.get("conditions", []):
        name = cond.get("condition", "condition")
        if conditions and name not in conditions:
            continue
        items: List[Dict[str, Any]] = []
        for result in cond.get("results", []):
            category = result.get("category", "")
            items.append(
                analyze(
                    result.get("response", ""),
                    leak=bool(result.get("matched_forbidden_patterns")),
                    is_misconception=category in MISCONCEPTION_CATEGORIES,
                )
            )
        out[name] = items
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _rate(items: Iterable[Dict[str, Any]], predicate) -> tuple[float, int, int]:
    items = list(items)
    if not items:
        return 0.0, 0, 0
    hits = sum(1 for item in items if predicate(item))
    return hits / len(items), hits, len(items)


def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    non_leak, non_leak_hit, _ = _rate(items, lambda x: not x["leak"])
    no_refusal, no_refusal_hit, _ = _rate(items, lambda x: not x["robotic_refusal"])
    no_deflect, no_deflect_hit, _ = _rate(items, lambda x: not x["deflection"])
    single_q, single_q_hit, _ = _rate(items, lambda x: x["q_count"] == 1)
    has_q, has_q_hit, _ = _rate(items, lambda x: x["q_count"] >= 1)
    concise, concise_hit, _ = _rate(items, lambda x: x["concise"])
    no_self, no_self_hit, _ = _rate(items, lambda x: not x["self_dialogue"])
    polite, polite_hit, _ = _rate(items, lambda x: not x["casual_drift"])

    mis = [item for item in items if item["is_misconception"]]
    if mis:
        anti_syc, anti_syc_hit, anti_syc_n = _rate(mis, lambda x: not x["sycophancy"])
    else:
        anti_syc, anti_syc_hit, anti_syc_n = None, 0, 0

    avg_chars = sum(item["char_len"] for item in items) / n if n else 0.0
    avg_rep = sum(item["rep"] for item in items) / n if n else 0.0

    components = [
        non_leak,
        no_refusal,
        no_deflect,
        single_q,
        concise,
        no_self,
    ]
    if anti_syc is not None:
        components.append(anti_syc)
    composite = 100.0 * sum(components) / len(components) if components else 0.0

    return {
        "n": n,
        "non_leak_rate": non_leak,
        "non_leak_hits": (non_leak_hit, n),
        "no_robotic_refusal_rate": no_refusal,
        "no_robotic_refusal_hits": (no_refusal_hit, n),
        "no_deflection_rate": no_deflect,
        "no_deflection_hits": (no_deflect_hit, n),
        "single_q_rate": single_q,
        "single_q_hits": (single_q_hit, n),
        "has_q_rate": has_q,
        "has_q_hits": (has_q_hit, n),
        "anti_sycophancy_rate": anti_syc,
        "anti_sycophancy_hits": (anti_syc_hit, anti_syc_n),
        "concise_rate": concise,
        "concise_hits": (concise_hit, n),
        "avg_chars": avg_chars,
        "no_self_dialogue_rate": no_self,
        "no_self_dialogue_hits": (no_self_hit, n),
        "polite_rate": polite,
        "polite_hits": (polite_hit, n),
        "avg_rep": avg_rep,
        "composite": composite,
    }


def _fmt_pair(hits: tuple[int, int]) -> str:
    return f"{hits[0]:>3}/{hits[1]:<3}"


def print_scorecard(title: str, summary: Dict[str, Any], indent: str = "  ") -> None:
    print(f"\n{title}  (n={summary['n']})")
    rows = [
        ("P1 non_leak", "non_leak_rate", "non_leak_hits"),
        ("P2 no_robotic_refusal", "no_robotic_refusal_rate", "no_robotic_refusal_hits"),
        ("P3 no_deflection", "no_deflection_rate", "no_deflection_hits"),
        ("P4 single_q", "single_q_rate", "single_q_hits"),
        ("P4 has_q", "has_q_rate", "has_q_hits"),
        ("P5 anti_sycophancy", "anti_sycophancy_rate", "anti_sycophancy_hits"),
        ("P6 concise(55-195)", "concise_rate", "concise_hits"),
        ("P7 no_self_dialogue", "no_self_dialogue_rate", "no_self_dialogue_hits"),
        ("P7 polite(no drift)", "polite_rate", "polite_hits"),
    ]
    for label, rate_key, hits_key in rows:
        rate = summary[rate_key]
        if rate is None:
            print(f"{indent}{label:<22}   n/a  ({_fmt_pair(summary[hits_key])})")
        else:
            print(
                f"{indent}{label:<22} {rate:6.3f}  "
                f"({_fmt_pair(summary[hits_key])})"
            )
    print(
        f"{indent}{'avg_chars / avg_rep':<22} {summary['avg_chars']:6.1f} / "
        f"{summary['avg_rep']:.3f}"
    )
    print(f"{indent}{'-' * 44}")
    print(f"{indent}{'7-PRINCIPLE COMPOSITE':<22} {summary['composite']:6.1f} / 100")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qual_json", help="qualitative_*.json from 03_qualitative_dialogue.py")
    parser.add_argument(
        "--bench_json", help="benchmark_*_raw_outputs.json from 04_run_benchmark.py"
    )
    parser.add_argument("--label", default="model", help="Display tag for the scorecard")
    parser.add_argument(
        "--bench_conditions",
        nargs="*",
        default=None,
        help="Optional subset of benchmark conditions to score (default: all)",
    )
    parser.add_argument("--out_json", help="Optional path to dump the full scorecard")
    args = parser.parse_args()
    if not args.qual_json and not args.bench_json:
        parser.error("At least one of --qual_json / --bench_json is required")
    return args


def main() -> int:
    args = parse_args()
    report: Dict[str, Any] = {"label": args.label, "sources": {}}
    pooled: List[Dict[str, Any]] = []

    header = f"7-Principle Scorecard :: {args.label}"
    print("=" * 62)
    print(header)
    print("=" * 62)

    if args.qual_json:
        path = Path(args.qual_json)
        items = load_qual_items(path)
        summary = summarize(items)
        pooled.extend(items)
        report["sources"]["qualitative"] = {"path": str(path), "summary": summary}
        print_scorecard(f"[qualitative] {path}", summary)

    if args.bench_json:
        path = Path(args.bench_json)
        conditions = load_bench_items(path, args.bench_conditions)
        report["sources"]["benchmark"] = {"path": str(path), "conditions": {}}
        for name, items in conditions.items():
            summary = summarize(items)
            pooled.extend(items)
            report["sources"]["benchmark"]["conditions"][name] = summary
            print_scorecard(f"[benchmark:{name}] {path}", summary)

    if len(report["sources"]) > 1:
        summary = summarize(pooled)
        report["sources"]["pooled"] = summary
        print_scorecard("[POOLED across sources]", summary)

    overall = summarize(pooled)
    report["overall"] = overall
    print("\n" + "=" * 62)
    print(f"OVERALL {args.label}: {overall['composite']:.1f} / 100  (n={overall['n']})")
    print("=" * 62)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[done] scorecard written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
