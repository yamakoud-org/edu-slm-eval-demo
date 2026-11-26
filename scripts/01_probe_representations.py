#!/usr/bin/env python3
"""Step 1: probe Qwen2.5-1.5B-Instruct representations for pedagogy vs. sycophancy.

Collects, for every transformer layer:
  * hidden-state pools over assistant response tokens (positive / negative)
  * input pools to ``mlp.down_proj`` and ``self_attn.o_proj``
Then derives a pedagogical mean-difference direction, a sycophancy principal
component, and layer diagnostics used later for weight surgery.

Hardware note: RTX 2060 SUPER (Turing, 8GB) -> torch.float16 ONLY.
bfloat16 is not supported on Turing and must never be used here.

Run with:
  /home/yamakoud/Project/experiment/.venv/bin/python scripts/01_probe_representations.py
"""

from __future__ import annotations

import argparse
import json
import math
import py_compile
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "あなたは親切なアシスタントです。"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe representation directions for educational weight surgery."
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--pairs_path", type=str, default="data/contrastive_pairs.json")
    parser.add_argument("--out_pt", type=str, default="artifacts/probing_vectors.pt")
    parser.add_argument("--out_json", type=str, default="data/probing_stats.json")
    parser.add_argument("--max_resp_tokens", type=int, default=32)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_pairs(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        pairs = json.load(fh)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path}: expected a non-empty JSON array of pairs")
    required = {
        "id",
        "category",
        "student_prompt",
        "negative_response",
        "positive_response",
    }
    for item in pairs:
        missing = required - set(item)
        if missing:
            raise ValueError(f"{path}: item {item.get('id', '?')} missing keys {sorted(missing)}")
    return pairs


def build_chat_ids(
    tokenizer: Any,
    user_prompt: str,
    assistant_response: str,
) -> tuple[torch.Tensor, int]:
    """Tokenize prompt+response and return (input_ids, resp_start).

    ``resp_start`` is the index of the first assistant response token, obtained by
    tokenizing the prompt alone with ``add_generation_prompt=True`` and checking
    that it is an exact prefix of the full sequence.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt_ids: List[int] = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids: List[int] = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": assistant_response}],
        tokenize=True,
        add_generation_prompt=False,
    )

    resp_start = len(prompt_ids)
    if full_ids[:resp_start] != prompt_ids:
        # Chat templates can differ by trailing whitespace / newline handling.
        # Fall back to the longest common prefix so the window stays aligned.
        resp_start = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            resp_start += 1
        print(
            f"[warn] chat-template prefix mismatch; using common prefix "
            f"resp_start={resp_start} (prompt_len={len(prompt_ids)})",
            file=sys.stderr,
        )

    if resp_start >= len(full_ids):
        raise ValueError("assistant response produced no tokens after the prompt")
    return torch.tensor([full_ids], dtype=torch.long), resp_start


def cohen_d(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cohen's d (a minus b) with pooled standard deviation."""
    if a.numel() < 2 or b.numel() < 2:
        return float("nan")
    na, nb = a.numel(), b.numel()
    var_a = a.var(unbiased=True) if na > 1 else torch.tensor(0.0)
    var_b = b.var(unbiased=True) if nb > 1 else torch.tensor(0.0)
    pooled = torch.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / max(na + nb - 2, 1))
    diff = a.mean() - b.mean()
    denom = pooled + 1e-8
    return float(diff / denom)


def make_folds(n: int, k: int, seed: int = 0) -> List[List[int]]:
    """Deterministic contiguous-ish folds (no sklearn dependency)."""
    k = max(1, min(k, n))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    folds: List[List[int]] = [[] for _ in range(k)]
    for idx, item in enumerate(perm):
        folds[idx % k].append(item)
    return folds


def cv_direction_accuracy(
    h_pos: torch.Tensor,
    h_neg: torch.Tensor,
    k: int = 5,
    seed: int = 0,
) -> float:
    """5-fold CV accuracy of classification along the train-fold mean-diff direction.

    The direction AND decision threshold are fit strictly on train folds.
    """
    n = h_pos.shape[0]
    x = torch.cat([h_pos, h_neg], dim=0)  # [2N, d]
    y = torch.cat([torch.ones(n), torch.zeros(n)])  # 1 = positive
    folds = make_folds(x.shape[0], k, seed=seed)

    accs: List[float] = []
    for fold in folds:
        test_idx = torch.tensor(sorted(fold), dtype=torch.long)
        train_mask = torch.ones(x.shape[0], dtype=torch.bool)
        train_mask[test_idx] = False
        if not train_mask.any() or test_idx.numel() == 0:
            continue

        x_tr, y_tr = x[train_mask], y[train_mask]
        pos_tr = x_tr[y_tr == 1]
        neg_tr = x_tr[y_tr == 0]
        direction = pos_tr.mean(0) - neg_tr.mean(0)
        direction = direction / (direction.norm() + 1e-8)

        proj_tr = x_tr @ direction
        thr = 0.5 * (proj_tr[y_tr == 1].mean() + proj_tr[y_tr == 0].mean())

        proj_te = x[test_idx] @ direction
        pred = (proj_te > thr).float()
        accs.append(float((pred == y[test_idx]).float().mean()))

    if not accs:
        return float("nan")
    return sum(accs) / len(accs)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


# --------------------------------------------------------------------------- #
# Capture machinery
# --------------------------------------------------------------------------- #
class ResponseCapture:
    """Forward hooks that mean-pool activations over the response token window.

    Keeps only pooled vectors per layer so that memory stays constant in the
    number of pairs (GPU stays well under 8GB).
    """

    def __init__(self, model: torch.nn.Module, max_resp_tokens: int) -> None:
        self.model = model
        self.max_resp_tokens = max_resp_tokens
        self.layers = model.model.layers
        self.start: int = 0
        self.window: int = 0
        self._handles: List[Any] = []

        n_layers = len(self.layers)
        # Per-forward scratch buffers, filled by hooks.
        self.h_pool: List[Optional[torch.Tensor]] = [None] * n_layers
        self.h_norm_sum: List[float] = [0.0] * n_layers
        self.h_norm_cnt: List[int] = [0] * n_layers
        self.h_last_prompt: List[Optional[torch.Tensor]] = [None] * n_layers
        self.z_mlp_pool: List[Optional[torch.Tensor]] = [None] * n_layers
        self.z_attn_pool: List[Optional[torch.Tensor]] = [None] * n_layers

    # -- hook registration --------------------------------------------------- #
    def register(self) -> None:
        for idx, layer in enumerate(self.layers):
            self._handles.append(
                layer.register_forward_hook(self._make_layer_hook(idx))
            )
            self._handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    self._make_pre_hook(idx, "mlp")
                )
            )
            self._handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    self._make_pre_hook(idx, "attn")
                )
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # -- per-forward state --------------------------------------------------- #
    def reset(self, resp_start: int, seq_len: int) -> None:
        self.start = resp_start
        self.window = max(0, min(self.max_resp_tokens, seq_len - resp_start))
        n_layers = len(self.layers)
        self.h_pool = [None] * n_layers
        self.h_norm_sum = [0.0] * n_layers
        self.h_norm_cnt = [0] * n_layers
        self.h_last_prompt = [None] * n_layers
        self.z_mlp_pool = [None] * n_layers
        self.z_attn_pool = [None] * n_layers

    def _response_slice(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, ...] -> [window, ...] float32 for the assistant response tokens."""
        if x.dim() == 3:  # [B, S, D]
            x = x[0]
        end = self.start + self.window
        return x[self.start:end].to(torch.float32)

    def _make_layer_hook(self, idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            win = self._response_slice(hidden)
            if win.numel() == 0:
                return
            self.h_pool[idx] = win.mean(dim=0)
            self.h_norm_sum[idx] += float(win.norm(dim=-1).sum())
            self.h_norm_cnt[idx] += int(win.shape[0])
            if self.start > 0:
                self.h_last_prompt[idx] = hidden[0, self.start - 1].to(torch.float32)

        return hook

    def _make_pre_hook(self, idx: int, which: str):
        def hook(_module, inputs):
            if not inputs:
                return
            win = self._response_slice(inputs[0])
            if win.numel() == 0:
                return
            pooled = win.mean(dim=0)
            if which == "mlp":
                self.z_mlp_pool[idx] = pooled
            else:
                self.z_attn_pool[idx] = pooled

        return hook

    # -- forward pass -------------------------------------------------------- #
    @torch.no_grad()
    def run(self, input_ids: torch.Tensor, resp_start: int) -> Dict[str, Any]:
        self.reset(resp_start, input_ids.shape[1])
        self.model(
            input_ids=input_ids.to(self.model.device),
            attention_mask=torch.ones_like(input_ids, device=self.model.device),
            use_cache=False,
        )
        return {
            "h_pool": self.h_pool,
            "h_norm_sum": list(self.h_norm_sum),
            "h_norm_cnt": list(self.h_norm_cnt),
            "h_last_prompt": self.h_last_prompt,
            "z_mlp_pool": self.z_mlp_pool,
            "z_attn_pool": self.z_attn_pool,
        }


# --------------------------------------------------------------------------- #
# Model inspection
# --------------------------------------------------------------------------- #
def describe_model(model: torch.nn.Module) -> Dict[str, Any]:
    config = model.config
    layer0 = model.model.layers[0]
    o_proj = layer0.self_attn.o_proj
    down_proj = layer0.mlp.down_proj

    info = {
        "model_type": getattr(config, "model_type", "unknown"),
        "num_hidden_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_attention_heads": int(getattr(config, "num_attention_heads", -1)),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", -1)),
        "o_proj_has_bias": o_proj.bias is not None,
        "down_proj_has_bias": down_proj.bias is not None,
        "o_proj_shape": list(o_proj.weight.shape),
        "down_proj_shape": list(down_proj.weight.shape),
        "torch_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
    }
    return info


def print_model_info(info: Dict[str, Any]) -> None:
    print("=" * 68)
    print("MODEL ARCHITECTURE")
    print("=" * 68)
    for key in (
        "model_type",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "o_proj_has_bias",
        "down_proj_has_bias",
        "o_proj_shape",
        "down_proj_shape",
        "torch_dtype",
    ):
        print(f"  {key:22s}: {info[key]}")
    print("=" * 68)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Syntax self-check (also handy when the file is edited in place).
    py_compile.compile(__file__, doraise=True)
    print(f"[ok] py_compile: {Path(__file__).name}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (RTX 2060 SUPER / Turing). Nothing to do on CPU.")

    pairs = load_pairs(args.pairs_path)
    n_pairs = len(pairs)
    categories = [p["category"] for p in pairs]
    print(f"[ok] loaded {n_pairs} contrastive pairs from {args.pairs_path}")
    print(f"     categories: {sorted(set(categories))}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,  # Turing: fp16 only, never bfloat16
        device_map="cuda",
    )
    model.eval()
    print(f"[ok] loaded {args.model_id} on cuda as float16")

    info = describe_model(model)
    print_model_info(info)

    n_layers = info["num_hidden_layers"]
    d_model = info["hidden_size"]
    d_inter = info["intermediate_size"]

    capture = ResponseCapture(model, max_resp_tokens=args.max_resp_tokens)
    capture.register()

    # Accumulators: [N, L, dim]
    h_pos = torch.zeros(n_pairs, n_layers, d_model, dtype=torch.float32)
    h_neg = torch.zeros(n_pairs, n_layers, d_model, dtype=torch.float32)
    z_mlp_pos = torch.zeros(n_pairs, n_layers, d_inter, dtype=torch.float32)
    z_mlp_neg = torch.zeros(n_pairs, n_layers, d_inter, dtype=torch.float32)
    z_attn_pos = torch.zeros(n_pairs, n_layers, d_model, dtype=torch.float32)
    z_attn_neg = torch.zeros(n_pairs, n_layers, d_model, dtype=torch.float32)
    h_last_prompt_pos = torch.zeros(n_pairs, n_layers, d_model, dtype=torch.float32)
    h_last_prompt_neg = torch.zeros(n_pairs, n_layers, d_model, dtype=torch.float32)

    # Residual norms accumulated across both polarities.
    resid_norm_sum = torch.zeros(n_layers, dtype=torch.float64)
    resid_norm_cnt = torch.zeros(n_layers, dtype=torch.long)

    try:
        for i, pair in enumerate(pairs):
            for polarity, key in (("pos", "positive_response"), ("neg", "negative_response")):
                input_ids, resp_start = build_chat_ids(
                    tokenizer, pair["student_prompt"], pair[key]
                )
                out = capture.run(input_ids, resp_start)

                for layer_idx in range(n_layers):
                    hp = out["h_pool"][layer_idx]
                    zm = out["z_mlp_pool"][layer_idx]
                    za = out["z_attn_pool"][layer_idx]
                    hl = out["h_last_prompt"][layer_idx]
                    if hp is None or zm is None or za is None:
                        raise RuntimeError(
                            f"missing activation for pair={pair['id']} "
                            f"polarity={polarity} layer={layer_idx}"
                        )
                    if polarity == "pos":
                        h_pos[i, layer_idx] = hp.cpu()
                        z_mlp_pos[i, layer_idx] = zm.cpu()
                        z_attn_pos[i, layer_idx] = za.cpu()
                        if hl is not None:
                            h_last_prompt_pos[i, layer_idx] = hl.cpu()
                    else:
                        h_neg[i, layer_idx] = hp.cpu()
                        z_mlp_neg[i, layer_idx] = zm.cpu()
                        z_attn_neg[i, layer_idx] = za.cpu()
                        if hl is not None:
                            h_last_prompt_neg[i, layer_idx] = hl.cpu()

                    resid_norm_sum[layer_idx] += out["h_norm_sum"][layer_idx]
                    if polarity == "neg":
                        resid_norm_cnt[layer_idx] += out["h_norm_cnt"][layer_idx]

            if (i + 1) % 8 == 0 or (i + 1) == n_pairs:
                print(f"[probe] {i + 1}/{n_pairs} pairs captured")
    finally:
        capture.remove()

    # ------------------------------------------------------------------ #
    # Per-layer directions and diagnostics
    # ------------------------------------------------------------------ #
    delta_h = torch.zeros(n_layers, d_model, dtype=torch.float32)
    delta_h_unit = torch.zeros(n_layers, d_model, dtype=torch.float32)
    v_syc = torch.zeros(n_layers, d_model, dtype=torch.float32)
    mu_pos = torch.zeros(n_layers, d_model, dtype=torch.float32)
    mu_neg = torch.zeros(n_layers, d_model, dtype=torch.float32)
    mu_z_mlp = torch.zeros(n_layers, d_inter, dtype=torch.float32)
    mu_z_mlp_pos = torch.zeros(n_layers, d_inter, dtype=torch.float32)
    mu_z_mlp_neg = torch.zeros(n_layers, d_inter, dtype=torch.float32)
    mu_z_attn = torch.zeros(n_layers, d_model, dtype=torch.float32)
    mu_z_attn_pos = torch.zeros(n_layers, d_model, dtype=torch.float32)
    mu_z_attn_neg = torch.zeros(n_layers, d_model, dtype=torch.float32)
    resid_norm = torch.zeros(n_layers, dtype=torch.float32)

    layer_stats: List[Dict[str, Any]] = []
    cat_index = {
        "math_jailbreak": [j for j, c in enumerate(categories) if c == "math_jailbreak"],
        "adversarial_override": [
            j for j, c in enumerate(categories) if c == "adversarial_override"
        ],
    }

    for layer_idx in range(n_layers):
        hp = h_pos[:, layer_idx, :]  # [N, d]
        hn = h_neg[:, layer_idx, :]

        m_pos = hp.mean(dim=0)
        m_neg = hn.mean(dim=0)
        d_vec = m_pos - m_neg
        d_norm = float(d_vec.norm())
        d_unit = d_vec / (d_norm + 1e-8)

        # Sycophancy principal component: top right singular vector of D = h_neg - h_pos.
        diff = hn - hp  # [N, d]
        try:
            _u, _s, vt = torch.linalg.svd(diff, full_matrices=False)
            syc = vt[0].to(torch.float32)
        except Exception as exc:  # extremely unlikely; keep the run alive
            print(f"[warn] SVD failed at layer {layer_idx}: {exc}", file=sys.stderr)
            syc = unit(-d_vec)
        if torch.dot(syc, m_neg - m_pos) <= 0:
            syc = -syc
        syc = syc / (syc.norm() + 1e-8)

        # MLP / attention key directions (Rank-1 baking targets).
        mlp_pos_mean = z_mlp_pos[:, layer_idx, :].mean(0)
        mlp_neg_mean = z_mlp_neg[:, layer_idx, :].mean(0)
        m_z_mlp = 0.5 * (mlp_pos_mean + mlp_neg_mean)
        r_mlp = m_z_mlp / (m_z_mlp.norm() ** 2 + 1e-8)

        attn_pos_mean = z_attn_pos[:, layer_idx, :].mean(0)
        attn_neg_mean = z_attn_neg[:, layer_idx, :].mean(0)
        m_z_attn = 0.5 * (attn_pos_mean + attn_neg_mean)
        r_attn = m_z_attn / (m_z_attn.norm() ** 2 + 1e-8)

        mlp_key_vals = z_mlp_neg[:, layer_idx, :] @ r_mlp
        attn_key_vals = z_attn_neg[:, layer_idx, :] @ r_attn

        # Diagnostics.
        if resid_norm_cnt[layer_idx] > 0:
            rn = float(resid_norm_sum[layer_idx] / resid_norm_cnt[layer_idx])
        else:
            rn = float("nan")

        proj_pos = hp @ d_unit
        proj_neg = hn @ d_unit
        d_cohen = cohen_d(proj_pos, proj_neg)
        cv_acc = cv_direction_accuracy(hp, hn, k=5, seed=0)

        cat_d: Dict[str, float] = {}
        for cat_name, idxs in cat_index.items():
            if len(idxs) >= 2:
                idx_t = torch.tensor(idxs, dtype=torch.long)
                cat_d[cat_name] = cohen_d(proj_pos[idx_t], proj_neg[idx_t])
            else:
                cat_d[cat_name] = float("nan")

        delta_h[layer_idx] = d_vec
        delta_h_unit[layer_idx] = d_unit
        v_syc[layer_idx] = syc
        mu_pos[layer_idx] = m_pos
        mu_neg[layer_idx] = m_neg
        mu_z_mlp[layer_idx] = m_z_mlp
        mu_z_mlp_pos[layer_idx] = mlp_pos_mean
        mu_z_mlp_neg[layer_idx] = mlp_neg_mean
        mu_z_attn[layer_idx] = m_z_attn
        mu_z_attn_pos[layer_idx] = attn_pos_mean
        mu_z_attn_neg[layer_idx] = attn_neg_mean
        resid_norm[layer_idx] = rn

        layer_stats.append(
            {
                "layer": layer_idx,
                "delta_norm": d_norm,
                "resid_norm": rn,
                "rel_delta_norm": (d_norm / rn) if rn and math.isfinite(rn) else float("nan"),
                "cos_syc_vs_neg_delta": float(torch.dot(syc, -d_unit)),
                "cohen_d": d_cohen,
                "cv_acc": cv_acc,
                "mlp_key_stability_mean": float(mlp_key_vals.mean()),
                "mlp_key_stability_std": float(mlp_key_vals.std(unbiased=False)),
                "attn_key_stability_mean": float(attn_key_vals.mean()),
                "attn_key_stability_std": float(attn_key_vals.std(unbiased=False)),
                "cohen_d_math_jailbreak": cat_d["math_jailbreak"],
                "cohen_d_adversarial_override": cat_d["adversarial_override"],
            }
        )

    # ------------------------------------------------------------------ #
    # Save tensors
    # ------------------------------------------------------------------ #
    out_pt_path = Path(args.out_pt)
    out_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "delta_h": delta_h.cpu(),
            "delta_h_unit": delta_h_unit.cpu(),
            "v_syc": v_syc.cpu(),
            "mu_pos": mu_pos.cpu(),
            "mu_neg": mu_neg.cpu(),
            "mu_z_mlp": mu_z_mlp.cpu(),
            "mu_z_mlp_pos": mu_z_mlp_pos.cpu(),
            "mu_z_mlp_neg": mu_z_mlp_neg.cpu(),
            "mu_z_attn": mu_z_attn.cpu(),
            "mu_z_attn_pos": mu_z_attn_pos.cpu(),
            "mu_z_attn_neg": mu_z_attn_neg.cpu(),
            "resid_norm": resid_norm.cpu(),
        },
        out_pt_path,
    )
    print(f"[ok] wrote tensors -> {out_pt_path}")

    # ------------------------------------------------------------------ #
    # Save human-readable JSON
    # ------------------------------------------------------------------ #
    def finite(x: float) -> Optional[float]:
        return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None

    def top_layers(key: str, k: int = 5) -> List[Dict[str, Any]]:
        ranked = sorted(
            (s for s in layer_stats if finite(s.get(key, float("nan"))) is not None),
            key=lambda s: s[key],
            reverse=True,
        )
        return [{"layer": s["layer"], key: finite(s[key])} for s in ranked[:k]]

    stats_payload = {
        "metadata": {
            "model_id": args.model_id,
            "num_hidden_layers": n_layers,
            "hidden_size": d_model,
            "intermediate_size": d_inter,
            "num_pairs": n_pairs,
            "categories": {c: categories.count(c) for c in sorted(set(categories))},
            "max_resp_tokens": args.max_resp_tokens,
            "system_prompt": SYSTEM_PROMPT,
            "torch_dtype": "float16",
            "device": "cuda",
            "pairs_path": args.pairs_path,
            "out_pt": str(out_pt_path),
            "model_architecture": info,
            "notes": {
                "v_syc": "top right singular vector of D = h_neg - h_pos, oriented so dot(v_syc, mu_neg - mu_pos) > 0",
                "delta_h": "mu_pos - mu_neg (pedagogical direction)",
                "mlp_key_stability": "z_mlp_neg @ r_mlp with r_mlp = mu_z_mlp / (||mu_z_mlp||^2 + 1e-8)",
                "cv_acc": "5-fold CV accuracy, direction and threshold fit on train folds only",
            },
        },
        "layers": [
            {
                "layer": s["layer"],
                "delta_norm": finite(s["delta_norm"]),
                "resid_norm": finite(s["resid_norm"]),
                "rel_delta_norm": finite(s["rel_delta_norm"]),
                "cos_syc_vs_neg_delta": finite(s["cos_syc_vs_neg_delta"]),
                "cohen_d": finite(s["cohen_d"]),
                "cv_acc": finite(s["cv_acc"]),
                "mlp_key_stability_mean": finite(s["mlp_key_stability_mean"]),
                "mlp_key_stability_std": finite(s["mlp_key_stability_std"]),
                "attn_key_stability_mean": finite(s["attn_key_stability_mean"]),
                "attn_key_stability_std": finite(s["attn_key_stability_std"]),
                "cohen_d_math_jailbreak": finite(s["cohen_d_math_jailbreak"]),
                "cohen_d_adversarial_override": finite(s["cohen_d_adversarial_override"]),
            }
            for s in layer_stats
        ],
        "top_layers_by_cohen_d": top_layers("cohen_d"),
        "top_layers_by_rel_delta": top_layers("rel_delta_norm"),
    }

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as fh:
        json.dump(stats_payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"[ok] wrote stats   -> {out_json_path}")

    best = stats_payload["top_layers_by_cohen_d"][0] if stats_payload["top_layers_by_cohen_d"] else None
    if best:
        print(
            f"[done] best layer by Cohen's d: L{best['layer']} (d={best['cohen_d']:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
