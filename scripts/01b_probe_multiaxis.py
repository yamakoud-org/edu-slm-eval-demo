#!/usr/bin/env python3
"""Dual-window (open + full) multi-axis representation probing for Qwen2.5.

Captures residual-stream hidden states (h), o_proj inputs (z_o), and
down_proj inputs (z_d) across all layers for each of the 3 orthogonal
pedagogical axes (answer_dump, misconception_syc, bloat_roleplay) plus
pooled 'all', over BOTH:
  - 'open': opening 16 response tokens (stance: refusal/sycophancy/dumping)
  - 'full': mean over all response tokens (conciseness, 1-step question, no roleplay)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NEUTRAL_SYSTEM_PROMPT = "あなたは親切で有能な日本語のAIアシスタントです。"
AXES = ["answer_dump", "misconception_syc", "bloat_roleplay"]


def cohen_d_and_cv(pos_vecs: np.ndarray, neg_vecs: np.ndarray) -> tuple[float, float]:
    """Compute Cohen's d along the difference-of-means direction and 5-fold CV accuracy."""
    mu_pos = pos_vecs.mean(axis=0)
    mu_neg = neg_vecs.mean(axis=0)
    diff = mu_neg - mu_pos
    norm = np.linalg.norm(diff) + 1e-8
    v = diff / norm
    p_proj = pos_vecs @ v
    n_proj = neg_vecs @ v
    pooled_std = np.sqrt(0.5 * (p_proj.var(ddof=1) + n_proj.var(ddof=1)) + 1e-8)
    d = float((n_proj.mean() - p_proj.mean()) / pooled_std)

    # 5-fold CV on paired indices
    n = len(pos_vecs)
    folds = 5
    correct = 0
    total = 0
    for f in range(folds):
        val_idx = [i for i in range(n) if i % folds == f]
        tr_idx = [i for i in range(n) if i % folds != f]
        mu_p_tr = pos_vecs[tr_idx].mean(axis=0)
        mu_n_tr = neg_vecs[tr_idx].mean(axis=0)
        w = mu_n_tr - mu_p_tr
        w = w / (np.linalg.norm(w) + 1e-8)
        thresh = 0.5 * (mu_p_tr @ w + mu_n_tr @ w)
        for i in val_idx:
            correct += int((pos_vecs[i] @ w) < thresh)
            correct += int((neg_vecs[i] @ w) >= thresh)
            total += 2
    return d, float(correct / max(1, total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--pairs_path", default="data/contrastive_pairs_v3.json")
    ap.add_argument("--out_pt", default="artifacts/probing_vectors_1.5b_v4.pt")
    ap.add_argument("--out_json", default="data/probing_stats_1.5b_v4.json")
    ap.add_argument("--open_tokens", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = json.loads(Path(args.pairs_path).read_text(encoding="utf-8"))
    pairs = data["pairs"] if isinstance(data, dict) else data
    print(f"[ok] loaded {len(pairs)} pairs from {args.pairs_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.float16,
        device_map=args.device,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    d_inter = model.config.intermediate_size
    print(f"[ok] loaded {args.model_id}: L={n_layers}, d={d_model}, d_ff={d_inter}")

    # Capture buffers per forward pass
    cur_h = [None] * n_layers
    cur_zo = [None] * n_layers
    cur_zd = [None] * n_layers
    hooks = []

    for l in range(n_layers):
        layer = model.model.layers[l]

        def make_layer_hook(idx):
            def fn(_mod, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                cur_h[idx] = hs[0].detach().float()
            return fn

        def make_pre_hook(buf, idx):
            def fn(_mod, inp):
                buf[idx] = inp[0][0].detach().float()
            return fn

        hooks.append(layer.register_forward_hook(make_layer_hook(l)))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(make_pre_hook(cur_zo, l)))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(make_pre_hook(cur_zd, l)))

    # Storage: [pair_idx][layer]
    records = {
        win: {
            key: [torch.zeros((len(pairs), dim), dtype=torch.float32) for _ in range(n_layers)]
            for key, dim in [
                ("h_pos", d_model),
                ("h_neg", d_model),
                ("zo_pos", d_model),
                ("zo_neg", d_model),
                ("zd_pos", d_inter),
                ("zd_neg", d_inter),
            ]
        }
        for win in ("open", "full")
    }
    pair_axes = [p["axis"] for p in pairs]

    def run_single(user_text: str, resp_text: str):
        messages = [
            {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        pref_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": resp_text}],
            tokenize=True,
            add_generation_prompt=False,
        )
        start = len(pref_ids)
        if full_ids[:start] != pref_ids:
            # Template may differ by trailing whitespace; fall back to common prefix.
            start = 0
            for a, b in zip(pref_ids, full_ids):
                if a != b:
                    break
                start += 1
        if start >= len(full_ids):
            raise ValueError("assistant response produced no tokens after the prompt")
        end_open = min(len(full_ids), start + args.open_tokens)
        end_full = len(full_ids)
        inp = torch.tensor([full_ids], device=args.device)
        with torch.no_grad():
            model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
        res = {"open": ([], [], []), "full": ([], [], [])}
        for l in range(n_layers):
            res["open"][0].append(cur_h[l][start:end_open].mean(dim=0).cpu())
            res["open"][1].append(cur_zo[l][start:end_open].mean(dim=0).cpu())
            res["open"][2].append(cur_zd[l][start:end_open].mean(dim=0).cpu())
            res["full"][0].append(cur_h[l][start:end_full].mean(dim=0).cpu())
            res["full"][1].append(cur_zo[l][start:end_full].mean(dim=0).cpu())
            res["full"][2].append(cur_zd[l][start:end_full].mean(dim=0).cpu())
        return res

    for idx, p in enumerate(pairs):
        r_pos = run_single(p["student_prompt"], p["positive_response"])
        r_neg = run_single(p["student_prompt"], p["negative_response"])
        for win in ("open", "full"):
            for l in range(n_layers):
                records[win]["h_pos"][l][idx] = r_pos[win][0][l]
                records[win]["zo_pos"][l][idx] = r_pos[win][1][l]
                records[win]["zd_pos"][l][idx] = r_pos[win][2][l]
                records[win]["h_neg"][l][idx] = r_neg[win][0][l]
                records[win]["zo_neg"][l][idx] = r_neg[win][1][l]
                records[win]["zd_neg"][l][idx] = r_neg[win][2][l]
        if (idx + 1) % 12 == 0:
            print(f"[probe] {idx + 1}/{len(pairs)} pairs captured", flush=True)

    for h in hooks:
        h.remove()

    axes_out = {}
    stats_out = {"model_id": args.model_id, "num_layers": n_layers, "axes": {}, "axis_cosines": []}

    for axis in AXES + ["all"]:
        idxs = [i for i, a in enumerate(pair_axes) if axis == "all" or a == axis]
        axes_out[axis] = {}
        stats_out["axes"][axis] = {}
        for win in ("open", "full"):
            v_neg_list, delta_h_list = [], []
            zo_pos_list, zo_neg_list, zd_pos_list, zd_neg_list = [], [], [], []
            layer_stats = []
            for l in range(n_layers):
                hp = records[win]["h_pos"][l][idxs]
                hn = records[win]["h_neg"][l][idxs]
                mu_p = hp.mean(dim=0)
                mu_n = hn.mean(dim=0)
                dh = mu_p - mu_n  # shift toward positive
                vn = mu_n - mu_p  # bad direction
                vn_unit = vn / (vn.norm() + 1e-8)
                rel_d = float(dh.norm() / (0.5 * (hp.norm(dim=1).mean() + hn.norm(dim=1).mean()) + 1e-8))
                d_val, cv_acc = cohen_d_and_cv(hp.numpy(), hn.numpy())

                v_neg_list.append(vn_unit)
                delta_h_list.append(dh)
                zo_pos_list.append(records[win]["zo_pos"][l][idxs].mean(dim=0))
                zo_neg_list.append(records[win]["zo_neg"][l][idxs].mean(dim=0))
                zd_pos_list.append(records[win]["zd_pos"][l][idxs].mean(dim=0))
                zd_neg_list.append(records[win]["zd_neg"][l][idxs].mean(dim=0))
                layer_stats.append({
                    "layer": l,
                    "cohen_d": round(d_val, 4),
                    "cv_acc": round(cv_acc, 4),
                    "rel_delta_norm": round(rel_d, 5),
                })

            axes_out[axis][win] = {
                "v_neg": torch.stack(v_neg_list, dim=0),
                "delta_h": torch.stack(delta_h_list, dim=0),
                "mu_z_o_pos": torch.stack(zo_pos_list, dim=0),
                "mu_z_o_neg": torch.stack(zo_neg_list, dim=0),
                "mu_z_d_pos": torch.stack(zd_pos_list, dim=0),
                "mu_z_d_neg": torch.stack(zd_neg_list, dim=0),
            }
            stats_out["axes"][axis][win] = layer_stats

    # Pairwise cosine similarity among the 3 primary axis vectors per layer
    for l in range(n_layers):
        v_dump = axes_out["answer_dump"]["open"]["v_neg"][l]
        v_syc = axes_out["misconception_syc"]["open"]["v_neg"][l]
        v_bloat = axes_out["bloat_roleplay"]["full"]["v_neg"][l]
        stats_out["axis_cosines"].append({
            "layer": l,
            "cos_dump_syc": round(float(torch.dot(v_dump, v_syc)), 4),
            "cos_dump_bloat": round(float(torch.dot(v_dump, v_bloat)), 4),
            "cos_syc_bloat": round(float(torch.dot(v_syc, v_bloat)), 4),
        })

    # Gram-Schmidt orthogonalized bad basis per layer, primary windows only.
    orth_basis = {a: [] for a in AXES}
    orth_pairs = {"dump_syc": [], "dump_bloat": [], "syc_bloat": []}
    for l in range(n_layers):
        v_dump = axes_out["answer_dump"]["open"]["v_neg"][l]
        v_syc = axes_out["misconception_syc"]["open"]["v_neg"][l]
        v_bloat = axes_out["bloat_roleplay"]["full"]["v_neg"][l]
        u_dump = v_dump / (v_dump.norm() + 1e-8)
        u_syc = v_syc - torch.dot(v_syc, u_dump) * u_dump
        u_syc = u_syc / (u_syc.norm() + 1e-8)
        u_bloat = v_bloat - torch.dot(v_bloat, u_dump) * u_dump - torch.dot(v_bloat, u_syc) * u_syc
        u_bloat = u_bloat / (u_bloat.norm() + 1e-8)
        for name, u in (("answer_dump", u_dump), ("misconception_syc", u_syc), ("bloat_roleplay", u_bloat)):
            orth_basis[name].append(u)
        orth_pairs["dump_syc"].append(round(float(torch.dot(u_dump, u_syc)), 6))
        orth_pairs["dump_bloat"].append(round(float(torch.dot(u_dump, u_bloat)), 6))
        orth_pairs["syc_bloat"].append(round(float(torch.dot(u_syc, u_bloat)), 6))

    stats_out["orthogonalized_basis_cosines"] = [dict(layer=l, **{k: v[l] for k, v in orth_pairs.items()}) for l in range(n_layers)]

    Path(args.out_pt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": args.model_id,
            "num_layers": n_layers,
            "axes": axes_out,
            "orth_basis": {a: torch.stack(v, dim=0) for a, v in orth_basis.items()},
        },
        args.out_pt,
    )
    stats_out["metadata"] = {
        "num_pairs": len(pairs),
        "per_axis_pairs": {a: pair_axes.count(a) for a in AXES},
        "system_prompt": NEUTRAL_SYSTEM_PROMPT,
        "torch_dtype": "float16",
        "open_tokens": args.open_tokens,
        "primary_window": {"answer_dump": "open", "misconception_syc": "open", "bloat_roleplay": "full"},
        "axes_definition": "v_neg = unit(mu_neg - mu_pos); delta_h = mu_pos - mu_neg",
    }
    Path(args.out_json).write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {args.out_pt} and {args.out_json}")

    # Compact summary
    for ax, win in [("answer_dump", "open"), ("misconception_syc", "open"), ("bloat_roleplay", "full")]:
        ls = stats_out["axes"][ax][win]
        best_d = max(ls, key=lambda x: x["cohen_d"])
        best_r = max(ls, key=lambda x: x["rel_delta_norm"])
        print(f"  [{ax:18s}/{win:4s}] peak d: L{best_d['layer']:02d} (d={best_d['cohen_d']:.3f}, cv={best_d['cv_acc']:.3f}) | peak rel_d: L{best_r['layer']:02d} ({best_r['rel_delta_norm']:.4f})")


if __name__ == "__main__":
    main()
