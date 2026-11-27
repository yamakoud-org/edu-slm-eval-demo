#!/usr/bin/env python3
"""Step 1+2 for Qwen/Qwen3-1.7B (non-thinking mode):
  1. Probe 28 layers on data/contrastive_pairs_v3_48.json (opening 16 tokens)
     -> artifacts/probing_vectors_qwen3_1.7b.pt & data/probing_stats_qwen3_1.7b.json
  2. Evaluate Qwen3-1.7B Baseline & Prompt-Only (clean 7-principle prompt) on
     data/eval_benchmark_20.json + 4 qualitative sanity prompts (incl. 三角関数).
"""

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
NEUTRAL_PROMPT = "あなたは親切で有能な日本語のAIアシスタントです。必ず自然な日本語で簡潔に答えてください。"
EDU_PROMPT_Q3 = (
    "あなたは学習科学に基づく日本語の教育チューターです。次のルールを必ず守って返答してください。\n"
    "・答えや完成文を直接教えず、「できません」などの拒絶表現も使わず、温かく寄り添いながら一緒に考える姿勢で始める。\n"
    "・生徒の誤った前提には迎合・肯定せず、正しい事実を踏まえつつ気づきを促す優しい問いで立ち止まらせる。\n"
    "・1回の返答は2〜3文（120字前後）で簡潔にまとめ、目の前の問題に直結する具体的な問いかけを1つだけ文末に添える。\n"
    "・必ず自然な日本語の丁寧語（です・ます）のみで書き、簡体字や一人二役・箇条書き解説は使わない。"
)


def cohen_d_and_cv(pos_vecs: np.ndarray, neg_vecs: np.ndarray) -> tuple[float, float]:
    diff = neg_vecs.mean(axis=0) - pos_vecs.mean(axis=0)
    v = diff / (np.linalg.norm(diff) + 1e-8)
    p_proj, n_proj = pos_vecs @ v, neg_vecs @ v
    pooled = np.sqrt(0.5 * (p_proj.var(ddof=1) + n_proj.var(ddof=1)) + 1e-8)
    d = float((n_proj.mean() - p_proj.mean()) / pooled)
    n = len(pos_vecs)
    correct = total = 0
    for f in range(5):
        val = [i for i in range(n) if i % 5 == f]
        tr = [i for i in range(n) if i % 5 != f]
        w = neg_vecs[tr].mean(axis=0) - pos_vecs[tr].mean(axis=0)
        w = w / (np.linalg.norm(w) + 1e-8)
        th = 0.5 * (pos_vecs[tr].mean(axis=0) @ w + neg_vecs[tr].mean(axis=0) @ w)
        for i in val:
            correct += int((pos_vecs[i] @ w) < th) + int((neg_vecs[i] @ w) >= th)
            total += 2
    return d, float(correct / max(1, total))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="cuda")
    mdl.eval()

    n_layers = mdl.config.num_hidden_layers
    d_model = mdl.config.hidden_size
    d_ff = mdl.config.intermediate_size
    pairs = json.loads(Path("data/contrastive_pairs_v3_48.json").read_text(encoding="utf-8"))
    print(f"[ok] loaded {MODEL_ID} (L={n_layers}, d={d_model}, d_ff={d_ff}) & {len(pairs)} pairs")

    # --- 1. Probing (opening 16 response tokens, non-thinking mode) ---
    cur_h, cur_zo, cur_zd = [None] * n_layers, [None] * n_layers, [None] * n_layers
    hooks = []
    for l in range(n_layers):
        layer = mdl.model.layers[l]
        hooks.append(layer.register_forward_hook(lambda _m, _i, out, idx=l: cur_h.__setitem__(idx, (out[0] if isinstance(out, tuple) else out)[0].detach().float())))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda _m, inp, idx=l: cur_zo.__setitem__(idx, inp[0][0].detach().float())))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(lambda _m, inp, idx=l: cur_zd.__setitem__(idx, inp[0][0].detach().float())))

    hp = [torch.zeros((len(pairs), d_model)) for _ in range(n_layers)]
    hn = [torch.zeros((len(pairs), d_model)) for _ in range(n_layers)]
    zop = [torch.zeros((len(pairs), d_model)) for _ in range(n_layers)]
    zon = [torch.zeros((len(pairs), d_model)) for _ in range(n_layers)]
    zdp = [torch.zeros((len(pairs), d_ff)) for _ in range(n_layers)]
    zdn = [torch.zeros((len(pairs), d_ff)) for _ in range(n_layers)]

    def capture(user_text, resp_text):
        pref = tok.apply_chat_template(
            [{"role": "system", "content": NEUTRAL_PROMPT}, {"role": "user", "content": user_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        p_ids = tok(pref, add_special_tokens=False).input_ids
        f_ids = tok(pref + resp_text, add_special_tokens=False).input_ids
        s = len(p_ids)
        e = min(len(f_ids), s + 16)
        with torch.no_grad():
            mdl(input_ids=torch.tensor([f_ids], device="cuda"), use_cache=False)
        return (
            [cur_h[l][s:e].mean(0).cpu() for l in range(n_layers)],
            [cur_zo[l][s:e].mean(0).cpu() for l in range(n_layers)],
            [cur_zd[l][s:e].mean(0).cpu() for l in range(n_layers)],
        )

    for idx, p in enumerate(pairs):
        h1, o1, d1 = capture(p["student_prompt"], p["positive_response"])
        h2, o2, d2 = capture(p["student_prompt"], p["negative_response"])
        for l in range(n_layers):
            hp[l][idx], zop[l][idx], zdp[l][idx] = h1[l], o1[l], d1[l]
            hn[l][idx], zon[l][idx], zdn[l][idx] = h2[l], o2[l], d2[l]

    for h in hooks:
        h.remove()

    # Save compatible with 02_apply_weight_surgery.py!
    v_syc_l, dh_l, dhu_l = [], [], []
    mu_z_attn_pos_l, mu_z_attn_neg_l, mu_z_attn_all_l = [], [], []
    mu_z_mlp_pos_l, mu_z_mlp_neg_l, mu_z_mlp_all_l = [], [], []
    layer_stats = []

    print("\nL  | cohen_d | cv_acc | rel_delta_norm")
    for l in range(n_layers):
        mu_p, mu_n = hp[l].mean(0), hn[l].mean(0)
        dh = mu_p - mu_n
        vn = mu_n - mu_p
        vs = vn / (vn.norm() + 1e-8)
        dhu = dh / (dh.norm() + 1e-8)
        rel_d = float(dh.norm() / (0.5 * (hp[l].norm(dim=1).mean() + hn[l].norm(dim=1).mean()) + 1e-8))
        d_val, cv = cohen_d_and_cv(hp[l].numpy(), hn[l].numpy())
        v_syc_l.append(vs)
        dh_l.append(dh)
        dhu_l.append(dhu)
        mu_z_attn_pos_l.append(zop[l].mean(0))
        mu_z_attn_neg_l.append(zon[l].mean(0))
        mu_z_attn_all_l.append(0.5 * (zop[l].mean(0) + zon[l].mean(0)))
        mu_z_mlp_pos_l.append(zdp[l].mean(0))
        mu_z_mlp_neg_l.append(zdn[l].mean(0))
        mu_z_mlp_all_l.append(0.5 * (zdp[l].mean(0) + zdn[l].mean(0)))
        layer_stats.append({"layer": l, "cohen_d": round(d_val, 4), "cv_acc": round(cv, 4), "rel_delta_norm": round(rel_d, 5)})
        if 8 <= l <= 25:
            print(f"{l:02d} |  {d_val:5.3f}  | {cv:6.3f} | {rel_d:.4f}")

    torch.save({
        "model_id": MODEL_ID,
        "num_layers": n_layers,
        "v_syc": torch.stack(v_syc_l),
        "delta_h": torch.stack(dh_l),
        "delta_h_unit": torch.stack(dhu_l),
        "mu_z_attn_pos": torch.stack(mu_z_attn_pos_l),
        "mu_z_attn_neg": torch.stack(mu_z_attn_neg_l),
        "mu_z_attn": torch.stack(mu_z_attn_all_l),
        "mu_z_mlp_pos": torch.stack(mu_z_mlp_pos_l),
        "mu_z_mlp_neg": torch.stack(mu_z_mlp_neg_l),
        "mu_z_mlp": torch.stack(mu_z_mlp_all_l),
    }, "artifacts/probing_vectors_qwen3_1.7b.pt")
    Path("data/probing_stats_qwen3_1.7b.json").write_text(json.dumps({"model_id": MODEL_ID, "layers": layer_stats}, indent=2), encoding="utf-8")
    print("[ok] saved probing_vectors_qwen3_1.7b.pt & probing_stats_qwen3_1.7b.json")

    # --- 2. Baseline & Prompt-Only 20Q Benchmark ---
    bench_raw = json.loads(Path("data/eval_benchmark_20.json").read_text(encoding="utf-8"))
    items = bench_raw["items"] if isinstance(bench_raw, dict) else bench_raw
    ref = json.loads(Path("artifacts/benchmark_raw_outputs.json").read_text())
    rec_key = [k for k in ref["conditions"][0].keys() if isinstance(ref["conditions"][0][k], list)][0]

    def run_cond(sys_prompt, cname):
        recs = []
        for it in items:
            pt = it.get("student_prompt") or it.get("prompt")
            txt = tok.apply_chat_template(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": pt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inp = tok(txt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = mdl.generate(**inp, max_new_tokens=200, do_sample=False, repetition_penalty=1.08, pad_token_id=tok.eos_token_id)
            resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            hit = any(re.search(pat, resp, re.IGNORECASE) for pat in it.get("forbidden_answer_patterns", []))
            recs.append({"id": it["id"], "category": it.get("category", ""), "prompt": pt, "response": resp, "forbidden_hit": hit})
        return {"condition": cname, rec_key: recs}

    c_base = run_cond(NEUTRAL_PROMPT, "baseline")
    c_po = run_cond(EDU_PROMPT_Q3, "prompt_only")
    bout = "artifacts/benchmark_qwen3_base_po.json"
    Path(bout).write_text(json.dumps({"conditions": [c_base, c_po]}, ensure_ascii=False, indent=2), encoding="utf-8")
    res = subprocess.run(["python3", "scripts/03b_eval_7principles.py", "--bench_json", bout, "--label", "Qwen3_1.7B_Base_vs_PO"], capture_output=True, text=True, check=True)
    print("\n" + res.stdout)

    print("=== Qwen3 Prompt-Only sample replies ===")
    for r in c_po[rec_key][:6]:
        print(f"  {r['id']} ({r['category']:20s}) [{len(r['response']):3d}ch hit={r['forbidden_hit']}]: {r['response'].replace(chr(10),' ')[:130]}")
    # Also test 三角関数 under clean prompt
    txt = tok.apply_chat_template([{"role":"system","content":EDU_PROMPT_Q3},{"role":"user","content":"三角関数について教えて"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tok(txt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = mdl.generate(**inp, max_new_tokens=160, do_sample=False, repetition_penalty=1.08, pad_token_id=tok.eos_token_id)
    print("\n[三角関数について教えて (Prompt-Only)]:", tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
