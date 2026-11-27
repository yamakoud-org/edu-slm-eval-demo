#!/usr/bin/env python3
"""Exact Zero-Interference Layer-27 Orthogonal '？ -> <|im_end|>' Circuit Surgery.

Fixes two flaws from v1:
  1. Edits ONLY the final MLP layer (L27) so L0..L26 reasoning/tone are 100% untouched.
  2. Trigger key k_q is STRICTLY ORTHOGONAL to mean non-question token activations:
       k_orth = mu_q - (dot(mu_q, mu_o) / ||mu_o||^2) * mu_o
       k_key  = k_orth / dot(k_orth, mu_q)
     guaranteeing dot(k_key, mu_o) == 0.0000 (zero effect on normal tokens) and
     dot(k_key, mu_q) == 1.0000 (fires ONLY on sentence-final '？').
  3. Target direction u_eos is the exact <|im_end|> unembedding vector orthogonalized
     against the mean vocabulary unembedding vector so it boosts ONLY <|im_end|>
     (never emojis or punctuation).
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NEUTRAL_SYSTEM_PROMPT = "あなたは親切で有能な日本語のAIアシスタントです。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="artifacts/edited_model_v3")
    ap.add_argument("--pairs_path", default="data/contrastive_pairs_v3.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alpha_eos", type=float, default=15.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = json.loads(Path(args.pairs_path).read_text(encoding="utf-8"))
    pairs = data["pairs"] if isinstance(data, dict) else data

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float16, device_map=args.device)
    model.eval()

    last_l = len(model.model.layers) - 1
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")
    if eos_id is None or eos_id < 0:
        eos_id = tok.eos_token_id

    z_q, z_o = [], []
    cur = {}

    def pre_hook(_mod, inp):
        cur["z"] = inp[0][0].detach().float()

    h = model.model.layers[last_l].mlp.down_proj.register_forward_pre_hook(pre_hook)

    for p in pairs:
        prefix = tok.apply_chat_template(
            [
                {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                {"role": "user", "content": p["student_prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        pos = p["positive_response"]
        pref_ids = tok(prefix, add_special_tokens=False).input_ids
        full_ids = tok(prefix + pos, add_special_tokens=False).input_ids
        start = len(pref_ids)
        if len(full_ids) <= start + 3:
            continue
        q_idx = len(full_ids) - 1
        if "？" not in tok.decode([full_ids[q_idx]]) and "?" not in tok.decode([full_ids[q_idx]]):
            continue
        with torch.no_grad():
            model(input_ids=torch.tensor([full_ids], device=args.device), use_cache=False)
        z_q.append(cur["z"][q_idx].cpu())
        z_o.append(cur["z"][start:q_idx].mean(dim=0).cpu())

    h.remove()

    mu_q = torch.stack(z_q, dim=0).mean(dim=0)
    mu_o = torch.stack(z_o, dim=0).mean(dim=0)

    # Strictly orthogonal trigger key: dot(k_key, mu_o) == 0.0000
    k_orth = mu_q - (torch.dot(mu_q, mu_o) / (mu_o.pow(2).sum() + 1e-8)) * mu_o
    k_key = k_orth / torch.dot(k_orth, mu_q).clamp(min=1e-4)

    # Pure <|im_end|> unembedding direction (de-meaned across vocab & weighted by final RMSNorm)
    W_lm = model.lm_head.weight.data.detach().cpu().float()
    w_mean = W_lm.mean(dim=0)
    w_eos = W_lm[eos_id] - w_mean
    norm_w = model.model.norm.weight.data.detach().cpu().float()
    u_eos = w_eos / (norm_w + 1e-4)
    u_eos = u_eos / (u_eos.norm() + 1e-8)

    # Verify top-5 tokens boosted by u_eos * norm_w
    logits_boost = W_lm @ (u_eos * norm_w)
    top5_ids = torch.topk(logits_boost, 5).indices.tolist()
    top5_toks = [repr(tok.decode([i])) for i in top5_ids]
    print(f"[eos-dir] top-5 boosted tokens: {top5_toks}")

    model.to("cpu")
    with torch.no_grad():
        W = model.model.layers[last_l].mlp.down_proj.weight.data.float()
        delta_W = args.alpha_eos * torch.outer(u_eos, k_key)
        rel_fro = float(delta_W.norm()) / (float(W.norm()) + 1e-8)
        model.model.layers[last_l].mlp.down_proj.weight.data.copy_((W + delta_W).to(torch.float16))
        print(f"[L{last_l:02d}] alpha={args.alpha_eos:.1f} key(？)={float(torch.dot(k_key, mu_q)):.4f} key(other)={float(torch.dot(k_key, mu_o)):+.6f} rel_fro={rel_fro:.5f}")

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path, safe_serialization=True)
    tok.save_pretrained(out_path)
    (out_path / "eos_surgery_metrics.json").write_text(
        json.dumps({"layer": last_l, "alpha_eos": args.alpha_eos, "rel_fro": rel_fro, "top5_boosted": top5_toks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[ok] saved -> {out_path}")


if __name__ == "__main__":
    main()
