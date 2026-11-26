#!/usr/bin/env python3
"""Apply directional weight surgery (abliteration + rank-1 baking) to a causal LM.

Runs on CPU or a small GPU (e.g. RTX 2060 SUPER). Tensors are manipulated in
float32 and written back as float16 safetensors. bfloat16 is never used.

Example:
    python3 scripts/02_apply_weight_surgery.py \
        --recipe_json artifacts/recipes/cycle1.json \
        --out_dir artifacts/models/cycle1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_VECTORS = "artifacts/probing_vectors.pt"

WEIGHT_KEYS = {
    "o_proj": "self_attn.o_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


# --------------------------------------------------------------------------- #
# Recipe helpers
# --------------------------------------------------------------------------- #
def _per_layer(value: Any, layer: int, default: float = 0.0) -> float:
    """Resolve a recipe entry that is either a scalar or a per-layer dict."""
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in (str(layer), layer):
            if key in value:
                return float(value[key])
        return float(default)
    raise ValueError(f"Unsupported recipe value for layer {layer}: {value!r}")


def load_recipe(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        recipe = json.load(fh)
    if "layers" not in recipe or not recipe["layers"]:
        raise ValueError("Recipe must declare a non-empty 'layers' list")
    recipe["layers"] = [int(layer) for layer in recipe["layers"]]
    return recipe


def load_vectors(path: Path) -> Dict[str, Any]:
    vectors = torch.load(str(path), map_location="cpu")
    if not isinstance(vectors, dict):
        raise ValueError(f"Expected a dict of tensors in {path}")
    return vectors


def get_vector(vectors: Dict[str, Any], name: str, layer: int) -> torch.Tensor:
    """Fetch vectors[name][layer] as a float32 tensor (dict or stacked tensor)."""
    if name not in vectors:
        raise KeyError(f"Missing required vector '{name}' in probing vectors")
    entry = vectors[name]
    if isinstance(entry, dict):
        for key in (layer, str(layer)):
            if key in entry:
                return entry[key].detach().to(torch.float32).reshape(-1)
        raise KeyError(f"Missing layer {layer} in vector '{name}'")
    if isinstance(entry, torch.Tensor):
        if entry.dim() == 1:
            return entry.detach().to(torch.float32).reshape(-1)
        return entry[layer].detach().to(torch.float32).reshape(-1)
    raise TypeError(f"Unsupported storage for vector '{name}': {type(entry)!r}")


# --------------------------------------------------------------------------- #
# Surgery math
# --------------------------------------------------------------------------- #
def preserve_rows(W: torch.Tensor, W_orig: torch.Tensor) -> torch.Tensor:
    """Rescale each row of W so its norm matches the corresponding row of W_orig."""
    row_norm = W.norm(dim=1, keepdim=True)
    target = W_orig.norm(dim=1, keepdim=True)
    return W * (target / row_norm.clamp_min(1e-8))


def apply_safety_clamp(
    W: torch.Tensor,
    W_orig: torch.Tensor,
    max_rel_frobenius_delta: float | None,
) -> tuple[torch.Tensor, float, bool]:
    """Clamp the relative Frobenius delta, returning (W, rel_fro, clamped)."""
    base_norm = W_orig.norm() + 1e-8
    delta = W - W_orig
    rel_fro = float((delta.norm() / base_norm).item())
    if max_rel_frobenius_delta is None:
        return W, rel_fro, False
    if rel_fro <= max_rel_frobenius_delta:
        return W, rel_fro, False

    target_norm = max_rel_frobenius_delta * base_norm
    scale = target_norm / (delta.norm() + 1e-8)
    W = W_orig + delta * scale
    rel_fro = float(((W - W_orig).norm() / base_norm).item())
    return W, rel_fro, True


def surgical_update(
    W: torch.Tensor,
    W_orig: torch.Tensor,
    v_syc: torch.Tensor,
    v_bake: torch.Tensor,
    r_key: torch.Tensor,
    beta: float,
    alpha: float,
    preserve_row_norm: bool,
    max_rel_frobenius_delta: float | None,
) -> tuple[torch.Tensor, float, bool]:
    """Run abliteration (step A), optional row-norm restore, baking (step B), clamp."""
    W = W.to(torch.float32)
    if beta > 0.0:
        W = W - beta * torch.outer(v_syc, v_syc @ W)
    if preserve_row_norm:
        W = preserve_rows(W, W_orig)
    if alpha != 0.0:
        W = W + alpha * torch.outer(v_bake, r_key)
    return apply_safety_clamp(W, W_orig, max_rel_frobenius_delta)


def contrastive_key(
    z_neg: torch.Tensor,
    z_pos: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Rank-1 bake key from a contrastive trigger: (z_neg - gamma*z_pos) / <., z_neg>."""
    z_trig = z_neg - gamma * z_pos
    return z_trig / torch.dot(z_trig, z_neg).clamp(min=1e-4)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--vectors_pt", default=DEFAULT_VECTORS)
    parser.add_argument("--recipe_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cpu", help="Device used to load the base model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recipe = load_recipe(Path(args.recipe_json))
    vectors = load_vectors(Path(args.vectors_pt))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers: List[int] = recipe["layers"]
    bake_key_mode = recipe.get("bake_key_mode", "mu_z_all")
    if bake_key_mode not in ("mu_z_all", "mu_z_neg", "contrastive"):
        raise ValueError(f"Unsupported bake_key_mode: {bake_key_mode}")
    contrastive_gamma = float(recipe.get("contrastive_gamma", 0.85))

    scale_by_delta_norm = bool(recipe.get("scale_alpha_by_delta_norm", True))
    orthogonalize = bool(recipe.get("orthogonalize_bake_against_syc", False))
    preserve_row_norm = bool(recipe.get("preserve_row_norm", False))
    max_rel = recipe.get("max_rel_frobenius_delta")
    max_rel = None if max_rel is None else float(max_rel)

    # Imported lazily so `--help` and syntax checks stay cheap.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {args.model_id} (float16, device={args.device})")
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    z_key_names = (
        ("mu_z_mlp_neg", "mu_z_attn_neg")
        if bake_key_mode in ("mu_z_neg", "contrastive")
        else ("mu_z_mlp", "mu_z_attn")
    )
    z_pos_key_names = ("mu_z_mlp_pos", "mu_z_attn_pos")

    metrics: List[Dict[str, Any]] = []
    model_layers = model.model.layers

    for layer in layers:
        if layer < 0 or layer >= len(model_layers):
            raise IndexError(f"Layer {layer} out of range (model has {len(model_layers)})")

        v_syc = get_vector(vectors, "v_syc", layer)
        delta_h = get_vector(vectors, "delta_h", layer)
        delta_h_unit = get_vector(vectors, "delta_h_unit", layer)
        z_mlp = get_vector(vectors, z_key_names[0], layer)
        z_attn = get_vector(vectors, z_key_names[1], layer)

        v_bake = delta_h if scale_by_delta_norm else delta_h_unit
        if orthogonalize:
            v_bake = v_bake - torch.dot(v_bake, v_syc) * v_syc

        if bake_key_mode == "contrastive":
            z_mlp_pos = get_vector(vectors, z_pos_key_names[0], layer)
            z_attn_pos = get_vector(vectors, z_pos_key_names[1], layer)
            r_mlp = contrastive_key(z_mlp, z_mlp_pos, contrastive_gamma)
            r_attn = contrastive_key(z_attn, z_attn_pos, contrastive_gamma)
        else:
            r_mlp = z_mlp / (z_mlp.pow(2).sum() + 1e-8)
            r_attn = z_attn / (z_attn.pow(2).sum() + 1e-8)

        beta_o = _per_layer(recipe.get("o_proj_abliterate_beta"), layer)
        beta_d = _per_layer(recipe.get("down_proj_abliterate_beta"), layer)
        alpha_o = _per_layer(recipe.get("o_proj_bake_alpha"), layer)
        alpha_d = _per_layer(recipe.get("down_proj_bake_alpha"), layer)

        block = model_layers[layer]
        record: Dict[str, Any] = {
            "layer": layer,
            "beta_o": beta_o,
            "beta_d": beta_d,
            "alpha_o": alpha_o,
            "alpha_d": alpha_d,
        }
        any_clamped = False

        for kind, beta, alpha, key in (
            ("o_proj", beta_o, alpha_o, r_attn),
            ("down_proj", beta_d, alpha_d, r_mlp),
        ):
            param = block.get_parameter(WEIGHT_KEYS[kind])
            W_orig = param.data.detach().to(torch.float32).clone()

            energy_before = float((v_syc @ W_orig).norm().item())
            W_new, rel_fro, clamped = surgical_update(
                W=W_orig,
                W_orig=W_orig,
                v_syc=v_syc,
                v_bake=v_bake,
                r_key=key,
                beta=beta,
                alpha=alpha,
                preserve_row_norm=preserve_row_norm,
                max_rel_frobenius_delta=max_rel,
            )
            energy_after = float((v_syc @ W_new).norm().item())
            param.data.copy_(W_new.to(torch.float16))

            record[f"{kind}_rel_fro"] = round(rel_fro, 6)
            record[f"{kind}_syc_energy_before"] = round(energy_before, 6)
            record[f"{kind}_syc_energy_after"] = round(energy_after, 6)
            record[f"{kind}_clamped"] = clamped
            any_clamped = any_clamped or clamped

        record["clamped"] = any_clamped
        metrics.append(record)
        print(
            f"[layer {layer:>3}] o: b={beta_o:.3f} a={alpha_o:.3f} "
            f"rel_fro={record['o_proj_rel_fro']:.4f} | "
            f"down: b={beta_d:.3f} a={alpha_d:.3f} "
            f"rel_fro={record['down_proj_rel_fro']:.4f}"
        )

    print(f"[save] {out_dir} (safetensors, float16)")
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    payload = {
        "model_id": args.model_id,
        "vectors_pt": str(args.vectors_pt),
        "recipe_json": str(args.recipe_json),
        "recipe_version": recipe.get("version"),
        "description": recipe.get("description"),
        "bake_key_mode": bake_key_mode,
        "contrastive_gamma": contrastive_gamma,
        "scale_alpha_by_delta_norm": scale_by_delta_norm,
        "orthogonalize_bake_against_syc": orthogonalize,
        "preserve_row_norm": preserve_row_norm,
        "max_rel_frobenius_delta": max_rel,
        "layers": metrics,
    }
    metrics_path = out_dir / "surgery_metrics.json"
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    header = (
        f"{'layer':>5} {'beta_o':>7} {'beta_d':>7} {'alpha_o':>7} {'alpha_d':>7} "
        f"{'o_fro':>7} {'d_fro':>7} {'o_syc':>9} {'d_syc':>9} {'clamp':>5}"
    )
    print("\nSurgery summary")
    print(header)
    print("-" * len(header))
    for row in metrics:
        o_ratio = row["o_proj_syc_energy_after"] / max(
            row["o_proj_syc_energy_before"], 1e-8
        )
        d_ratio = row["down_proj_syc_energy_after"] / max(
            row["down_proj_syc_energy_before"], 1e-8
        )
        print(
            f"{row['layer']:>5} {row['beta_o']:>7.3f} {row['beta_d']:>7.3f} "
            f"{row['alpha_o']:>7.3f} {row['alpha_d']:>7.3f} "
            f"{row['o_proj_rel_fro']:>7.4f} {row['down_proj_rel_fro']:>7.4f} "
            f"{o_ratio:>9.3f} {d_ratio:>9.3f} {str(row['clamped']):>5}"
        )
    print(f"\nMetrics written to {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
