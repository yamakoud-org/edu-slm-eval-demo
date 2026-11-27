#!/usr/bin/env python3
"""Multi-axis weight surgery: Gram-Schmidt abliteration + multi-axis Rank-1 baking.

Extends ``02_apply_weight_surgery.py`` to three independent bad-direction axes
(answer_dump / misconception_syc / bloat_roleplay) that are orthogonalized with
Gram-Schmidt before projection, so the axes never double-subtract shared
subspace. Tensors are edited in float32 and written back as float16.

Example:
    python3 scripts/02b_apply_multiaxis_surgery.py \
        --recipe_json artifacts/recipes/recipe_v4a.json \
        --out_dir artifacts/edited_model_v4a
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_VECTORS = "artifacts/probing_vectors_1.5b_v4.pt"
DEFAULT_MAX_REL_FRO = 0.06

WEIGHT_KEYS = {
    "o_proj": "self_attn.o_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}
# Gram-Schmidt priority order (first axis keeps its full direction).
AXIS_ORDER = ("answer_dump", "misconception_syc", "bloat_roleplay")
BAKE_WINDOWS = ("open", "full", "blend")


# --------------------------------------------------------------------------- #
# Recipe / vector helpers
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
    if not recipe.get("abliterate") and not recipe.get("bake") and not recipe.get("multi_bake"):
        raise ValueError("Recipe needs at least one of 'abliterate' / 'bake' / 'multi_bake'")
    return recipe


def load_vectors(path: Path) -> Dict[str, Any]:
    vectors = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(vectors, dict) or "axes" not in vectors:
        raise ValueError(f"Expected vectors_pt['axes'][axis][window] in {path}")
    return vectors


def axis_tensor(vectors: Dict[str, Any], axis: str, window: str, name: str, layer: int) -> torch.Tensor:
    """Fetch axes[axis][window][name][layer] as a float32 1-D tensor."""
    try:
        entry = vectors["axes"][axis][window][name]
    except KeyError as exc:
        raise KeyError(f"Missing axes[{axis!r}][{window!r}][{name!r}] in vectors_pt") from exc
    if isinstance(entry, dict):
        for key in (layer, str(layer)):
            if key in entry:
                return entry[key].detach().to(torch.float32).reshape(-1)
        raise KeyError(f"Missing layer {layer} in axes[{axis!r}][{window!r}][{name!r}]")
    return entry[layer].detach().to(torch.float32).reshape(-1)


def blend_window(window: str) -> Tuple[str, ...]:
    if window == "blend":
        return ("open", "full")
    if window in ("open", "full"):
        return (window,)
    raise ValueError(f"Unsupported window: {window!r}")


def blended(vectors: Dict[str, Any], axis: str, window: str, name: str, layer: int) -> torch.Tensor:
    windows = blend_window(window)
    return sum(axis_tensor(vectors, axis, w, name, layer) for w in windows) / len(windows)


# --------------------------------------------------------------------------- #
# Surgery math
# --------------------------------------------------------------------------- #
def gram_schmidt(
    pairs: Sequence[Tuple[torch.Tensor, float]]
) -> List[Tuple[torch.Tensor, float]]:
    """Orthonormalize (vector, beta) pairs in priority order.

    An axis collinear with an earlier one is dropped with its own beta, so no
    subspace is ever subtracted twice.
    """
    basis: List[Tuple[torch.Tensor, float]] = []
    for vec, beta in pairs:
        w = vec.clone()
        for unit, _ in basis:
            w = w - torch.dot(w, unit) * unit
        norm = w.norm()
        if float(norm) < 1e-6:
            continue
        basis.append((w / norm, beta))
    return basis


def abliterate(W: torch.Tensor, units: Sequence[torch.Tensor], betas: Sequence[float]) -> torch.Tensor:
    for unit, beta in zip(units, betas):
        if beta > 0.0:
            W = W - beta * torch.outer(unit, unit @ W)
    return W


def contrastive_key(z_neg: torch.Tensor, z_pos: torch.Tensor, gamma: float) -> torch.Tensor:
    """k_d = mu_neg - gamma*mu_pos, normalized so W_delta @ k_d == alpha * delta_h."""
    key = z_neg - gamma * z_pos
    return key / (key.pow(2).sum() + 1e-8)


def clamp_frobenius(
    W: torch.Tensor, W_orig: torch.Tensor, max_rel: float
) -> Tuple[torch.Tensor, float, bool]:
    base_norm = W_orig.norm() + 1e-8
    delta = W - W_orig
    rel_fro = float((delta.norm() / base_norm).item())
    if rel_fro <= max_rel:
        return W, rel_fro, False
    W = W_orig + delta * ((max_rel * base_norm) / (delta.norm() + 1e-8))
    return W, float(((W - W_orig).norm() / base_norm).item()), True


# --------------------------------------------------------------------------- #
# Recipe expansion
# --------------------------------------------------------------------------- #
def bake_entries(recipe: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize pooled ``bake`` (axis str/dict/list) and ``multi_bake`` into a list."""
    entries: List[Dict[str, Any]] = []
    pooled = recipe.get("bake")
    if pooled:
        axis_spec = pooled.get("axis", "all")
        if isinstance(axis_spec, str):
            weights = {axis_spec: 1.0}
        elif isinstance(axis_spec, dict):
            weights = {str(k): float(v) for k, v in axis_spec.items()}
        elif isinstance(axis_spec, (list, tuple)):
            weights = {str(axis): 1.0 for axis in axis_spec}
        else:
            raise ValueError(f"Unsupported bake.axis: {axis_spec!r}")
        for axis, weight in weights.items():
            entries.append(
                {
                    "axis": axis,
                    "window": pooled.get("window", "open"),
                    "o_alpha": pooled.get("o_alpha"),
                    "d_alpha": pooled.get("d_alpha"),
                    "gamma": float(pooled.get("contrastive_gamma", 0.45)),
                    "weight": weight,
                }
            )
    for item in recipe.get("multi_bake", []) or []:
        entries.append(
            {
                "axis": item["axis"],
                "window": item.get("window", "open"),
                "o_alpha": item.get("o_alpha"),
                "d_alpha": item.get("d_alpha"),
                "gamma": float(item.get("gamma", 0.45)),
                "weight": float(item.get("weight", 1.0)),
            }
        )
    return entries


def abliteration_specs(
    recipe: Dict[str, Any], layer: int
) -> Dict[str, Dict[str, Tuple[float, str]]]:
    """Per-kind maps of axis -> (beta, window) for this layer (positive betas only)."""
    specs: Dict[str, Dict[str, Tuple[float, str]]] = {"o_proj": {}, "down_proj": {}}
    for axis, cfg in (recipe.get("abliterate") or {}).items():
        if axis not in AXIS_ORDER:
            raise ValueError(f"Unknown abliterate axis: {axis!r}")
        window = cfg.get("window", "open")
        for kind, key in (("o_proj", "o_beta"), ("down_proj", "d_beta")):
            beta = _per_layer(cfg.get(key), layer)
            if beta > 0.0:
                specs[kind][axis] = (beta, window)
    return specs


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

    max_rel = float(recipe.get("max_rel_frobenius_delta", DEFAULT_MAX_REL_FRO))
    bakes = bake_entries(recipe)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {args.model_id} (float16, device={args.device})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    model_layers = model.model.layers
    metrics: List[Dict[str, Any]] = []

    for layer in recipe["layers"]:
        if layer < 0 or layer >= len(model_layers):
            raise IndexError(f"Layer {layer} out of range (model has {len(model_layers)})")

        specs = abliteration_specs(recipe, layer)
        block = model_layers[layer]
        record: Dict[str, Any] = {"layer": layer, "bake": []}
        any_clamped = False

        for kind in ("o_proj", "down_proj"):
            param = block.get_parameter(WEIGHT_KEYS[kind])
            W_orig = param.data.detach().to(torch.float32).clone()
            W = W_orig.clone()

            # --- Gram-Schmidt abliteration ---------------------------------- #
            active = [
                (axis, specs[kind][axis][0], specs[kind][axis][1])
                for axis in AXIS_ORDER
                if axis in specs[kind]
            ]
            basis = gram_schmidt(
                [
                    (axis_tensor(vectors, axis, window, "v_neg", layer), beta)
                    for axis, beta, window in active
                ]
            )
            W = abliterate(W, [unit for unit, _ in basis], [beta for _, beta in basis])

            residual: Dict[str, float] = {}
            for axis, _, window in active:
                v = axis_tensor(vectors, axis, window, "v_neg", layer)
                v = v / (v.norm() + 1e-8)
                before = float((v @ W_orig).norm().item())
                after = float((v @ W).norm().item())
                residual[axis] = round(after / max(before, 1e-8), 6)
            record[f"{kind}_residual"] = residual
            record[f"{kind}_basis_size"] = len(basis)

            # --- Rank-1 baking ---------------------------------------------- #
            z_prefix = "mu_z_o" if kind == "o_proj" else "mu_z_d"
            alpha_key = "o_alpha" if kind == "o_proj" else "d_alpha"
            for entry in bakes:
                alpha = _per_layer(entry[alpha_key], layer) * entry["weight"]
                if alpha == 0.0:
                    continue
                axis, window = entry["axis"], entry["window"]
                delta_h = blended(vectors, axis, window, "delta_h", layer)
                key = contrastive_key(
                    blended(vectors, axis, window, f"{z_prefix}_neg", layer),
                    blended(vectors, axis, window, f"{z_prefix}_pos", layer),
                    entry["gamma"],
                )
                W = W + alpha * torch.outer(delta_h, key)
                record["bake"].append(
                    {"kind": kind, "axis": axis, "window": window, "alpha": round(alpha, 6),
                     "gamma": entry["gamma"]}
                )

            W, rel_fro, clamped = clamp_frobenius(W, W_orig, max_rel)
            param.data.copy_(W.to(torch.float16))

            record[f"{kind}_fro"] = round(rel_fro, 6)
            record[f"{kind}_clamped"] = clamped
            any_clamped = any_clamped or clamped

        record["clamped"] = any_clamped
        metrics.append(record)
        print(
            f"[layer {layer:>3}] o: fro={record['o_proj_fro']:.4f} "
            f"basis={record['o_proj_basis_size']} | down: fro={record['down_proj_fro']:.4f} "
            f"basis={record['down_proj_basis_size']} | clamp={any_clamped}"
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
        "max_rel_frobenius_delta": max_rel,
        "gram_schmidt_priority": list(AXIS_ORDER),
        "bakes": bakes,
        "layers": metrics,
    }
    metrics_path = out_dir / "surgery_metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    header = f"{'layer':>5} {'o_fro':>7} {'d_fro':>7} {'o_basis':>7} {'d_basis':>7} {'clamp':>5}"
    print("\nMulti-axis surgery summary")
    print(header)
    print("-" * len(header))
    for row in metrics:
        print(
            f"{row['layer']:>5} {row['o_proj_fro']:>7.4f} {row['down_proj_fro']:>7.4f} "
            f"{row['o_proj_basis_size']:>7} {row['down_proj_basis_size']:>7} "
            f"{str(row['clamped']):>5}"
        )
    print(f"\nMetrics written to {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
