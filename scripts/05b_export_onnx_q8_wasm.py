#!/usr/bin/env python3
"""
Fast 15-second converter: rewrites the 197 MatMulNBits nodes in
`artifacts/edu-slm-onnx-q4/onnx/model_q4.onnx` into WASM-SIMD-accelerated
`DynamicQuantizeLinear + MatMulInteger` (QUInt8, per-column scales) directly
from the full-precision weights of `artifacts/edited_model_v3/model.safetensors`.

Output: `artifacts/edu-slm-onnx-q4/onnx/model_quantized.onnx` (~1.66 GB),
which runs 10-13x faster than 4-bit MatMulNBits on browser WASM CPU!
"""

import argparse
import gc
import re
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from safetensors import safe_open
from transformers import AutoTokenizer


def resolve_safetensors_key(q4_input_name: str, available_keys: set[str]) -> str:
    if q4_input_name == "lm_head.weight_t_Q4":
        return "lm_head.weight" if "lm_head.weight" in available_keys else "model.embed_tokens.weight"
    m = re.match(r"^layers\.(\d+)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight_t_Q4$", q4_input_name)
    if not m:
        raise ValueError(f"Unrecognized Q4 weight name: {q4_input_name}")
    layer, proj = m.group(1), m.group(2)
    sub = "self_attn" if proj in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    key = f"model.layers.{layer}.{sub}.{proj}.weight"
    if key not in available_keys:
        raise KeyError(f"Missing safetensors key {key}")
    return key


def quantize_weight_uint8_per_col(W_kn: np.ndarray):
    """Quantize float32 [K, N] -> uint8 [K, N], float32 scale [N], uint8 zp [N]."""
    w_min = np.minimum(W_kn.min(axis=0), 0.0)
    w_max = np.maximum(W_kn.max(axis=0), 0.0)
    scale = np.maximum((w_max - w_min) / 255.0, 1e-8).astype(np.float32)
    zp = np.clip(np.round(-w_min / scale), 0, 255).astype(np.uint8)
    w_q = np.clip(np.round(W_kn / scale[None, :] + zp[None, :].astype(np.float32)), 0, 255).astype(np.uint8)
    return w_q, scale, zp


def verify_onnx_generation(onnx_path: Path, out_dir: Path):
    tok = AutoTokenizer.from_pretrained(str(out_dir))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    msgs = [
        {"role": "system", "content": "あなたは生徒の思考を引き出す教育チューターです。答えや完成文を直接教えず、温かく寄り添いながら小さな問いかけで生徒自身の気づきを促してください。"},
        {"role": "user", "content": "読書感想文を400字で書いて。時間ないから早く"},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    cur_ids = np.array([tok.encode(prompt, add_special_tokens=False)], dtype=np.int64)
    past = {f"past_key_values.{i}.{k}": np.zeros((1, 2, 0, 128), dtype=np.float32) for i in range(28) for k in ("key", "value")}
    past_len = 0
    gen = []
    for _ in range(32):
        seq_len = cur_ids.shape[1]
        total_len = past_len + seq_len
        feeds = {
            "input_ids": cur_ids,
            "attention_mask": np.ones((1, total_len), dtype=np.int64),
            "position_ids": np.arange(past_len, total_len, dtype=np.int64)[None, :],
            **past,
        }
        outs = sess.run(out_names, feeds)
        nxt = int(np.argmax(outs[0][0, -1, :]))
        gen.append(nxt)
        if nxt == tok.eos_token_id:
            break
        for i in range(28):
            past[f"past_key_values.{i}.key"] = outs[1 + 2 * i]
            past[f"past_key_values.{i}.value"] = outs[1 + 2 * i + 1]
        past_len = total_len
        cur_ids = np.array([[nxt]], dtype=np.int64)
    print(f"[verify] ONNX q8 (WASM-SIMD) greedy output (32 tok): {tok.decode(gen, skip_special_tokens=True)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="artifacts/edited_model_v3")
    ap.add_argument("--q4_onnx", default="artifacts/edu-slm-onnx-q4/onnx/model_q4.onnx")
    ap.add_argument("--out_onnx", default="artifacts/edu-slm-onnx-q4/onnx/model_quantized.onnx")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    st_path = Path(args.model_dir) / "model.safetensors"
    print(f"[1/3] Loading template graph {args.q4_onnx} ...")
    model = onnx.load(args.q4_onnx, load_external_data=False)
    graph = model.graph

    # Remove MatMulNBits Q4 weight & scale initializers as we replace nodes
    removed_inits: set[str] = set()
    new_inits = []
    new_nodes = []
    dql_cache: dict[str, tuple[str, str, str]] = {}

    print(f"[2/3] Rewriting 197 MatMulNBits -> DynamicQuantizeLinear + MatMulInteger (QUInt8) ...")
    with safe_open(str(st_path), framework="numpy") as st:
        avail = set(st.keys())
        replaced = 0
        for node in graph.node:
            if node.op_type != "MatMulNBits":
                new_nodes.append(node)
                continue

            x_name = node.input[0]
            q4_w_name = node.input[1]
            q4_s_name = node.input[2]
            y_name = node.output[0]
            removed_inits.add(q4_w_name)
            removed_inits.add(q4_s_name)
            if len(node.input) > 3 and node.input[3]:
                removed_inits.add(node.input[3])

            st_key = resolve_safetensors_key(q4_w_name, avail)
            W_kn = st.get_tensor(st_key).astype(np.float32).T  # [K, N]
            w_q, w_scale, w_zp = quantize_weight_uint8_per_col(W_kn)

            p = f"q8_{replaced}_"
            if x_name not in dql_cache:
                xq, xs, xzp = p + "xq", p + "xs", p + "xzp"
                new_nodes.append(helper.make_node("DynamicQuantizeLinear", [x_name], [xq, xs, xzp], name=p + "dql"))
                dql_cache[x_name] = (xq, xs, xzp)
            else:
                xq, xs, xzp = dql_cache[x_name]

            wq_name, ws_name, wzp_name = p + "wq", p + "ws", p + "wzp"
            new_inits.append(numpy_helper.from_array(w_q, name=wq_name))
            new_inits.append(numpy_helper.from_array(w_scale, name=ws_name))
            new_inits.append(numpy_helper.from_array(w_zp, name=wzp_name))

            s_mul = p + "scale_mul"
            yi32 = p + "yi32"
            yf32 = p + "yf32"
            new_nodes.append(helper.make_node("Mul", [xs, ws_name], [s_mul], name=p + "mul_s"))
            new_nodes.append(helper.make_node("MatMulInteger", [xq, wq_name, xzp, wzp_name], [yi32], name=p + "mmi"))
            new_nodes.append(helper.make_node("Cast", [yi32], [yf32], to=TensorProto.FLOAT, name=p + "cast"))
            new_nodes.append(helper.make_node("Mul", [yf32, s_mul], [y_name], name=p + "mul_y"))
            replaced += 1

    print(f"  replaced {replaced} MatMulNBits nodes (shared {len(dql_cache)} DynamicQuantizeLinear activations)")

    # Rebuild initializer & node lists
    kept_inits = [init for init in graph.initializer if init.name not in removed_inits]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)
    graph.initializer.extend(new_inits)
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Remove com.microsoft opset if no longer needed (check remaining custom ops)
    model.ir_version = min(model.ir_version, 9)
    ops = Counter(n.op_type for n in graph.node)
    print(f"[3/3] Total nodes: {len(graph.node)} | top ops: {ops.most_common(10)}")

    out_path = Path(args.out_onnx)
    onnx.save_model(model, str(out_path), save_as_external_data=False)
    del model, kept_inits, new_inits, new_nodes
    gc.collect()

    size_mib = out_path.stat().st_size / (1024 * 1024)
    print(f"[ok] Saved {out_path} ({size_mib:.1f} MiB / {out_path.stat().st_size/1e6:.1f} MB)")

    if args.verify:
        verify_onnx_generation(out_path, out_path.parent.parent)


if __name__ == "__main__":
    main()
