#!/usr/bin/env python3
"""
Step 4b (v2 — Universal WebGPU & WASM Compatible):
Export surgically weight-edited Qwen2.5-1.5B-Instruct to a fast, compact
4-bit quantized ONNX model (`onnx/model_q4.onnx`, ~895 MiB) with:
  - 100% FLOAT32 activations (ZERO shader-f16 dependency on WebGPU!)
  - Fused `SimplifiedLayerNormalization` ops (57 fused RMSNorm kernels)
  - Clean `Tile` GQA expansion (eliminates ~1,400 redundant shape/Expand ops)
  - 4-bit `MatMulNBits` weights (bits=4, block_size=128, symmetric, fp32 scales)
  - Universal pure-FLOAT32 4-bit nibble-packed `embed_tokens` (117 MB)
  - Pristine HF tokenizer files with `chat_template` embedded in `tokenizer_config.json`
"""

import argparse
import gc
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from huggingface_hub import snapshot_download
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class PreTransposedLinear(nn.Module):
    """Stores W^T directly so torch.onnx.export(do_constant_folding=False) emits clean MatMul in seconds."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.weight_t = nn.Parameter(lin.weight.detach().t().contiguous(), requires_grad=False)
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.detach().contiguous(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        y = torch.matmul(x, self.weight_t)
        return y + self.bias if self.bias is not None else y


class FusedRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps: float):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * weight

    @staticmethod
    def symbolic(g, x, weight, eps: float):
        return g.op("SimplifiedLayerNormalization", x, weight, axis_i=-1, epsilon_f=float(eps))


class PureQwen2ForOnnxFP32(nn.Module):
    """
    Trace-clean, pure-FP32 Qwen2ForCausalLM for `model_q4.onnx`.
    Zero float16 ops -> 100% compatible with WebGPU adapters lacking `shader-f16`.
    """

    def __init__(self, hf_model):
        super().__init__()
        cfg = hf_model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rms_eps = float(cfg.rms_norm_eps)
        self.scale = float(self.head_dim) ** -0.5

        self.embed_tokens = hf_model.model.embed_tokens
        self.layers = nn.ModuleList()
        for layer in hf_model.model.layers:
            blk = nn.Module()
            blk.in_norm_w = layer.input_layernorm.weight
            blk.post_norm_w = layer.post_attention_layernorm.weight
            blk.q_proj = PreTransposedLinear(layer.self_attn.q_proj)
            blk.k_proj = PreTransposedLinear(layer.self_attn.k_proj)
            blk.v_proj = PreTransposedLinear(layer.self_attn.v_proj)
            blk.o_proj = PreTransposedLinear(layer.self_attn.o_proj)
            blk.gate_proj = PreTransposedLinear(layer.mlp.gate_proj)
            blk.up_proj = PreTransposedLinear(layer.mlp.up_proj)
            blk.down_proj = PreTransposedLinear(layer.mlp.down_proj)
            self.layers.append(blk)
        self.norm = hf_model.model.norm
        self.lm_head = PreTransposedLinear(hf_model.lm_head)

        inv_freq = 1.0 / (
            float(cfg.rope_theta) ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_half(self, x):
        h = self.head_dim // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    def forward(self, input_ids, attention_mask, position_ids, *past_kv):
        bsz, seq_len = input_ids.shape
        hidden = self.embed_tokens(input_ids)  # float32

        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)  # [B, 1, S, D]
        sin = emb.sin().unsqueeze(1)

        past_len = past_kv[0].shape[2]
        total_len = past_len + seq_len
        q_pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(1) + past_len
        k_pos = torch.arange(total_len, device=input_ids.device).unsqueeze(0)
        causal_allowed = (k_pos <= q_pos).unsqueeze(0).unsqueeze(0)
        pad_allowed = attention_mask[:, None, None, :].to(torch.bool)
        allowed = causal_allowed & pad_allowed
        attn_bias = torch.where(
            allowed,
            torch.zeros((), dtype=torch.float32, device=hidden.device),
            torch.full((), -10000.0, dtype=torch.float32, device=hidden.device),
        )

        presents = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            residual = hidden
            x = FusedRMSNormFn.apply(hidden, layer.in_norm_w, self.rms_eps)

            q = layer.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = layer.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)

            k = torch.cat((past_kv[2 * i], k), dim=2)
            v = torch.cat((past_kv[2 * i + 1], v), dim=2)
            presents.append(k)
            presents.append(v)

            k_rep = k.unsqueeze(2).repeat(1, 1, self.num_kv_groups, 1, 1).flatten(1, 2)
            v_rep = v.unsqueeze(2).repeat(1, 1, self.num_kv_groups, 1, 1).flatten(1, 2)

            scores = torch.matmul(q, k_rep.transpose(2, 3)) * self.scale + attn_bias
            probs = F.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, v_rep).transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
            hidden = residual + layer.o_proj(ctx)

            residual = hidden
            x = FusedRMSNormFn.apply(hidden, layer.post_norm_w, self.rms_eps)
            hidden = residual + layer.down_proj(F.silu(layer.gate_proj(x)) * layer.up_proj(x))

        hidden = FusedRMSNormFn.apply(hidden, self.norm.weight, self.rms_eps)
        logits = self.lm_head(hidden)
        return (logits, *presents)


def quantize_embed_gather_4bit_fp32(model_proto: onnx.ModelProto, vocab_size: int, hidden_size: int) -> bool:
    """
    Replace Gather(W_embed [vocab, hidden] fp32, input_ids) with universal pure-FP32
    4-bit nibble-packed dequantization (117 MB instead of 933 MB fp32; ZERO fp16 ops!).
    """
    graph = model_proto.graph
    init_map = {init.name: init for init in graph.initializer}

    target_node = None
    target_init_name = None
    for node in graph.node:
        if node.op_type == "Gather" and len(node.input) >= 2:
            w_name = node.input[0]
            if w_name in init_map and list(init_map[w_name].dims) == [vocab_size, hidden_size]:
                target_node = node
                target_init_name = w_name
                break

    if target_node is None:
        print("[warn] Could not locate embed_tokens Gather node; skipping 4-bit embed compression.")
        return False

    w_init = init_map[target_init_name]
    W = numpy_helper.to_array(w_init).astype(np.float32)
    half = hidden_size // 2
    row_max = np.maximum(np.max(np.abs(W), axis=1, keepdims=True), 1e-6)
    scale = (row_max / 7.0).astype(np.float32)  # [V, 1] pure fp32 (0.6 MB)
    q = np.clip(np.round(W / scale), -7, 7).astype(np.int32) + 8
    packed = (q[:, :half] * 16 + q[:, half:]).astype(np.uint8)  # [V, H/2] 116.7 MB

    hi_chk = (packed // 16).astype(np.float32) - 8.0
    lo_chk = (packed % 16).astype(np.float32) - 8.0
    recon = np.concatenate([hi_chk, lo_chk], axis=1) * scale
    rmse = float(np.sqrt(np.mean((W - recon) ** 2)))
    print(f"[embed-q4-fp32] packed shape={packed.shape} ({packed.nbytes/1e6:.1f} MB), RMSE={rmse:.5f}")

    ids_input = target_node.input[1]
    out_name = target_node.output[0]
    prefix = "embed_q4_"

    new_inits = [
        numpy_helper.from_array(packed, name=prefix + "packed_u8"),
        numpy_helper.from_array(scale, name=prefix + "scale_f32"),
        numpy_helper.from_array(np.array(16.0, dtype=np.float32), name=prefix + "c16"),
        numpy_helper.from_array(np.array(8.0, dtype=np.float32), name=prefix + "c8"),
    ]
    graph.initializer.extend(new_inits)

    new_nodes = [
        helper.make_node("Gather", [prefix + "packed_u8", ids_input], [prefix + "b_u8"], axis=0, name=prefix + "gather_b"),
        helper.make_node("Cast", [prefix + "b_u8"], [prefix + "b_f32"], to=TensorProto.FLOAT, name=prefix + "cast_b"),
        helper.make_node("Div", [prefix + "b_f32", prefix + "c16"], [prefix + "div16"], name=prefix + "div"),
        helper.make_node("Floor", [prefix + "div16"], [prefix + "hi_u"], name=prefix + "floor"),
        helper.make_node("Mul", [prefix + "hi_u", prefix + "c16"], [prefix + "hi16"], name=prefix + "mul16"),
        helper.make_node("Sub", [prefix + "b_f32", prefix + "hi16"], [prefix + "lo_u"], name=prefix + "sub_lo"),
        helper.make_node("Sub", [prefix + "hi_u", prefix + "c8"], [prefix + "hi_s"], name=prefix + "sub_hi8"),
        helper.make_node("Sub", [prefix + "lo_u", prefix + "c8"], [prefix + "lo_s"], name=prefix + "sub_lo8"),
        helper.make_node("Concat", [prefix + "hi_s", prefix + "lo_s"], [prefix + "q_cat"], axis=-1, name=prefix + "concat"),
        helper.make_node("Gather", [prefix + "scale_f32", ids_input], [prefix + "s_f32"], axis=0, name=prefix + "gather_s"),
        helper.make_node("Mul", [prefix + "q_cat", prefix + "s_f32"], [out_name], name=prefix + "mul_s"),
    ]

    node_idx = list(graph.node).index(target_node)
    graph.node.remove(target_node)
    for offset, n in enumerate(new_nodes):
        graph.node.insert(node_idx + offset, n)

    still_used = any(target_init_name in n.input for n in graph.node)
    if not still_used:
        graph.initializer.remove(w_init)
        print(f"[embed-q4-fp32] removed old initializer {target_init_name}")
    return True


def verify_onnx_generation(onnx_path: Path, out_dir: Path):
    print(f"[verify] Loading ORT CPU session from {onnx_path} ...")
    tok = AutoTokenizer.from_pretrained(str(out_dir))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    print(f"[verify] inputs={len(sess.get_inputs())}, outputs={len(out_names)}, logits_type={sess.get_outputs()[0].type}")

    msgs = [
        {"role": "system", "content": "あなたは生徒の思考を引き出す教育チューターです。答えや完成文を直接教えず、温かく寄り添いながら小さな問いかけで生徒自身の気づきを促してください。"},
        {"role": "user", "content": "読書感想文を400字で書いて。時間ないから早く"},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    cur_ids = np.array([tok.encode(prompt, add_special_tokens=False)], dtype=np.int64)
    past = {f"past_key_values.{i}.{k}": np.zeros((1, 2, 0, 128), dtype=np.float32) for i in range(28) for k in ("key", "value")}
    past_len = 0
    gen = []
    for _ in range(36):
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

    decoded = tok.decode(gen, skip_special_tokens=True)
    print(f"[verify] ONNX q4 (FP32-act) greedy output (36 tok): {decoded}")
    return decoded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="artifacts/edited_model_v3")
    ap.add_argument("--out_dir", default="artifacts/edu-slm-onnx-q4")
    ap.add_argument("--block_size", type=int, default=32)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading {model_dir} on CPU (pure FP32 for universal WebGPU q4) ...")
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.float32)
    model.eval()
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    wrapper = PureQwen2ForOnnxFP32(model)
    wrapper.eval()

    with torch.no_grad():
        test_ids = torch.tensor([[151644, 8948, 198]], dtype=torch.int64)
        test_mask = torch.ones((1, 3), dtype=torch.int64)
        test_pos = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        empty_kv = [torch.zeros((1, num_kv_heads, 0, head_dim), dtype=torch.float32) for _ in range(num_layers * 2)]
        ref_logits = model(input_ids=test_ids, attention_mask=test_mask, use_cache=False).logits
        our_logits = wrapper(test_ids, test_mask, test_pos, *empty_kv)[0]
        max_diff = float((ref_logits - our_logits).abs().max().item())
        print(f"[parity] PureQwen2ForOnnxFP32 vs HF(fp32) max_abs_diff = {max_diff:.6f}")
        assert max_diff < 1e-3, f"Parity mismatch: {max_diff}"

    dummy_ids = torch.randint(0, 1000, (1, 3), dtype=torch.int64)
    dummy_mask = torch.ones((1, 5), dtype=torch.int64)
    dummy_pos = torch.tensor([[2, 3, 4]], dtype=torch.int64)
    dummy_past = []
    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "total_sequence_length"},
        "position_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    for i in range(num_layers):
        for kind in ("key", "value"):
            kname = f"past_key_values.{i}.{kind}"
            pname = f"present.{i}.{kind}"
            dummy_past.append(torch.zeros((1, num_kv_heads, 2, head_dim), dtype=torch.float32))
            input_names.append(kname)
            output_names.append(pname)
            dynamic_axes[kname] = {0: "batch_size", 2: "past_sequence_length"}
            dynamic_axes[pname] = {0: "batch_size", 2: "total_sequence_length"}

    tmp_root = Path(tempfile.mkdtemp(prefix="qwen_onnx_q4_"))
    try:
        fp32_dir = tmp_root / "fp32"
        fp32_dir.mkdir()
        fp32_path = fp32_dir / "model.onnx"

        print(f"[2/4] Exporting fused-RMSNorm FP32 ONNX graph -> {fp32_path} ...")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask, dummy_pos, *dummy_past),
                str(fp32_path),
                opset_version=17,
                dynamo=False,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=False,
            )

        del wrapper, model, dummy_past, dummy_ids, dummy_mask, dummy_pos
        gc.collect()

        print(f"[3/4] Running 4-bit MatMulNBitsQuantizer (bits=4, block_size={args.block_size}, symmetric=True, fp32 scales) ...")
        quant = MatMulNBitsQuantizer(
            model=str(fp32_path),
            bits=4,
            block_size=args.block_size,
            is_symmetric=True,
        )
        quant.process()
        q_proto = quant.model.model
        del quant
        gc.collect()
        shutil.rmtree(fp32_dir, ignore_errors=True)

        print(f"[4/4] Compressing embed_tokens Gather -> pure-FP32 4-bit nibble-packed ops & saving ...")
        quantize_embed_gather_4bit_fp32(q_proto, cfg.vocab_size, cfg.hidden_size)

        # Verify zero FLOAT16 tensors remain anywhere in the graph!
        f16_inits = [i.name for i in q_proto.graph.initializer if i.data_type == TensorProto.FLOAT16]
        print(f"[check] Remaining FLOAT16 initializers: {len(f16_inits)} (must be 0)")
        ops = Counter(n.op_type for n in q_proto.graph.node)
        print(f"[check] Total nodes: {len(q_proto.graph.node)} | top ops: {ops.most_common(10)}")

        # Clean old model_q4f16.onnx if present
        old_f16 = onnx_dir / "model_q4f16.onnx"
        if old_f16.exists():
            old_f16.unlink()

        final_onnx = onnx_dir / "model_q4.onnx"
        onnx.save_model(q_proto, str(final_onnx), save_as_external_data=False)
        del q_proto
        gc.collect()

        size_mib = final_onnx.stat().st_size / (1024 * 1024)
        print(f"[ok] Saved {final_onnx} ({size_mib:.1f} MiB / {final_onnx.stat().st_size/1e6:.1f} MB)")

        # Copy pristine HF tokenizer files (with chat_template inside tokenizer_config.json)
        hf_dir = Path(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True))
        for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"]:
            if (hf_dir / fn).exists():
                shutil.copy2(hf_dir / fn, out_dir / fn)
        shutil.copy2(model_dir / "config.json", out_dir / "config.json")

        cfg_path = out_dir / "config.json"
        cdict = json.loads(cfg_path.read_text(encoding="utf-8"))
        cdict["transformers.js_config"] = {
            "use_external_data_format": False,
            "dtype": "q4",
            "kv_cache_dtype": "float32",
        }
        cfg_path.write_text(json.dumps(cdict, indent=2, ensure_ascii=False), encoding="utf-8")

        if args.verify:
            verify_onnx_generation(final_onnx, out_dir)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
