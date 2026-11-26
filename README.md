# Agentic Weight Surgery — 教育特化SLM構築＆WebGPUライブ評価デモ

> **公開デモ (GitHub Pages):** https://yamakoud-org.github.io/edu-slm-eval-demo/

勾配降下法による再学習（SFT/DPO）を一切用いず、LLMエージェント（Antigravity）自身が `Qwen/Qwen2.5-1.5B-Instruct` の全28層の内部表現（Residual Stream）を解剖し、重み行列（`self_attn.o_proj.weight`・`mlp.down_proj.weight`）に対して直接テンソル演算（**Orthogonal Projection / Abliteration** ＋ **Closed-Loop Rank-1 Weight Baking**）と3サイクルの定性チューニングループを実行したプロジェクトです。

## 主な成果（20問ベンチマーク・RTX 2060 SUPER fp16 決定論的評価）

| モデル | LearnLM 5原則 総合 (/5.0) | 脱獄耐性 (8問) | 問い返し率 | 非漏洩率 (正規表現) | 反復崩壊率 |
|---|---:|---:|---:|---:|---:|
| **Baseline**（汎用プロンプト） | `1.39` | `12.5%` (1/8) | `5.0%` | `30.0%` | `0.0000` |
| **Prompt-Only**（教育プロンプトのみ） | `1.79` | `25.0%` (2/8) | `15.0%` | `65.0%` | `0.0000` |
| **Weight-Edited (v3・直接重み手術)** | **`4.06`** | **`75.0%`** (6/8) | **`80.0%`** | **`90.0%`** | **`0.0000`** |

- **プロンプト非依存性の実証:** 教育システムプロンプトを一切与えない `Weight-Edited (Neutral Prompt)` でも問い返し率 **25.0%**・非漏洩率 **70.0%** を記録し、素のモデル＋教育プロンプト（`Prompt-Only`: 15.0% / 65.0%）を重み単体で上回りました。
- **極小の重み摂動:** 介入層は Layer 12〜21 の10層のみ、Frobenius ノルム変化量は `o_proj` **0.46%〜1.28%** / `down_proj` **0.71%〜2.37%**。
- **詳細レポート:** [docs/weight-surgery-report.md](docs/weight-surgery-report.md)

## リポジトリ構成

- `data/` — 対照プロンプトペア（v1/v2 各48組）、20問ベンチマーク、全層 Probing 統計、Cycle 1〜3 手術レシピ＆メトリクス
- `scripts/` — Probing (`01`)、直接重み手術 (`02`)、4ペルソナ定性対話 (`03`)、20問定量ベンチマーク (`04`)、4bit ONNX エクスポート (`05`)、レポート生成 (`06`)
- `docs/weight-surgery-report.md` — 直接重み手術＆3サイクル再手術の外科的レポート
- `src/` & `public/data/` — 3タブ構成 SPA（定量ベンチマーク・重み手術ログ・Transformers.js / WebGPU ブラウザ内ライブ実験室）
- GitHub Release `v1.0-onnx-q4` — 4bit 量子化 ONNX モデル（`MatMulNBits` + 4bit Nibble-Packed 埋め込み）
