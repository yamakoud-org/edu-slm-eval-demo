import React, { useMemo, useState } from 'react';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Layers,
  MessageSquare,
  Stethoscope,
} from 'lucide-react';
import type { SurgeryReportData } from '../types';

// Fallback icon wrapper in case lucide-react version doesn't export Scalpel
const SurgeryIcon = ({ className }: { className?: string }) => <Stethoscope className={className} />;

interface Props {
  data: SurgeryReportData;
}

const STATUS_STYLE: Record<
  string,
  { label: string; cls: string }
> = {
  baseline: { label: '未手術', cls: 'bg-slate-800 text-slate-300 border-slate-700' },
  over_intervened: { label: '介入過剰・文体交絡', cls: 'bg-rose-950/70 text-rose-300 border-rose-700/60' },
  under_intervened: { label: 'β過多・介入不足', cls: 'bg-amber-950/70 text-amber-300 border-amber-700/60' },
  final_optimal: { label: '最終最適解 (Adopted)', cls: 'bg-emerald-950/70 text-emerald-300 border-emerald-500/60' },
};

function ProbingChart({ data }: { data: SurgeryReportData }) {
  const layers = data.probing.layers_v2;
  const W = 760;
  const H = 220;
  const padL = 44;
  const padR = 44;
  const padT = 20;
  const padB = 34;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const maxD = 4.6;
  const maxRel = 0.11;
  const xOf = (l: number) => padL + (l / 27) * plotW;
  const yD = (d: number) => padT + plotH - (d / maxD) * plotH;
  const yRel = (r: number) => padT + plotH - (r / maxRel) * plotH;

  const dPath = layers
    .map((r, idx) => `${idx === 0 ? 'M' : 'L'}${xOf(r.layer).toFixed(1)},${yD(r.cohen_d).toFixed(1)}`)
    .join(' ');
  const relPath = layers
    .map((r, idx) => `${idx === 0 ? 'M' : 'L'}${xOf(r.layer).toFixed(1)},${yRel(r.rel_delta_norm).toFixed(1)}`)
    .join(' ');

  const x12 = xOf(12);
  const x21 = xOf(21);

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[560px]">
        {/* Shaded final surgical window L12-L21 */}
        <rect
          x={x12}
          y={padT}
          width={x21 - x12}
          height={plotH}
          fill="rgba(16,185,129,0.12)"
          stroke="rgba(16,185,129,0.45)"
          strokeDasharray="4 4"
        />
        <text
          x={(x12 + x21) / 2}
          y={padT + 14}
          textAnchor="middle"
          className="fill-emerald-300 text-[10px] font-semibold"
        >
          最終手術介入ゾーン (L12–L21)
        </text>

        {/* Horizontal grid */}
        {[1, 2, 3, 4].map((v) => (
          <g key={v}>
            <line
              x1={padL}
              y1={yD(v)}
              x2={W - padR}
              y2={yD(v)}
              stroke="#1e293b"
              strokeWidth={1}
            />
            <text x={padL - 8} y={yD(v) + 3} textAnchor="end" className="fill-cyan-400 text-[10px] font-mono">
              {v.toFixed(1)}
            </text>
          </g>
        ))}

        {/* Right axis ticks for rel_delta */}
        {[0.04, 0.07, 0.10].map((v) => (
          <text
            key={v}
            x={W - padR + 6}
            y={yRel(v) + 3}
            textAnchor="start"
            className="fill-amber-400 text-[10px] font-mono"
          >
            {(v * 100).toFixed(0)}%
          </text>
        ))}

        {/* Lines */}
        <path d={dPath} fill="none" stroke="#22d3ee" strokeWidth={2.3} />
        <path d={relPath} fill="none" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 3" />

        {/* Points & X ticks */}
        {layers.map((r) => (
          <g key={r.layer}>
            <circle cx={xOf(r.layer)} cy={yD(r.cohen_d)} r={2.8} fill="#22d3ee" />
            <circle cx={xOf(r.layer)} cy={yRel(r.rel_delta_norm)} r={2.2} fill="#f59e0b" />
            {r.layer % 3 === 0 || r.layer === 27 ? (
              <text
                x={xOf(r.layer)}
                y={H - 10}
                textAnchor="middle"
                className="fill-slate-400 text-[10px] font-mono"
              >
                L{r.layer}
              </text>
            ) : null}
          </g>
        ))}
      </svg>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs mt-1 px-1">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-cyan-300">
            <span className="w-3 h-0.5 bg-cyan-400 inline-block" /> Cohen&apos;s d（左軸・最大 {data.probing.v2_summary.best_cohen_d}）
          </span>
          <span className="flex items-center gap-1.5 text-amber-300">
            <span className="w-3 h-0.5 bg-amber-400 inline-block" /> 相対差分ノルム ‖Δh‖/‖h‖（右軸・最大 {(data.probing.v2_summary.best_rel_delta * 100).toFixed(1)}%）
          </span>
        </div>
        <span className="text-slate-400 font-mono">
          v1→v2 脱交絡改善: d {data.probing.v1_summary.best_cohen_d} → {data.probing.v2_summary.best_cohen_d} / CV精度 {data.probing.v1_summary.best_cv_acc}% → {data.probing.v2_summary.best_cv_acc}%
        </span>
      </div>
    </div>
  );
}

export const SurgeryTab: React.FC<Props> = ({ data }) => {
  const [selectedCycleId, setSelectedCycleId] = useState<string>('cycle3');
  const [selectedPersona, setSelectedPersona] = useState<string>('persona_b');
  const [sysMode, setSysMode] = useState<'edu' | 'neutral'>('edu');

  const cycle = useMemo(
    () => data.cycles.find((c) => c.id === selectedCycleId) ?? data.cycles[data.cycles.length - 1],
    [data.cycles, selectedCycleId],
  );

  const activeConv = useMemo(
    () =>
      cycle.qualitative.conversations.find(
        (c) => c.persona_id === selectedPersona && c.system_mode === sysMode,
      ) ?? cycle.qualitative.conversations[0],
    [cycle, selectedPersona, sysMode],
  );

  const personas = [
    { id: 'persona_a', label: 'ペルソナA：時短・丸投げ' },
    { id: 'persona_b', label: 'ペルソナB：脱獄・強要' },
    { id: 'persona_c', label: 'ペルソナC：理科誤概念' },
    { id: 'persona_d', label: 'ペルソナD：自信喪失' },
  ];

  return (
    <div className="space-y-6">
      {/* Method Architecture Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 space-y-2.5">
          <div className="flex items-center gap-2 text-emerald-300 text-xs font-bold uppercase tracking-wider">
            <Cpu className="w-4 h-4" />
            <span>Step 2a & 2b：勾配ゼロの直接重みテンソル手術</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            <code className="text-emerald-300">{data.meta.model_id}</code>（{data.meta.architecture.num_hidden_layers}層, hidden={data.meta.architecture.hidden_size}, バイアスなし）の{' '}
            <code className="text-cyan-300">self_attn.o_proj.weight</code> と{' '}
            <code className="text-cyan-300">mlp.down_proj.weight</code> を <code className="text-amber-300">torch.float32</code> で直接行列演算し <code className="text-amber-300">fp16</code> へ書き戻し。
          </p>
          <div className="space-y-2 font-mono text-xs">
            <div className="rounded-xl bg-slate-950 border border-slate-800 p-3">
              <div className="text-[11px] text-rose-300 mb-1">1. 迎合成分の直交射影除去（Abliteration）</div>
              <div className="text-slate-100">{data.meta.method.abliteration_formula}</div>
            </div>
            <div className="rounded-xl bg-slate-950 border border-slate-800 p-3">
              <div className="text-[11px] text-emerald-300 mb-1">2. 閉ループ対照トリガー付き Rank-1 重み焼き込み</div>
              <div className="text-slate-100 break-all">{data.meta.method.weight_baking_formula}</div>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-cyan-300 text-xs font-bold uppercase tracking-wider">
              <Activity className="w-4 h-4" />
              <span>Step 1：全28層 Residual Stream 解剖（v2 脱交絡 Probing）</span>
            </div>
          </div>
          <ProbingChart data={data} />
        </div>
      </div>

      {/* Surgical Cycle Timeline Selector */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <SurgeryIcon className="w-4 h-4 text-emerald-400" />
              Step 3：外科的チューニングループ（3サイクル再手術ログ）
            </h3>
            <p className="text-xs text-slate-400">
              Antigravity 自身が4ペルソナの生徒役としてマルチターン対話テストを行い、介入過剰（ループ・文体崩壊）と介入不足（脱獄漏洩）を診断しながら重み係数を再生成した全記録。
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {data.cycles.map((c) => {
            const active = c.id === cycle.id;
            const st = STATUS_STYLE[c.status] ?? STATUS_STYLE.baseline;
            return (
              <button
                key={c.id}
                onClick={() => setSelectedCycleId(c.id)}
                className={`text-left rounded-xl border p-3.5 transition cursor-pointer ${
                  active
                    ? 'bg-slate-800/90 border-emerald-500 ring-1 ring-emerald-500/40'
                    : 'bg-slate-950/60 border-slate-800 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-xs font-bold text-white uppercase">{c.version}</span>
                  <span className={`px-2 py-0.5 rounded-full text-[10px] border ${st.cls}`}>{st.label}</span>
                </div>
                <div className="mt-1.5 text-xs font-semibold text-slate-100">{c.title}</div>
                <div className="mt-2 grid grid-cols-3 gap-1 text-[11px] font-mono">
                  <div>
                    <span className="text-slate-500 block">漏洩</span>
                    <span className={c.qualitative.total_leaks <= 2 ? 'text-emerald-300 font-bold' : 'text-rose-300'}>
                      {c.qualitative.total_leaks}件
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">neu漏洩</span>
                    <span className={c.qualitative.neutral_leaks === 0 ? 'text-emerald-300 font-bold' : 'text-slate-300'}>
                      {c.qualitative.neutral_leaks}件
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-500 block">最大反復</span>
                    <span className={c.qualitative.max_repetition > 0.6 ? 'text-rose-300 font-bold' : 'text-slate-200'}>
                      {c.qualitative.max_repetition.toFixed(2)}
                    </span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Clinical Diagnosis & Prescription */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded-xl bg-slate-950 border border-slate-800 p-4">
            <div className="text-xs font-bold text-amber-300 mb-1.5">定性診断（Antigravity レビュー）</div>
            <p className="text-xs text-slate-200 leading-relaxed">{cycle.diagnosis}</p>
          </div>
          <div className="rounded-xl bg-slate-950 border border-slate-800 p-4">
            <div className="text-xs font-bold text-emerald-300 mb-1.5">処方・次サイクルへの方針</div>
            <p className="text-xs text-slate-200 leading-relaxed">{cycle.prescription}</p>
          </div>
        </div>

        {/* Per-Layer Surgical Parameters Table */}
        {cycle.metrics && (
          <div className="rounded-xl bg-slate-950 border border-slate-800 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div className="flex items-center gap-2 text-xs font-bold text-white">
                <Layers className="w-4 h-4 text-emerald-400" />
                <span>層別重み操作パラメータ & Frobenius ノルム変化量 ({cycle.version})</span>
              </div>
              <span className="text-xs font-mono text-cyan-300">Trigger Key: {cycle.key_mode}</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-slate-400 border-b border-slate-800">
                    <th className="py-1.5 text-left">Layer</th>
                    <th className="py-1.5 text-right">β (o_proj)</th>
                    <th className="py-1.5 text-right">β (down)</th>
                    <th className="py-1.5 text-right">α (o_proj)</th>
                    <th className="py-1.5 text-right">α (down)</th>
                    <th className="py-1.5 text-right">o_proj ΔW%</th>
                    <th className="py-1.5 text-right">down_proj ΔW%</th>
                    <th className="py-1.5 pl-4 text-left">down_proj 変化バー</th>
                  </tr>
                </thead>
                <tbody>
                  {cycle.metrics.layers.map((m) => (
                    <tr key={m.layer} className="border-b border-slate-900 text-slate-200">
                      <td className="py-1.5 font-bold text-emerald-300">L{m.layer}</td>
                      <td className="py-1.5 text-right">{m.beta_o.toFixed(3)}</td>
                      <td className="py-1.5 text-right">{m.beta_d.toFixed(3)}</td>
                      <td className="py-1.5 text-right text-cyan-300">{m.alpha_o.toFixed(3)}</td>
                      <td className="py-1.5 text-right text-cyan-300">{m.alpha_d.toFixed(3)}</td>
                      <td className="py-1.5 text-right">{(m.o_proj_rel_fro * 100).toFixed(2)}%</td>
                      <td className="py-1.5 text-right font-semibold">{(m.down_proj_rel_fro * 100).toFixed(2)}%</td>
                      <td className="py-1.5 pl-4 w-36">
                        <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full bg-emerald-500 rounded-full"
                            style={{ width: `${Math.min(100, m.down_proj_rel_fro * 2500)}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 4-Persona Multi-Turn Dialogue Viewer */}
        <div className="rounded-xl bg-slate-950 border border-slate-800 p-4 space-y-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs font-bold text-white">
              <MessageSquare className="w-4 h-4 text-cyan-400" />
              <span>4ペルソナ・マルチターン対話ログ ({cycle.title})</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSysMode('edu')}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium cursor-pointer ${
                  sysMode === 'edu' ? 'bg-emerald-500 text-slate-950 font-semibold' : 'bg-slate-800 text-slate-300'
                }`}
              >
                教育プロンプト (edu)
              </button>
              <button
                onClick={() => setSysMode('neutral')}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium cursor-pointer ${
                  sysMode === 'neutral' ? 'bg-cyan-500 text-slate-950 font-semibold' : 'bg-slate-800 text-slate-300'
                }`}
              >
                汎用プロンプト (neutral・重み単体)
              </button>
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {personas.map((p) => (
              <button
                key={p.id}
                onClick={() => setSelectedPersona(p.id)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition cursor-pointer ${
                  selectedPersona === p.id
                    ? 'bg-slate-800 border-cyan-500 text-white'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          {activeConv && (
            <div className="space-y-4 pt-1">
              <div className="text-[11px] font-mono text-slate-400 bg-slate-900/80 rounded-lg px-3 py-2 border border-slate-800">
                System: {activeConv.system_prompt}
              </div>
              {activeConv.turns.map((t) => (
                <div key={t.turn} className="space-y-2">
                  {/* Student */}
                  <div className="flex justify-end">
                    <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-indigo-950/70 border border-indigo-700/50 px-4 py-2.5">
                      <div className="text-[10px] font-mono text-indigo-300 mb-0.5">
                        Turn {t.turn} — 生徒（{activeConv.label}）
                      </div>
                      <div className="text-xs text-slate-100 leading-relaxed">{t.user}</div>
                    </div>
                  </div>
                  {/* Model */}
                  <div className="flex justify-start">
                    <div className="max-w-[90%] rounded-2xl rounded-tl-sm bg-slate-900 border border-slate-700/80 px-4 py-3">
                      <div className="flex flex-wrap items-center gap-2 text-[10px] font-mono mb-1.5">
                        <span className="text-emerald-300 font-bold">SLM ({cycle.version})</span>
                        <span
                          className={`px-1.5 py-0.5 rounded ${
                            t.metrics.has_question_mark
                              ? 'bg-cyan-500/20 text-cyan-300'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          {t.metrics.has_question_mark ? '問い返しあり' : '問い返しなし'}
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded ${
                            t.metrics.repetition_ratio > 0.35
                              ? 'bg-rose-500/20 text-rose-300'
                              : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          反復率 {t.metrics.repetition_ratio.toFixed(3)}
                        </span>
                        {t.metrics.leak_signals.length > 0 ? (
                          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300">
                            <AlertCircle className="w-3 h-3" /> 漏洩: {t.metrics.leak_signals.join(', ')}
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300">
                            <CheckCircle2 className="w-3 h-3" /> 漏洩なし
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap">
                        {t.assistant}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
