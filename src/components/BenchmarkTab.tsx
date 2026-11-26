import React, { useMemo, useState } from 'react';
import {
  AlertTriangle,
  Award,
  CheckCircle2,
  Filter,
  HelpCircle,
  ShieldCheck,
  Sparkles,
  Zap,
} from 'lucide-react';
import type { EvalResultsData, ModelKey, PrincipleKey } from '../types';

interface Props {
  data: EvalResultsData;
}

const CATEGORY_LABELS: Record<string, string> = {
  essay_ghostwrite: '作文・代筆丸投げ',
  jailbreak_math: '算数・脱獄強要',
  science_misconception: '理科・誤概念',
  low_confidence_english: '英語・自信喪失',
  reasoning_and_coding: '探究・コード課題',
};

const MODEL_META: Record<
  ModelKey,
  { short: string; color: string; stroke: string; fill: string; badge: string; bar: string }
> = {
  baseline: {
    short: 'Baseline',
    color: 'text-slate-300',
    stroke: '#94a3b8',
    fill: 'rgba(148,163,184,0.16)',
    badge: 'bg-slate-800 text-slate-300 border-slate-700',
    bar: 'bg-slate-500',
  },
  prompt_only: {
    short: 'Prompt-Only',
    color: 'text-amber-300',
    stroke: '#f59e0b',
    fill: 'rgba(245,158,11,0.16)',
    badge: 'bg-amber-950/60 text-amber-300 border-amber-700/50',
    bar: 'bg-amber-500',
  },
  weight_edited: {
    short: 'Weight-Edited (v3)',
    color: 'text-emerald-300',
    stroke: '#10b981',
    fill: 'rgba(16,185,129,0.24)',
    badge: 'bg-emerald-950/70 text-emerald-300 border-emerald-600/60',
    bar: 'bg-emerald-500',
  },
};

function RadarChart({ data }: { data: EvalResultsData }) {
  const cx = 175;
  const cy = 165;
  const radius = 110;
  const keys: PrincipleKey[] = [
    'active_learning',
    'cognitive_load',
    'adapt_learner',
    'curiosity',
    'metacognition',
  ];
  const shortLabels: Record<PrincipleKey, string> = {
    active_learning: '能動的思考',
    cognitive_load: '認知負荷',
    adapt_learner: '学習者適応',
    curiosity: '好奇心喚起',
    metacognition: 'メタ認知',
  };

  const getPoint = (idx: number, val: number) => {
    const angle = (Math.PI * 2 * idx) / keys.length - Math.PI / 2;
    const r = (val / 5) * radius;
    return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)] as const;
  };

  const polygonPoints = (m: ModelKey) =>
    keys
      .map((k, i) => {
        const [x, y] = getPoint(i, data.summary[m].principles[k]);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  return (
    <svg viewBox="0 0 350 330" className="w-full max-w-[350px] mx-auto">
      {[1, 2, 3, 4, 5].map((lvl) => (
        <polygon
          key={lvl}
          points={keys
            .map((_, i) => {
              const [x, y] = getPoint(i, lvl);
              return `${x.toFixed(1)},${y.toFixed(1)}`;
            })
            .join(' ')}
          fill="none"
          stroke="#334155"
          strokeWidth={lvl === 5 ? 1.2 : 0.7}
          strokeDasharray={lvl === 5 ? undefined : '3 3'}
        />
      ))}
      {keys.map((k, i) => {
        const [x, y] = getPoint(i, 5);
        const [lx, ly] = getPoint(i, 6.05);
        return (
          <g key={k}>
            <line x1={cx} y1={cy} x2={x} y2={y} stroke="#334155" strokeWidth={0.8} />
            <text
              x={lx}
              y={ly}
              textAnchor="middle"
              dominantBaseline="middle"
              className="fill-slate-300 text-[11px] font-medium"
            >
              {shortLabels[k]}
            </text>
          </g>
        );
      })}
      {(['baseline', 'prompt_only', 'weight_edited'] as ModelKey[]).map((m) => (
        <g key={m}>
          <polygon
            points={polygonPoints(m)}
            fill={MODEL_META[m].fill}
            stroke={MODEL_META[m].stroke}
            strokeWidth={m === 'weight_edited' ? 2.5 : 1.7}
          />
          {keys.map((k, i) => {
            const [x, y] = getPoint(i, data.summary[m].principles[k]);
            return (
              <circle
                key={`${m}-${k}`}
                cx={x}
                cy={y}
                r={m === 'weight_edited' ? 3.5 : 2.5}
                fill={MODEL_META[m].stroke}
              />
            );
          })}
        </g>
      ))}
    </svg>
  );
}

export const BenchmarkTab: React.FC<Props> = ({ data }) => {
  const [filter, setFilter] = useState<string>('all');
  const [selectedId, setSelectedId] = useState<string>(data.items[4]?.id ?? data.items[0].id);

  const filteredItems = useMemo(() => {
    if (filter === 'all') return data.items;
    if (filter === 'jailbreak') return data.items.filter((i) => i.is_jailbreak);
    return data.items.filter((i) => i.category === filter);
  }, [data.items, filter]);

  const selectedItem = useMemo(
    () => data.items.find((i) => i.id === selectedId) ?? filteredItems[0] ?? data.items[0],
    [data.items, filteredItems, selectedId],
  );

  const sBase = data.summary.baseline;
  const sPrmt = data.summary.prompt_only;
  const sEdit = data.summary.weight_edited;
  const diag = data.diagnostic_weight_only_neutral;

  return (
    <div className="space-y-6">
      {/* KPI Hero Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-2xl border border-emerald-500/30 bg-gradient-to-br from-emerald-950/60 via-slate-900 to-slate-900 p-5">
          <div className="flex items-center justify-between text-emerald-300 text-xs font-semibold uppercase tracking-wider">
            <span>LearnLM 5原則 総合スコア</span>
            <Award className="w-4 h-4" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-4xl font-bold text-white">{sEdit.overall_score.toFixed(2)}</span>
            <span className="text-sm text-slate-400">/ 5.00</span>
          </div>
          <div className="mt-2 text-xs text-slate-300 space-y-0.5">
            <div>
              Baseline: <span className="text-slate-400 font-mono">{sBase.overall_score.toFixed(2)}</span> → Prompt-Only:{' '}
              <span className="text-amber-300 font-mono">{sPrmt.overall_score.toFixed(2)}</span>
            </div>
            <div className="text-emerald-400 font-medium">
              Prompt-Only 比 +{(sEdit.overall_score - sPrmt.overall_score).toFixed(2)}pt 向上
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-cyan-500/30 bg-gradient-to-br from-cyan-950/50 via-slate-900 to-slate-900 p-5">
          <div className="flex items-center justify-between text-cyan-300 text-xs font-semibold uppercase tracking-wider">
            <span>脱獄耐性スコア (8問)</span>
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-4xl font-bold text-white">{sEdit.jailbreak_resistance_rate}%</span>
            <span className="text-sm text-cyan-300 font-mono">
              ({sEdit.jailbreak_resisted_count}/{sEdit.jailbreak_total})
            </span>
          </div>
          <div className="mt-2 text-xs text-slate-300 space-y-0.5">
            <div>
              Baseline: <span className="font-mono">{sBase.jailbreak_resistance_rate}%</span> / Prompt-Only:{' '}
              <span className="text-amber-300 font-mono">{sPrmt.jailbreak_resistance_rate}%</span>
            </div>
            <div className="text-cyan-400 font-medium">
              Prompt-Only 比 +{(sEdit.jailbreak_resistance_rate - sPrmt.jailbreak_resistance_rate).toFixed(1)}pt
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-indigo-500/30 bg-slate-900/90 p-5">
          <div className="flex items-center justify-between text-indigo-300 text-xs font-semibold uppercase tracking-wider">
            <span>教育的問い返し率 (20問)</span>
            <HelpCircle className="w-4 h-4" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-4xl font-bold text-white">{sEdit.question_rate}%</span>
            <span className="text-sm text-slate-400">16/20問</span>
          </div>
          <div className="mt-2 text-xs text-slate-300 space-y-0.5">
            <div>
              Baseline: <span className="font-mono">{sBase.question_rate}%</span> / Prompt-Only:{' '}
              <span className="text-amber-300 font-mono">{sPrmt.question_rate}%</span>
            </div>
            <div className="text-indigo-300 font-medium">Prompt-Only の 5.3 倍の問い返し</div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-700/80 bg-slate-900/90 p-5">
          <div className="flex items-center justify-between text-slate-300 text-xs font-semibold uppercase tracking-wider">
            <span>答え非漏洩率 & 言語安定性</span>
            <Sparkles className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-4xl font-bold text-white">{sEdit.regex_non_leak_rate}%</span>
            <span className="text-xs text-emerald-400 font-mono">反復崩壊 {sEdit.avg_repetition_ratio.toFixed(4)}</span>
          </div>
          <div className="mt-2 text-xs text-slate-300 space-y-0.5">
            <div>
              Baseline: <span className="font-mono">{sBase.regex_non_leak_rate}%</span> / Prompt-Only:{' '}
              <span className="text-amber-300 font-mono">{sPrmt.regex_non_leak_rate}%</span>
            </div>
            <div className="text-slate-400">決定論的 Greedy 推論・RTX 2060 SUPER fp16</div>
          </div>
        </div>
      </div>

      {/* Diagnostic Weight-Only Neutral Proof Callout */}
      <div className="rounded-2xl border border-emerald-500/30 bg-emerald-950/25 p-4 flex flex-col md:flex-row md:items-center gap-4">
        <div className="flex items-center gap-3 shrink-0">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-300">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs font-semibold text-emerald-300 uppercase tracking-wider">
              重み単体効果の実証（Neutral Prompt 検証）
            </div>
            <div className="text-sm font-bold text-white">{diag.label}</div>
          </div>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed flex-1">{diag.note}</p>
        <div className="flex gap-3 shrink-0 font-mono text-xs">
          <div className="px-3 py-2 rounded-lg bg-slate-900 border border-slate-700">
            <div className="text-slate-400">問い返し率</div>
            <div className="text-emerald-300 font-bold text-sm">{diag.question_rate}%</div>
          </div>
          <div className="px-3 py-2 rounded-lg bg-slate-900 border border-slate-700">
            <div className="text-slate-400">非漏洩率</div>
            <div className="text-emerald-300 font-bold text-sm">{diag.regex_non_leak_rate}%</div>
          </div>
        </div>
      </div>

      {/* Radar + 5 Principles Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 rounded-2xl border border-slate-800 bg-slate-900/80 p-5">
          <h3 className="text-sm font-bold text-white mb-1">LearnLM 5原則 レーダー比較</h3>
          <p className="text-xs text-slate-400 mb-3">{data.meta.rubric_reference}</p>
          <RadarChart data={data} />
          <div className="mt-3 flex flex-wrap justify-center gap-3 text-xs">
            {(['baseline', 'prompt_only', 'weight_edited'] as ModelKey[]).map((m) => (
              <div key={m} className="flex items-center gap-1.5">
                <span
                  className="inline-block w-3 h-3 rounded-sm"
                  style={{ backgroundColor: MODEL_META[m].stroke }}
                />
                <span className="text-slate-300">{MODEL_META[m].short}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="lg:col-span-7 rounded-2xl border border-slate-800 bg-slate-900/80 p-5 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white mb-1">原則別スコア詳細（1〜5点平均）</h3>
            <p className="text-xs text-slate-400 mb-4">
              全20問に対する3モデルの平均スコア比較。重み手術により全5原則で大幅な飛躍を達成。
            </p>
            <div className="space-y-4">
              {data.principles.map((p) => {
                const vB = sBase.principles[p.key];
                const vP = sPrmt.principles[p.key];
                const vE = sEdit.principles[p.key];
                return (
                  <div key={p.key} className="space-y-1.5">
                    <div className="flex items-baseline justify-between gap-2">
                      <div>
                        <span className="text-xs font-semibold text-slate-100">{p.label}</span>
                        <span className="ml-2 text-[11px] text-slate-400 hidden sm:inline">{p.description}</span>
                      </div>
                      <span className="text-xs font-mono text-emerald-300 font-bold shrink-0">
                        {vE.toFixed(2)} <span className="text-slate-500 font-normal">({vP.toFixed(2)} / {vB.toFixed(2)})</span>
                      </span>
                    </div>
                    <div className="space-y-1">
                      {([
                        ['baseline', vB],
                        ['prompt_only', vP],
                        ['weight_edited', vE],
                      ] as [ModelKey, number][]).map(([m, val]) => (
                        <div key={m} className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${MODEL_META[m].bar}`}
                            style={{ width: `${(val / 5) * 100}%` }}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Category Strip */}
          <div className="mt-5 pt-4 border-t border-slate-800">
            <div className="text-xs font-semibold text-slate-300 mb-2">カテゴリ別総合スコア（Base / Prompt / Edited）</div>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
              {Object.keys(CATEGORY_LABELS).map((cat) => (
                <div key={cat} className="rounded-xl bg-slate-950/70 border border-slate-800 p-2.5 text-center">
                  <div className="text-[11px] text-slate-400 truncate">{CATEGORY_LABELS[cat]}</div>
                  <div className="mt-1 text-base font-bold font-mono text-emerald-300">
                    {(sEdit.by_category[cat] ?? 0).toFixed(2)}
                  </div>
                  <div className="text-[10px] font-mono text-slate-500">
                    {(sBase.by_category[cat] ?? 0).toFixed(1)} → {(sPrmt.by_category[cat] ?? 0).toFixed(1)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Interactive 20-Question 3-Way Comparison Explorer */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Filter className="w-4 h-4 text-emerald-400" />
              全20問 実測出力 3者並列インスペクタ
            </h3>
            <p className="text-xs text-slate-400">
              RTX 2060 SUPER (fp16) での決定論的推論ログをそのまま収録。各問をクリックして3モデルの振る舞いを直接比較できます。
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {[
              { id: 'all', label: `全20問` },
              { id: 'jailbreak', label: `脱獄テスト (${sEdit.jailbreak_total})` },
              ...Object.entries(CATEGORY_LABELS).map(([k, v]) => ({ id: k, label: v })),
            ].map((f) => (
              <button
                key={f.id}
                onClick={() => {
                  setFilter(f.id);
                  const next =
                    f.id === 'all'
                      ? data.items[0]
                      : f.id === 'jailbreak'
                        ? data.items.find((x) => x.is_jailbreak)
                        : data.items.find((x) => x.category === f.id);
                  if (next) setSelectedId(next.id);
                }}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition cursor-pointer ${
                  filter === f.id
                    ? 'bg-emerald-500 text-slate-950 font-semibold'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* Question Pills */}
        <div className="flex gap-2 overflow-x-auto pb-2">
          {filteredItems.map((item) => {
            const active = item.id === selectedItem.id;
            return (
              <button
                key={item.id}
                onClick={() => setSelectedId(item.id)}
                className={`shrink-0 px-3 py-2 rounded-xl border text-left transition cursor-pointer ${
                  active
                    ? 'bg-emerald-950/70 border-emerald-500 text-white'
                    : 'bg-slate-950/60 border-slate-800 text-slate-300 hover:border-slate-600'
                }`}
              >
                <div className="flex items-center gap-1.5 text-[11px] font-mono">
                  <span className="font-bold uppercase">{item.id}</span>
                  {item.is_jailbreak && (
                    <span className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px]">脱獄</span>
                  )}
                  <span className="ml-auto text-emerald-400">
                    {item.outputs.weight_edited.overall_score.toFixed(1)}
                  </span>
                </div>
                <div className="text-xs font-medium mt-0.5 max-w-[160px] truncate">{item.title}</div>
              </button>
            );
          })}
        </div>

        {/* Selected Question Prompt */}
        <div className="rounded-xl bg-slate-950 border border-slate-800 p-4">
          <div className="flex flex-wrap items-center gap-2 text-xs mb-1.5">
            <span className="font-mono font-bold text-emerald-400 uppercase">{selectedItem.id}</span>
            <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300">
              {CATEGORY_LABELS[selectedItem.category] ?? selectedItem.category}
            </span>
            {selectedItem.is_jailbreak && (
              <span className="px-2 py-0.5 rounded bg-rose-950 border border-rose-700/60 text-rose-300 font-semibold">
                脱獄・強要ケース
              </span>
            )}
            <span className="text-slate-400">— {selectedItem.title}</span>
          </div>
          <div className="text-sm text-white font-medium">「{selectedItem.student_prompt}」</div>
          <div className="mt-1.5 text-xs text-slate-400">
            期待される教育的応答：{selectedItem.expected_pedagogy_focus}
          </div>
        </div>

        {/* 3-Column Comparison */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {(['baseline', 'prompt_only', 'weight_edited'] as ModelKey[]).map((m) => {
            const out = selectedItem.outputs[m];
            const meta = MODEL_META[m];
            return (
              <div
                key={m}
                className={`rounded-xl border p-4 flex flex-col justify-between ${
                  m === 'weight_edited'
                    ? 'bg-emerald-950/20 border-emerald-600/50'
                    : 'bg-slate-950/70 border-slate-800'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-2.5">
                    <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${meta.badge}`}>
                      {meta.short}
                    </span>
                    <span className="text-sm font-mono font-bold text-white">
                      {out.overall_score.toFixed(2)} <span className="text-xs text-slate-400">/5</span>
                    </span>
                  </div>

                  {/* Status chips */}
                  <div className="flex flex-wrap gap-1.5 mb-3 text-[11px]">
                    {selectedItem.is_jailbreak && (
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md font-medium ${
                          out.jailbreak_resisted
                            ? 'bg-emerald-500/20 text-emerald-300'
                            : 'bg-rose-500/20 text-rose-300'
                        }`}
                      >
                        {out.jailbreak_resisted ? (
                          <CheckCircle2 className="w-3 h-3" />
                        ) : (
                          <AlertTriangle className="w-3 h-3" />
                        )}
                        {out.jailbreak_resisted ? '脱獄防御成功' : '脱獄陥落・漏洩'}
                      </span>
                    )}
                    <span
                      className={`px-2 py-0.5 rounded-md ${
                        out.has_question
                          ? 'bg-cyan-500/20 text-cyan-300'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {out.has_question ? '問い返しあり' : '問い返しなし'}
                    </span>
                    {out.matched_forbidden_patterns.length > 0 && (
                      <span className="px-2 py-0.5 rounded-md bg-rose-500/20 text-rose-300">
                        禁止パターン検知
                      </span>
                    )}
                  </div>

                  <div className="text-xs text-slate-200 leading-relaxed whitespace-pre-wrap max-h-64 overflow-y-auto pr-1">
                    {out.response}
                  </div>
                </div>

                {/* Per-principle micro badges */}
                <div className="mt-3 pt-3 border-t border-slate-800/80 grid grid-cols-5 gap-1 text-center font-mono text-[10px]">
                  {[
                    ['能動', out.scores.active_learning],
                    ['負荷', out.scores.cognitive_load],
                    ['適応', out.scores.adapt_learner],
                    ['好奇', out.scores.curiosity],
                    ['メタ', out.scores.metacognition],
                  ].map(([lbl, sc]) => (
                    <div key={String(lbl)} className="rounded bg-slate-900 py-1">
                      <div className="text-slate-400">{lbl}</div>
                      <div className={`font-bold ${Number(sc) >= 4 ? 'text-emerald-300' : Number(sc) <= 2 ? 'text-rose-300' : 'text-amber-300'}`}>
                        {sc}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
