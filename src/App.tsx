import React, { useEffect, useState } from 'react';
import { BarChart3, Brain, Cpu, ExternalLink, FlaskConical, Loader2 } from 'lucide-react';
import { BenchmarkTab } from './components/BenchmarkTab';
import { SurgeryTab } from './components/SurgeryTab';
import { LiveLabTab } from './components/LiveLabTab';
import type { EvalResultsData, SurgeryReportData } from './types';

type TabId = 'benchmark' | 'surgery' | 'livelab';

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('benchmark');
  const [evalData, setEvalData] = useState<EvalResultsData | null>(null);
  const [surgeryData, setSurgeryData] = useState<SurgeryReportData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const base = import.meta.env.BASE_URL;
    Promise.all([
      fetch(`${base}data/eval-results.json`).then((r) => {
        if (!r.ok) throw new Error(`eval-results.json HTTP ${r.status}`);
        return r.json() as Promise<EvalResultsData>;
      }),
      fetch(`${base}data/surgery-report.json`).then((r) => {
        if (!r.ok) throw new Error(`surgery-report.json HTTP ${r.status}`);
        return r.json() as Promise<SurgeryReportData>;
      }),
    ])
      .then(([ev, sr]) => {
        setEvalData(ev);
        setSurgeryData(sr);
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
      });
  }, []);

  const tabs: { id: TabId; label: string; sub: string; icon: React.ReactNode }[] = [
    {
      id: 'benchmark',
      label: 'Tab 1: 定量ベンチマーク',
      sub: '3者比較・LearnLM 5原則・脱獄耐性',
      icon: <BarChart3 className="w-4 h-4" />,
    },
    {
      id: 'surgery',
      label: 'Tab 2: 重み手術ログ＆定性検証',
      sub: '28層Probing・3サイクル再手術・4ペルソナ',
      icon: <Brain className="w-4 h-4" />,
    },
    {
      id: 'livelab',
      label: 'Tab 3: ブラウザ内 SLM 実験室',
      sub: 'Transformers.js / WebGPU 4-bit ライブ推論',
      icon: <FlaskConical className="w-4 h-4" />,
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Top Header */}
      <header className="border-b border-slate-800/90 bg-slate-950/90 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div>
              <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono mb-1">
                <span className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-semibold">
                  Agentic Weight Surgery
                </span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                  Qwen2.5-1.5B-Instruct
                </span>
                <span className="px-2 py-0.5 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                  Zero Gradient Training
                </span>
                <span className="px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 inline-flex items-center gap-1">
                  <Cpu className="w-3 h-3" /> WebGPU 4-bit ONNX
                </span>
              </div>
              <h1 className="text-lg sm:text-xl font-extrabold tracking-tight text-white">
                LLMエージェントによる直接重み手術 — 教育特化 SLM 構築＆評価ラボ
              </h1>
            </div>
            <a
              href="https://github.com/yamakoud-org/edu-slm-eval-demo"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs text-slate-200 self-start md:self-auto"
            >
              GitHub Repository <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>

          {/* 3-Tab Switcher */}
          <nav className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-2">
            {tabs.map((t) => {
              const active = activeTab === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className={`flex items-center gap-3 px-4 py-2.5 rounded-xl border text-left transition cursor-pointer ${
                    active
                      ? 'bg-emerald-500/15 border-emerald-500 text-white'
                      : 'bg-slate-900/70 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                  }`}
                >
                  <div className={active ? 'text-emerald-400' : 'text-slate-500'}>{t.icon}</div>
                  <div className="min-w-0">
                    <div className="text-xs font-bold truncate">{t.label}</div>
                    <div className="text-[11px] opacity-75 truncate">{t.sub}</div>
                  </div>
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        {error ? (
          <div className="rounded-2xl border border-rose-700/60 bg-rose-950/50 p-6 text-sm text-rose-200">
            データの読み込みに失敗しました: {error}
          </div>
        ) : !evalData || !surgeryData ? (
          <div className="py-24 flex flex-col items-center justify-center gap-3 text-slate-400">
            <Loader2 className="w-7 h-7 animate-spin text-emerald-400" />
            <div className="text-xs font-mono">ベンチマーク・重み手術データを読み込み中...</div>
          </div>
        ) : (
          <>
            {activeTab === 'benchmark' && <BenchmarkTab data={evalData} />}
            {activeTab === 'surgery' && <SurgeryTab data={surgeryData} />}
            {activeTab === 'livelab' && <LiveLabTab />}
          </>
        )}
      </main>

      <footer className="border-t border-slate-900 mt-12 py-6 text-center text-xs text-slate-500">
        Built autonomously by Antigravity via Direct Weight Surgery (Abliteration + Closed-Loop Rank-1 Weight Baking) on Qwen2.5-1.5B-Instruct.
      </footer>
    </div>
  );
}
