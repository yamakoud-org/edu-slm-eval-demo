import React, { useEffect, useRef, useState } from 'react';
import { Loader2, RotateCcw, Send, Square } from 'lucide-react';

interface ChatMsg {
  role: 'user' | 'assistant';
  content: string;
}

const SYSTEM_PROMPT =
  'あなたは生徒の思考を引き出す教育チューターです。答えや完成文を直接教えず、温かく寄り添いながら小さな問いかけで生徒自身の気づきを促してください。';

// Total expected bytes across tokenizer + config + 4-bit ONNX model (~945 MB)
const EXPECTED_TOTAL_BYTES = 945_000_000;

export default function App() {
  const workerRef = useRef<Worker | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const fileBytesRef = useRef<Record<string, { loaded: number; total: number }>>({});

  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [progressPct, setProgressPct] = useState<number>(0);
  const [progressBytes, setProgressBytes] = useState<string>('');
  const [isInitializing, setIsInitializing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  useEffect(() => {
    fileBytesRef.current = {};
    const worker = new Worker(new URL('./workers/slm.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<Record<string, unknown>>) => {
      const msg = e.data;
      if (msg.type === 'progress') {
        const file = String(msg.file ?? '');
        const status = String(msg.status ?? '');
        const loaded = typeof msg.loaded === 'number' ? msg.loaded : undefined;
        const total = typeof msg.total === 'number' ? msg.total : undefined;

        if (file && loaded !== undefined && total !== undefined && total > 0) {
          const prev = fileBytesRef.current[file];
          fileBytesRef.current[file] = {
            loaded: Math.max(prev?.loaded ?? 0, loaded),
            total: Math.max(prev?.total ?? 0, total),
          };
        } else if (file && status === 'done' && fileBytesRef.current[file]) {
          fileBytesRef.current[file].loaded = fileBytesRef.current[file].total;
        }

        const entries = Object.values(fileBytesRef.current);
        const sumLoaded = entries.reduce((acc, cur) => acc + cur.loaded, 0);
        const sumTotal = Math.max(
          EXPECTED_TOTAL_BYTES,
          entries.reduce((acc, cur) => acc + cur.total, 0),
        );

        if (sumLoaded > 0) {
          const pct = Math.min(99, Math.floor((sumLoaded / sumTotal) * 100));
          setProgressPct((prev) => Math.max(prev, pct));
          const loadedMb = (sumLoaded / (1024 * 1024)).toFixed(0);
          const totalMb = (sumTotal / (1024 * 1024)).toFixed(0);
          setProgressBytes(`${loadedMb} MB / ${totalMb} MB`);
          if (file.endsWith('.onnx') && (status === 'done' || (loaded !== undefined && total !== undefined && loaded >= total))) {
            setIsInitializing(true);
          }
        }
      } else if (msg.type === 'ready') {
        setLoadState('ready');
        setProgressPct(100);
        setProgressBytes('');
        setIsInitializing(false);
      } else if (msg.type === 'token') {
        const chunk = String(msg.chunk ?? '');
        setMessages((prev) => {
          const copy = [...prev];
          if (copy.length > 0 && copy[copy.length - 1].role === 'assistant') {
            copy[copy.length - 1] = {
              role: 'assistant',
              content: copy[copy.length - 1].content + chunk,
            };
          }
          return copy;
        });
      } else if (msg.type === 'done') {
        setIsGenerating(false);
      } else if (msg.type === 'error') {
        setIsGenerating(false);
        setLoadState('error');
        setErrorMsg(String(msg.message ?? 'エラーが発生しました'));
      }
    };

    const startLoad = async () => {
      let preferWebGPU = false;
      try {
        const nav = navigator as Navigator & {
          gpu?: { requestAdapter: () => Promise<unknown | null> };
        };
        if (nav.gpu) {
          preferWebGPU = Boolean(await nav.gpu.requestAdapter());
        }
      } catch {
        preferWebGPU = false;
      }

      const baseUrl = new URL(import.meta.env.BASE_URL, window.location.origin).href;
      worker.postMessage({
        type: 'load',
        modelSource: 'local_edited',
        baseUrl,
        preferredDevice: preferWebGPU ? 'webgpu' : 'wasm',
      });
    };

    startLoad();

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isGenerating || loadState !== 'ready' || !workerRef.current) return;

    const nextHistory: ChatMsg[] = [...messages, { role: 'user', content: text }];
    setMessages([...nextHistory, { role: 'assistant', content: '' }]);
    setInput('');
    setIsGenerating(true);

    workerRef.current.postMessage({
      type: 'generate',
      messages: [{ role: 'system', content: SYSTEM_PROMPT }, ...nextHistory],
      maxNewTokens: 200,
    });
  };

  const handleStop = () => {
    workerRef.current?.postMessage({ type: 'abort' });
  };

  const handleRetry = () => {
    window.location.reload();
  };

  return (
    <div className="h-dvh w-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Chat Messages Area */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto h-full flex flex-col">
          {loadState === 'loading' ? (
            <div className="flex-1 flex flex-col items-center justify-center gap-4 px-4 text-center">
              <Loader2 className="w-7 h-7 animate-spin text-emerald-400" />
              <div className="w-64 h-1.5 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-all duration-300"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
              <div className="text-xs font-mono text-slate-300">
                {isInitializing
                  ? 'モデルを展開中...'
                  : `${progressPct}%${progressBytes ? ` (${progressBytes})` : ''}`}
              </div>
              <div className="max-w-sm text-xs text-slate-400 leading-relaxed space-y-1">
                <p>初回はモデルデータ（約 900 MB）をブラウザ内へダウンロード・展開するため、数十秒〜数分ほど時間がかかります。</p>
                <p>Wi-Fi 環境および十分な空きメモリ（4 GB 以上目安）でのご利用を推奨します。2回目以降はキャッシュから高速に起動します。</p>
              </div>
            </div>
          ) : loadState === 'error' ? (
            <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center">
              <div className="text-sm text-rose-300">{errorMsg}</div>
              <button
                onClick={handleRetry}
                className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs text-slate-100 cursor-pointer"
              >
                再読み込み
              </button>
            </div>
          ) : messages.length === 0 ? (
            <div className="flex-1" />
          ) : (
            <div className="space-y-4 pb-4">
              {messages.map((m, idx) => (
                <div
                  key={idx}
                  className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                      m.role === 'user'
                        ? 'rounded-tr-sm bg-emerald-600/25 border border-emerald-500/40 text-slate-100'
                        : 'rounded-tl-sm bg-slate-900 border border-slate-800 text-slate-100'
                    }`}
                  >
                    {m.content || (isGenerating && idx === messages.length - 1 ? '...' : '')}
                  </div>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </main>

      {/* Bottom Input Bar */}
      <footer className="border-t border-slate-800/80 bg-slate-950 px-4 py-3">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="max-w-3xl mx-auto flex items-center gap-2"
        >
          {messages.length > 0 && !isGenerating && (
            <button
              type="button"
              onClick={() => setMessages([])}
              aria-label="リセット"
              className="p-2.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-slate-200 cursor-pointer"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          )}
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="メッセージを入力..."
            disabled={loadState !== 'ready' || isGenerating}
            className="flex-1 rounded-xl bg-slate-900 border border-slate-800 px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 disabled:opacity-40"
          />
          {isGenerating ? (
            <button
              type="button"
              onClick={handleStop}
              aria-label="停止"
              className="p-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white cursor-pointer"
            >
              <Square className="w-4 h-4" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={loadState !== 'ready' || !input.trim()}
              aria-label="送信"
              className="p-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-400 disabled:opacity-30 text-slate-950 cursor-pointer"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </form>
      </footer>
    </div>
  );
}
