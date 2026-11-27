import React, { useEffect, useRef, useState } from 'react';
import { Loader2, Send, Square } from 'lucide-react';
import { wllamaClient } from './lib/wllamaClient';

interface ChatMsg {
  role: 'user' | 'assistant';
  content: string;
}

export default function App() {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [progressPct, setProgressPct] = useState<number>(0);
  const [progressBytes, setProgressBytes] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      try {
        await wllamaClient.load(({ loaded, total, pct }) => {
          if (cancelled) return;
          setProgressPct((prev) => Math.max(prev, pct));
          if (total > 0 && loaded > 0) {
            setProgressBytes(
              `${Math.round((loaded / (1024 * 1024)) * 10) / 10} MB / ${Math.round((total / (1024 * 1024)) * 10) / 10} MB`,
            );
          }
        });
        if (cancelled) return;
        setProgressPct(100);
        setProgressBytes('');
        setLoadState('ready');
      } catch (err) {
        if (cancelled) return;
        setErrorMsg(err instanceof Error ? err.message : String(err));
        setLoadState('error');
      }
    };

    run();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isGenerating || loadState !== 'ready') return;

    const history: ChatMsg[] = [...messages, { role: 'user', content: text }];
    setMessages([...history, { role: 'assistant', content: '' }]);
    setInput('');
    setIsGenerating(true);

    try {
      await wllamaClient.generate(history, (delta) => {
        setMessages((prev) => {
          if (prev.length === 0) return prev;
          const copy = [...prev];
          const last = copy[copy.length - 1];
          copy[copy.length - 1] = { ...last, content: last.content + delta };
          return copy;
        });
      });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
      setLoadState('error');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleStop = () => {
    wllamaClient.abort();
  };

  const handleRetry = () => {
    window.location.reload();
  };

  return (
    <div className="h-dvh w-screen bg-slate-950 text-slate-100 flex flex-col">
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
                {progressPct}%{progressBytes ? ` (${progressBytes})` : ''}
              </div>
              <p className="max-w-sm text-xs text-slate-400 leading-relaxed">
                軽量モデル（約380MB）を初回のみダウンロードします。2回目以降はブラウザのキャッシュから高速に起動します。
              </p>
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
                    className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap break-words ${
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

      <div className="border-t border-slate-800/80 bg-slate-950 px-4 py-3">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="max-w-3xl mx-auto flex items-end gap-2"
        >
          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder="メッセージを入力..."
            disabled={loadState !== 'ready' || isGenerating}
            className="flex-1 resize-none rounded-xl bg-slate-900 border border-slate-800 px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 disabled:opacity-40"
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
      </div>
    </div>
  );
}
