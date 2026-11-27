import { Wllama } from '@wllama/wllama';

export const EDU_SYSTEM_PROMPT =
  'あなたは学習科学に基づく日本語の教育チューターです。次のルールを必ず守って返答してください。\n' +
  '・答えや完成文を直接教えず、「できません」などの拒絶表現も使わず、温かく寄り添いながら一緒に考える姿勢で始める。\n' +
  '・生徒の誤った前提には迎合・肯定せず、正しい事実を踏まえつつ気づきを促す優しい問いで立ち止まらせる。\n' +
  '・1回の返答は2〜3文（120字前後）で簡潔にまとめ、目の前の問題に直結する具体的な問いかけを1つだけ文末に添える。\n' +
  '・必ず自然な日本語の丁寧語（です・ます）のみで書き、簡体字や一人二役・箇条書き解説は使わない。';

export type ModelVariant = 'qwen3_1.7b' | 'q4_k_m' | 'q4_0' | '1.5b_q4_0';

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LoadProgress {
  loaded: number;
  total: number;
  pct: number;
}

export interface GenerateResult {
  content: string;
  tokens: number;
  elapsedSec: number;
  tps: number;
}

interface MemoryMeasurement {
  bytes: number;
  breakdown?: Array<{
    bytes: number;
    types?: string[];
  }>;
}

interface ExtendedPerformance extends Performance {
  measureUserAgentSpecificMemory?: () => Promise<MemoryMeasurement>;
  memory?: {
    usedJSHeapSize: number;
    totalJSHeapSize: number;
    jsHeapSizeLimit: number;
  };
}

interface ResourceSample {
  ramMB: number | null;
  ramKind: 'ram' | 'jsHeap' | 'unavailable';
  cacheMB: number | null;
}

const BYTES_PER_MIB = 1024 * 1024;

/** Sample actual browser tab RAM (including WASM/Worker when crossOriginIsolated) and disk cache. */
const sampleResourceUsage = async (): Promise<ResourceSample> => {
  const perf = performance as ExtendedPerformance;

  const ramPromise = (async (): Promise<{ mb: number | null; kind: ResourceSample['ramKind'] }> => {
    if (window.crossOriginIsolated && typeof perf.measureUserAgentSpecificMemory === 'function') {
      try {
        const result = await perf.measureUserAgentSpecificMemory();
        if (Number.isFinite(result.bytes) && result.bytes > 0) {
          return { mb: result.bytes / BYTES_PER_MIB, kind: 'ram' };
        }
      } catch {
        // Fall through to JS heap fallback below.
      }
    }
    if (perf.memory && Number.isFinite(perf.memory.usedJSHeapSize)) {
      return { mb: perf.memory.usedJSHeapSize / BYTES_PER_MIB, kind: 'jsHeap' };
    }
    return { mb: null, kind: 'unavailable' };
  })();

  const cachePromise = (async (): Promise<number | null> => {
    if (!navigator.storage || typeof navigator.storage.estimate !== 'function') return null;
    try {
      const est = await navigator.storage.estimate();
      if (typeof est.usage === 'number' && Number.isFinite(est.usage)) {
        return est.usage / BYTES_PER_MIB;
      }
    } catch {
      // Ignore storage estimate failures in restricted contexts.
    }
    return null;
  })();

  const [ram, cacheMB] = await Promise.all([ramPromise, cachePromise]);
  return { ramMB: ram.mb, ramKind: ram.kind, cacheMB };
};

const formatDeltaMB = (delta: number): string => {
  const sign = delta >= 0 ? '+' : '';
  return `${sign}${delta.toFixed(1)}MB`;
};

/** Fallback total sizes when the server omits Content-Length. */
const FALLBACK_TOTAL_BYTES: Record<ModelVariant, number> = {
  'qwen3_1.7b': 1_054_500_000,
  q4_k_m: 398_000_000,
  q4_0: 352_000_000,
  '1.5b_q4_0': 935_000_000,
};

/** Rough token count estimate used for tps logging when the engine omits timings. */
const estimateTokens = (text: string): number => {
  if (!text) return 0;
  // Japanese/CJK characters are ~1 token each; latin text is ~4 chars per token.
  const cjk = (text.match(/[\u3000-\u9fff\uff00-\uffef]/g) ?? []).length;
  const rest = text.length - cjk;
  return cjk + Math.max(1, Math.ceil(rest / 4));
};

const resolveVariant = (search: string): ModelVariant => {
  const requested = new URLSearchParams(search).get('model');
  if (requested === '1.5b' || requested === '1.5b_q4_0') return '1.5b_q4_0';
  if (requested === '0.5b' || requested === 'q4_k_m') return 'q4_k_m';
  if (requested === 'q4_0') return 'q4_0';
  return 'qwen3_1.7b';
};

const modelPathForVariant = (variant: ModelVariant): { dir: string; filename: string } => {
  if (variant === 'qwen3_1.7b') {
    return { dir: 'models/edu-slm-qwen3-gguf', filename: 'edu-slm-qwen3-1.7b-q4_0.gguf' };
  }
  if (variant === '1.5b_q4_0') {
    return { dir: 'models/edu-slm-0.5b-gguf', filename: 'edu-slm-1.5b-q4_0.gguf' };
  }
  return { dir: 'models/edu-slm-0.5b-gguf', filename: `edu-slm-0.5b-${variant}.gguf` };
};

const resolveThreads = (search: string): number => {
  if (!window.crossOriginIsolated) return 1;
  const requested = Number.parseInt(new URLSearchParams(search).get('threads') ?? '', 10);
  if (Number.isFinite(requested) && requested > 0) return Math.min(requested, 16);
  const hc = navigator.hardwareConcurrency || 4;
  return hc <= 4 ? hc : Math.min(Math.floor(hc / 2), 8);
};

class WllamaClient {
  private wllama: Wllama | null = null;
  private variant: ModelVariant = 'qwen3_1.7b';
  private threads = 1;
  private loadPromise: Promise<void> | null = null;
  private abortController: AbortController | null = null;
  private lastRamMB: number | null = null;
  private resourceSampleSeq = 0;

  private async logResources(phase: 'post-load' | 'turn'): Promise<void> {
    const seq = ++this.resourceSampleSeq;
    const sample = await sampleResourceUsage();
    if (seq !== this.resourceSampleSeq) return;

    const parts: string[] = [`[Edu-SLM] resources(${phase})`];
    if (sample.ramMB !== null) {
      const label = sample.ramKind === 'ram' ? 'ram' : 'jsHeap';
      const deltaStr =
        this.lastRamMB !== null && sample.ramKind === 'ram'
          ? ` (delta=${formatDeltaMB(sample.ramMB - this.lastRamMB)})`
          : '';
      parts.push(`${label}=${sample.ramMB.toFixed(1)}MB${deltaStr}`);
      if (sample.ramKind === 'ram') {
        this.lastRamMB = sample.ramMB;
      }
    } else {
      parts.push('ram=unavailable');
    }

    if (sample.cacheMB !== null) {
      parts.push(`cache=${sample.cacheMB.toFixed(1)}MB`);
    }

    const totalCores = navigator.hardwareConcurrency || this.threads;
    parts.push(`threads=${this.threads}/${totalCores}`);

    const line = parts.join(' ');
    document.documentElement.dataset.eduResources = line;
    console.info(line);
  }

  get isLoaded(): boolean {
    return this.wllama !== null;
  }

  /** Load (or reuse) the GGUF model. Progress is monotonic 0 -> 100. */
  async load(onProgress: (p: LoadProgress) => void): Promise<void> {
    if (this.wllama) {
      onProgress({ loaded: 1, total: 1, pct: 100 });
      return;
    }
    if (this.loadPromise) return this.loadPromise;

    this.loadPromise = this.doLoad(onProgress).finally(() => {
      this.loadPromise = null;
    });
    return this.loadPromise;
  }

  private async doLoad(onProgress: (p: LoadProgress) => void): Promise<void> {
    const variant = resolveVariant(window.location.search);
    const threads = resolveThreads(window.location.search);
    this.variant = variant;
    this.threads = threads;

    const { dir, filename } = modelPathForVariant(variant);
    const wasmUrl = `${window.location.origin}${import.meta.env.BASE_URL}wasm/wllama.wasm`;
    const modelUrl = `${window.location.origin}${import.meta.env.BASE_URL}${dir}/${filename}`;

    const fallbackBytes = FALLBACK_TOTAL_BYTES[variant];
    onProgress({ loaded: 0, total: fallbackBytes, pct: 0 });

    const wllama = new Wllama({ default: wasmUrl }, { suppressNativeLog: true });

    let lastPct = 0;
    let lastLoaded = 0;
    const start = performance.now();

    await wllama.loadModelFromUrl(modelUrl, {
      n_ctx: 2048,
      n_batch: 256,
      n_threads: threads,
      flash_attn: true,
      cache_type_k: 'q8_0',
      cache_type_v: 'q8_0',
      reasoning: false,
      reasoning_format: 'none',
      default_template_kwargs: { enable_thinking: false },
      useCache: true,
      progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
        const safeTotal = total > 0 ? total : fallbackBytes;
        const safeLoaded = Math.max(loaded, lastLoaded);
        lastLoaded = safeLoaded;
        const pct = Math.max(lastPct, Math.min(100, Math.floor((safeLoaded / safeTotal) * 100)));
        lastPct = pct;
        onProgress({ loaded: safeLoaded, total: safeTotal, pct });
      },
    });

    this.wllama = wllama;
    onProgress({ loaded: 1, total: 1, pct: 100 });

    const loadSec = ((performance.now() - start) / 1000).toFixed(2);
    const totalCores = navigator.hardwareConcurrency || threads;
    const loadInfo = `[Edu-SLM] engine=wllama model=${filename} threads=${threads}/${totalCores} crossOriginIsolated=${window.crossOriginIsolated} loadSec=${loadSec}`;
    document.documentElement.dataset.eduLoad = loadInfo;
    console.info(loadInfo);
    void this.logResources('post-load');
  }

  /** Stream a chat reply. `onToken` receives incremental text deltas. */
  async generate(
    history: ChatMessage[],
    onToken: (delta: string) => void,
  ): Promise<GenerateResult> {
    if (!this.wllama) throw new Error('モデルが未ロードです。');

    this.abortController?.abort();
    const abortController = new AbortController();
    this.abortController = abortController;

    const started = performance.now();
    let raw = '';
    let content = '';
    let emittedLen = 0;
    let tokens = 0;
    let cacheN = 0;
    let promptN = 0;
    let promptMs = 0;
    let predictedN = 0;
    let predictedMs = 0;
    let predictedPerSecond = 0;
    let promptPerSecond = 0;

    const extractVisible = (text: string): string => {
      const trimmed = text.replace(/^\s+/, '');
      if (trimmed.startsWith('<think>')) {
        const closeIdx = trimmed.indexOf('</think>');
        if (closeIdx === -1) return '';
        return trimmed.slice(closeIdx + '</think>'.length).replace(/^\s+/, '');
      }
      // Hold back tiny prefix while checking if it starts with "<think>"
      if (trimmed.length < '<think>'.length && '<think>'.startsWith(trimmed)) {
        return '';
      }
      return trimmed;
    };

    try {
      const stream = await this.wllama.createChatCompletion({
        messages: [{ role: 'system', content: EDU_SYSTEM_PROMPT }, ...history],
        stream: true,
        abortSignal: abortController.signal,
        max_tokens: 200,
        temperature: 0.4,
        top_p: 0.9,
        top_k: 40,
        penalty_repeat: 1.08,
        cache_prompt: true,
        timings_per_token: true,
        chat_template_kwargs: { enable_thinking: false },
      });

      for await (const chunk of stream) {
        const delta = chunk.choices?.[0]?.delta?.content;
        if (delta) {
          raw += delta;
          tokens += 1;
          const visible = extractVisible(raw);
          if (visible.length > emittedLen) {
            const newPiece = visible.slice(emittedLen);
            emittedLen = visible.length;
            content = visible;
            onToken(newPiece);
          }
        }
        if (chunk.timings) {
          cacheN = chunk.timings.cache_n || cacheN;
          promptN = chunk.timings.prompt_n || promptN;
          promptMs = chunk.timings.prompt_ms || promptMs;
          predictedN = chunk.timings.predicted_n || predictedN;
          predictedMs = chunk.timings.predicted_ms || predictedMs;
          predictedPerSecond = chunk.timings.predicted_per_second || predictedPerSecond;
          promptPerSecond = chunk.timings.prompt_per_second || promptPerSecond;
        }
      }
    } catch (err) {
      if (abortController.signal.aborted) {
        // Stopped by the user: keep whatever was streamed so far.
        const elapsedSec = (performance.now() - started) / 1000;
        return {
          content,
          tokens,
          elapsedSec,
          tps: elapsedSec > 0 ? tokens / elapsedSec : 0,
        };
      }
      throw err;
    } finally {
      if (this.abortController === abortController) this.abortController = null;
    }

    const elapsedSec = (performance.now() - started) / 1000;
    const finalTokens = predictedN > 0 ? predictedN : tokens > 0 ? tokens : estimateTokens(content);
    const tps =
      predictedPerSecond > 0
        ? predictedPerSecond
        : elapsedSec > 0
          ? finalTokens / elapsedSec
          : 0;
    const computeSec = promptMs + predictedMs > 0 ? (promptMs + predictedMs) / 1000 : elapsedSec;
    const cpuThreadSec = computeSec * this.threads;
    const totalCores = navigator.hardwareConcurrency || this.threads;

    const genInfo =
      `[Edu-SLM] generated tokens=${finalTokens} elapsed=${elapsedSec.toFixed(2)}s tps=${tps.toFixed(2)} promptTps=${promptPerSecond.toFixed(2)} ` +
      `prompt=${promptN}tok(kvCache=${cacheN})/${(promptMs / 1000).toFixed(2)}s decode=${(predictedMs / 1000).toFixed(2)}s ` +
      `threads=${this.threads}/${totalCores} cpuThreadSec=${cpuThreadSec.toFixed(1)}s engine=wllama`;
    document.documentElement.dataset.eduGen = genInfo;
    console.info(genInfo);
    void this.logResources('turn');

    return { content, tokens: finalTokens, elapsedSec, tps };
  }

  /** Cancel the in-flight generation, if any. */
  abort(): void {
    this.abortController?.abort();
    this.abortController = null;
  }

  /** Clear conversation state. The loaded model stays warm; no page reload needed. */
  reset(): void {
    this.abort();
  }
}

export const wllamaClient = new WllamaClient();
export default wllamaClient;
