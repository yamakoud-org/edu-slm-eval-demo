import { Wllama } from '@wllama/wllama';

export const EDU_SYSTEM_PROMPT =
  'あなたは生徒の思考を引き出す教育チューターです。答えや完成文を直接教えず、温かく寄り添いながら小さな問いかけで生徒自身の気づきを促してください。';

export type ModelVariant = 'q4_0' | 'q4_k_m' | '1.5b_q4_0';

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

/** Fallback total sizes when the server omits Content-Length. */
const FALLBACK_TOTAL_BYTES: Record<ModelVariant, number> = {
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
  if (requested === 'q4_0') return 'q4_0';
  return 'q4_k_m';
};

const modelFilename = (variant: ModelVariant): string => {
  if (variant === '1.5b_q4_0') return 'edu-slm-1.5b-q4_0.gguf';
  return `edu-slm-0.5b-${variant}.gguf`;
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
  private variant: ModelVariant = 'q4_0';
  private threads = 1;
  private loadPromise: Promise<void> | null = null;
  private abortController: AbortController | null = null;

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

    const filename = modelFilename(variant);
    const wasmUrl = `${window.location.origin}${import.meta.env.BASE_URL}wasm/wllama.wasm`;
    const modelUrl = `${window.location.origin}${import.meta.env.BASE_URL}models/edu-slm-0.5b-gguf/${filename}`;

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
    const loadInfo = `[Edu-SLM] engine=wllama model=${filename} threads=${threads} crossOriginIsolated=${window.crossOriginIsolated} loadSec=${loadSec}`;
    document.documentElement.dataset.eduLoad = loadInfo;
    console.info(loadInfo);
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
    let content = '';
    let tokens = 0;
    let predictedN = 0;
    let predictedPerSecond = 0;
    let promptPerSecond = 0;

    try {
      const stream = await this.wllama.createChatCompletion({
        messages: [{ role: 'system', content: EDU_SYSTEM_PROMPT }, ...history],
        stream: true,
        abortSignal: abortController.signal,
        max_tokens: 200,
        temperature: 0.35,
        top_p: 0.9,
        top_k: 40,
        penalty_repeat: 1.12,
        cache_prompt: true,
        timings_per_token: true,
      });

      for await (const chunk of stream) {
        const delta = chunk.choices?.[0]?.delta?.content;
        if (delta) {
          content += delta;
          tokens += 1;
          onToken(delta);
        }
        if (chunk.timings) {
          predictedN = chunk.timings.predicted_n || predictedN;
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

    const genInfo = `[Edu-SLM] generated tokens=${finalTokens} elapsed=${elapsedSec.toFixed(2)}s tps=${tps.toFixed(2)} promptTps=${promptPerSecond.toFixed(2)} engine=wllama threads=${this.threads}`;
    document.documentElement.dataset.eduGen = genInfo;
    console.info(genInfo);

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
