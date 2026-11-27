import {
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  StoppingCriteriaList,
  TextStreamer,
  env,
} from '@huggingface/transformers';

type ModelSource = 'local_edited' | 'hf_baseline';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

type WorkerInMessage =
  | {
      type: 'load';
      modelSource: ModelSource;
      baseUrl: string;
      preferredDevice: 'webgpu' | 'wasm';
    }
  | {
      type: 'generate';
      messages: ChatMessage[];
      maxNewTokens: number;
    }
  | {
      type: 'abort';
    };

let tokenizer: Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>> | null = null;
let model: Awaited<ReturnType<typeof AutoModelForCausalLM.from_pretrained>> | null = null;
let loadedSource: ModelSource | null = null;
let loadedDevice: 'webgpu' | 'wasm' = 'wasm';
const stoppingCriteria = new InterruptableStoppingCriteria();

interface GPUAdapterLike {
  isFallbackAdapter?: boolean;
  info?: { vendor?: string; architecture?: string; description?: string };
}

async function detectHardwareWebGPU(): Promise<{ ok: boolean; label: string }> {
  try {
    const nav = self.navigator as Navigator & {
      gpu?: {
        requestAdapter: (opts?: { powerPreference?: string }) => Promise<GPUAdapterLike | null>;
      };
    };
    if (!nav.gpu) return { ok: false, label: 'no-navigator-gpu' };
    const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { ok: false, label: 'no-adapter' };

    const desc = [
      adapter.info?.vendor ?? '',
      adapter.info?.architecture ?? '',
      adapter.info?.description ?? '',
    ]
      .join(' ')
      .toLowerCase();

    const isSoftware =
      Boolean(adapter.isFallbackAdapter) ||
      /swiftshader|llvmpipe|softpipe|lavapipe|microsoft basic render|cpu/.test(desc);

    if (isSoftware) {
      return { ok: false, label: `software-adapter-rejected(${desc.trim() || 'fallback'})` };
    }
    return { ok: true, label: desc.trim() || 'hardware-webgpu' };
  } catch (e) {
    return { ok: false, label: `error(${String(e)})` };
  }
}

self.addEventListener('message', async (event: MessageEvent<WorkerInMessage>) => {
  const msg = event.data;

  if (msg.type === 'abort') {
    stoppingCriteria.interrupt();
    return;
  }

  if (msg.type === 'load') {
    try {
      const gpuInfo = await detectHardwareWebGPU();
      const targetDevice: 'webgpu' | 'wasm' =
        msg.preferredDevice === 'webgpu' && gpuInfo.ok ? 'webgpu' : 'wasm';

      const onnxEnv = env.backends?.onnx as
        | { wasm?: { numThreads?: number; simd?: boolean; proxy?: boolean } }
        | undefined;
      const numThreads = self.crossOriginIsolated
        ? Math.min(Math.max(self.navigator.hardwareConcurrency || 4, 2), 4)
        : 1;
      if (onnxEnv?.wasm) {
        onnxEnv.wasm.simd = true;
        onnxEnv.wasm.numThreads = numThreads;
      }

      const selectedDtype =
        msg.modelSource === 'local_edited'
          ? targetDevice === 'webgpu'
            ? 'q4'
            : 'q8'
          : 'q4f16';

      self.postMessage({
        type: 'meta',
        device: targetDevice,
        dtype: selectedDtype,
        expectedBytes: selectedDtype === 'q8' ? 1_670_000_000 : 1_090_000_000,
      });

      console.info(
        `[Edu-SLM] device=${targetDevice} dtype=${selectedDtype} gpu=${gpuInfo.label} crossOriginIsolated=${Boolean(self.crossOriginIsolated)} wasmThreads=${numThreads}`,
      );

      if (model && tokenizer && loadedSource === msg.modelSource && loadedDevice === targetDevice) {
        self.postMessage({ type: 'ready', device: targetDevice, modelSource: msg.modelSource });
        return;
      }

      // Release previous model if switching
      if (model) {
        try {
          await (model as unknown as { dispose?: () => Promise<void> }).dispose?.();
        } catch {
          // ignore
        }
        model = null;
        tokenizer = null;
      }

      const progress_callback = (p: Record<string, unknown>) => {
        self.postMessage({
          type: 'progress',
          status: String(p.status ?? ''),
          file: String(p.file ?? ''),
          progress: typeof p.progress === 'number' ? p.progress : undefined,
          loaded: typeof p.loaded === 'number' ? p.loaded : undefined,
          total: typeof p.total === 'number' ? p.total : undefined,
        });
      };

      let modelId: string;
      if (msg.modelSource === 'local_edited') {
        env.allowLocalModels = true;
        env.allowRemoteModels = false;
        env.localModelPath = `${msg.baseUrl}models/`;
        modelId = 'edu-slm-q4-b32';
      } else {
        env.allowLocalModels = false;
        env.allowRemoteModels = true;
        modelId = 'onnx-community/Qwen2.5-1.5B-Instruct';
      }

      self.postMessage({
        type: 'status',
        message: `トークナイザーを読み込み中 (${modelId})...`,
      });
      tokenizer = await AutoTokenizer.from_pretrained(modelId, { progress_callback });

      self.postMessage({
        type: 'status',
        message: `ONNX モデル (${selectedDtype}) を ${targetDevice.toUpperCase()} へ展開中...`,
      });
      model = await AutoModelForCausalLM.from_pretrained(modelId, {
        dtype: selectedDtype,
        device: targetDevice,
        progress_callback,
      });

      loadedSource = msg.modelSource;
      loadedDevice = targetDevice;
      self.postMessage({ type: 'ready', device: targetDevice, modelSource: msg.modelSource });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      self.postMessage({ type: 'error', message });
    }
    return;
  }

  if (msg.type === 'generate') {
    if (!model || !tokenizer) {
      self.postMessage({ type: 'error', message: 'モデルが未ロードです。先にモデルをロードしてください。' });
      return;
    }
    try {
      stoppingCriteria.reset();
      const inputs = tokenizer.apply_chat_template(msg.messages, {
        add_generation_prompt: true,
        return_dict: true,
      }) as Record<string, unknown>;

      let tokenCount = 0;
      const startTime = performance.now();

      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (chunk: string) => {
          tokenCount += 1;
          const elapsedSec = (performance.now() - startTime) / 1000;
          self.postMessage({
            type: 'token',
            chunk,
            tokens: tokenCount,
            tps: elapsedSec > 0 ? tokenCount / elapsedSec : 0,
            elapsedSec,
          });
        },
      });

      const criteriaList = new StoppingCriteriaList();
      criteriaList.push(stoppingCriteria);

      const genParams = {
        ...inputs,
        max_new_tokens: msg.maxNewTokens,
        do_sample: false,
        repetition_penalty: 1.08,
        streamer,
        stopping_criteria: criteriaList,
      } as Parameters<typeof model.generate>[0];
      await model.generate(genParams);

      const elapsedSec = (performance.now() - startTime) / 1000;
      const tps = elapsedSec > 0 ? tokenCount / elapsedSec : 0;
      console.info(
        `[Edu-SLM] generated tokens=${tokenCount} elapsed=${elapsedSec.toFixed(2)}s tps=${tps.toFixed(2)} device=${loadedDevice}`,
      );
      self.postMessage({
        type: 'done',
        tokens: tokenCount,
        tps,
        elapsedSec,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      self.postMessage({ type: 'error', message });
    }
  }
});
