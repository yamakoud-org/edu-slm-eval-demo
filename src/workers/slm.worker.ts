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

async function detectWebGPU(): Promise<boolean> {
  try {
    const nav = self.navigator as Navigator & {
      gpu?: { requestAdapter: () => Promise<unknown | null> };
    };
    if (!nav.gpu) return false;
    const adapter = await nav.gpu.requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
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
      const hasWebGPU = await detectWebGPU();
      const targetDevice: 'webgpu' | 'wasm' =
        msg.preferredDevice === 'webgpu' && hasWebGPU ? 'webgpu' : 'wasm';

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
        message: `4bit ONNX モデルを ${targetDevice.toUpperCase()} へ展開中...`,
      });
      model = await AutoModelForCausalLM.from_pretrained(modelId, {
        dtype: msg.modelSource === 'local_edited' ? 'q4' : 'q4f16',
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
      self.postMessage({
        type: 'done',
        tokens: tokenCount,
        tps: elapsedSec > 0 ? tokenCount / elapsedSec : 0,
        elapsedSec,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      self.postMessage({ type: 'error', message });
    }
  }
});
