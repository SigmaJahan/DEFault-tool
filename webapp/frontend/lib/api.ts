import type {
  AnalyzeCodeResponse,
  DataMode,
  FaultTaxonomyResponse,
  FullAnalysisResponse,
  TrainEvent,
  TrainingHistory,
} from "@/types/analysis";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail));
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail));
  }
  return res.json();
}

export async function analyzeCode(
  code: string,
  modelName = "pasted_model"
): Promise<AnalyzeCodeResponse> {
  return post<AnalyzeCodeResponse>("/api/analyze-code", { code, model_name: modelName });
}

export async function analyzeHistory(
  history: TrainingHistory,
  code: string | null,
  modelName = "pasted_model"
): Promise<FullAnalysisResponse> {
  return post<FullAnalysisResponse>("/api/analyze-history", {
    loss: history.loss,
    val_loss: history.val_loss,
    train_acc: history.train_acc,
    val_acc: history.val_acc,
    code: code ?? undefined,
    model_name: modelName,
  });
}

export async function getFaultTaxonomy(): Promise<FaultTaxonomyResponse> {
  return get<FaultTaxonomyResponse>("/api/fault-taxonomy");
}

export async function* streamTraining(
  params: {
    code: string;
    modelName: string;
    dataMode: DataMode;
    epochs: number;
    numSamples: number;
    datasetFile: File | null;
  },
  signal?: AbortSignal,
): AsyncGenerator<TrainEvent> {
  const form = new FormData();
  form.append("code",        params.code);
  form.append("model_name",  params.modelName);
  form.append("data_mode",   params.dataMode);
  form.append("epochs",      String(params.epochs));
  form.append("num_samples", String(params.numSamples));
  if (params.datasetFile) {
    form.append("dataset_file", params.datasetFile);
  }

  const res = await fetch(`${BASE}/api/train-and-diagnose`, {
    method: "POST",
    body:   form,
    signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail));
  }

  const reader  = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE format: "data: {...}\n\n"
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const trimmed = part.trim();
      if (trimmed.startsWith("data: ")) {
        try {
          yield JSON.parse(trimmed.slice(6)) as TrainEvent;
        } catch {
          // skip malformed lines
        }
      }
    }
  }

  // Drain any remaining buffered data
  if (buffer.trim().startsWith("data: ")) {
    try {
      yield JSON.parse(buffer.trim().slice(6)) as TrainEvent;
    } catch {
      // ignore
    }
  }
}
