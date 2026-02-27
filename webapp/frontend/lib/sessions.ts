import type {
  DataMode,
  EpochMetric,
  FullAnalysisResponse,
  AnalyzeCodeResponse,
} from "@/types/analysis";

// ── Session shape ──────────────────────────────────────────────────────────

export interface SavedSession {
  id: string;
  name: string;
  savedAt: string; // ISO 8601
  code: string;
  modelName: string;
  dataMode: DataMode;
  trainingEpochs: number;
  numSamples: number;
  fullResult: FullAnalysisResponse | null;
  codeResult: AnalyzeCodeResponse | null;
  liveMetrics: EpochMetric[];
  isDummyData: boolean;
}

// ── Storage helpers ────────────────────────────────────────────────────────

const STORAGE_KEY = "default-tool-sessions";
const MAX_SESSIONS = 10;

export function getSessions(): SavedSession[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as SavedSession[];
  } catch {
    return [];
  }
}

export function saveSession(
  partial: Omit<SavedSession, "id" | "savedAt">
): SavedSession {
  const session: SavedSession = {
    ...partial,
    id: `s-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    savedAt: new Date().toISOString(),
  };
  // Replace any existing session with the same name
  const rest = getSessions().filter((s) => s.name !== session.name);
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify([session, ...rest].slice(0, MAX_SESSIONS))
  );
  return session;
}

export function deleteSession(id: string): void {
  const updated = getSessions().filter((s) => s.id !== id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
}

// ── Auto-draft ─────────────────────────────────────────────────────────────
// Saves silently under the name "Draft" every time called.

export function saveDraft(partial: Omit<SavedSession, "id" | "savedAt" | "name">): void {
  saveSession({ ...partial, name: "Draft" });
}

// ── Export ─────────────────────────────────────────────────────────────────

export function exportSessionJSON(session: SavedSession): void {
  const blob = new Blob([JSON.stringify(session, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `default-${session.modelName.replace(/\s+/g, "_")}-${session.savedAt.slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Time formatting ────────────────────────────────────────────────────────

export function relativeTime(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const m = Math.floor(ms / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}
