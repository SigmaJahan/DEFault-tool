"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  Save, FolderOpen, Download, Trash2, Clock, Check, X,
} from "lucide-react";
import type { SavedSession } from "@/lib/sessions";
import {
  getSessions, saveSession, deleteSession,
  exportSessionJSON, relativeTime,
} from "@/lib/sessions";

interface SessionManagerProps {
  /** Current editor state — used when saving */
  currentState: {
    code: string;
    modelName: string;
    dataMode: "dummy" | "uploaded";
    trainingEpochs: number;
    numSamples: number;
    fullResult: import("@/types/analysis").FullAnalysisResponse | null;
    codeResult: import("@/types/analysis").AnalyzeCodeResponse | null;
    liveMetrics: import("@/types/analysis").EpochMetric[];
    isDummyData: boolean;
  };
  /** Called when user picks a saved session to restore */
  onLoad: (session: SavedSession) => void;
  /** True while analysis is running (disables save) */
  isRunning: boolean;
  /** True when there are results to export */
  hasResults: boolean;
}

export function SessionManager({
  currentState,
  onLoad,
  isRunning,
  hasResults,
}: SessionManagerProps) {
  const [open, setOpen] = useState(false);
  const [sessions, setSessions] = useState<SavedSession[]>([]);
  const [saveName, setSaveName] = useState("");
  const [saving, setSaving] = useState(false);  // name-input mode
  const [justSaved, setJustSaved] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Refresh session list whenever the dropdown opens
  useEffect(() => {
    if (open) setSessions(getSessions());
  }, [open]);

  // Close on outside click
  useEffect(() => {
    function handle(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setSaving(false);
      }
    }
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, []);

  // Focus name input when save mode opens
  useEffect(() => {
    if (saving) inputRef.current?.focus();
  }, [saving]);

  const handleSave = useCallback((name: string) => {
    const trimmed = name.trim() || `Session ${new Date().toLocaleTimeString()}`;
    saveSession({ ...currentState, name: trimmed });
    setSessions(getSessions());
    setSaveName("");
    setSaving(false);
    setJustSaved(trimmed);
    setTimeout(() => setJustSaved(null), 2500);
  }, [currentState]);

  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    deleteSession(id);
    setSessions(getSessions());
  };

  const handleExport = () => {
    const all = getSessions();
    // Export the most recent session that has results, or build one on-the-fly
    const withResults = all.find((s) => s.fullResult || s.codeResult);
    const toExport: SavedSession = withResults ?? {
      ...currentState,
      id: `export-${Date.now()}`,
      name: "Export",
      savedAt: new Date().toISOString(),
    };
    exportSessionJSON(toExport);
    setOpen(false);
  };

  // ⌘S / Ctrl+S — save with timestamp name
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault();
        if (!isRunning) {
          setOpen(true);
          setSaving(true);
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isRunning]);

  return (
    <div ref={ref} className="relative" style={{ zIndex: 50 }}>
      {/* Trigger button */}
      <button
        className="btn-ghost flex items-center gap-1.5 text-[11px] px-2.5 py-1.5 rounded-md"
        onClick={() => { setOpen((o) => !o); setSaving(false); }}
        aria-label="Open session manager"
        aria-expanded={open}
        aria-haspopup="true"
        title="Sessions (⌘S to save)"
      >
        {justSaved ? (
          <>
            <Check size={12} aria-hidden="true" style={{ color: "var(--safe-green)" }} />
            <span style={{ color: "var(--safe-green)" }}>Saved</span>
          </>
        ) : (
          <>
            <FolderOpen size={12} aria-hidden="true" />
            <span>Sessions</span>
          </>
        )}
      </button>

      {/* Dropdown panel */}
      {open && (
        <div
          className="absolute right-0 top-full mt-1.5 rounded-xl shadow-panel overflow-hidden animate-fade-in"
          style={{
            width: 280,
            background: "var(--bg-overlay)",
            border: "1px solid var(--border)",
          }}
          role="dialog"
          aria-label="Session manager"
        >
          {/* Header row */}
          <div
            className="flex items-center justify-between px-3 py-2.5"
            style={{ borderBottom: "1px solid var(--border-subtle)" }}
          >
            <span className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: "var(--text-muted)" }}>
              Sessions
            </span>
            <div className="flex gap-1">
              {/* Export button */}
              {hasResults && (
                <button
                  className="btn-ghost p-1 rounded"
                  onClick={handleExport}
                  title="Export current results as JSON"
                  aria-label="Export results as JSON"
                >
                  <Download size={12} aria-hidden="true" />
                </button>
              )}
              {/* Save button */}
              <button
                className="btn-ghost p-1 rounded flex items-center gap-1 text-[11px]"
                onClick={() => setSaving((v) => !v)}
                disabled={isRunning}
                title="Save session (⌘S)"
                aria-label="Save current session"
              >
                <Save size={12} aria-hidden="true" />
              </button>
            </div>
          </div>

          {/* Save-as-name form */}
          {saving && (
            <div
              className="flex items-center gap-2 px-3 py-2"
              style={{ borderBottom: "1px solid var(--border-subtle)", background: "var(--bg-card)" }}
            >
              <input
                ref={inputRef}
                className="flex-1 text-xs px-2 py-1 rounded-md font-mono"
                style={{
                  background: "var(--bg-input)",
                  border: "1px solid var(--border)",
                  color: "var(--text-primary)",
                  outline: "none",
                }}
                placeholder="Session name…"
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSave(saveName);
                  if (e.key === "Escape") { setSaving(false); setSaveName(""); }
                }}
                aria-label="Session name"
              />
              <button
                className="p-1 rounded"
                style={{ color: "var(--safe-green)" }}
                onClick={() => handleSave(saveName)}
                aria-label="Confirm save"
                title="Save (Enter)"
              >
                <Check size={13} aria-hidden="true" />
              </button>
              <button
                className="p-1 rounded"
                style={{ color: "var(--text-muted)" }}
                onClick={() => { setSaving(false); setSaveName(""); }}
                aria-label="Cancel"
                title="Cancel (Esc)"
              >
                <X size={13} aria-hidden="true" />
              </button>
            </div>
          )}

          {/* Session list */}
          <div
            className="overflow-y-auto"
            style={{ maxHeight: 280 }}
            role="list"
            aria-label="Saved sessions"
          >
            {sessions.length === 0 ? (
              <div className="px-4 py-6 text-center text-[11px]" style={{ color: "var(--text-muted)" }}>
                No saved sessions yet.
                <br />
                <span style={{ color: "var(--text-secondary)" }}>Press ⌘S to save one.</span>
              </div>
            ) : (
              sessions.map((s) => (
                <button
                  key={s.id}
                  className="w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors session-row"
                  onClick={() => { onLoad(s); setOpen(false); }}
                  role="listitem"
                  aria-label={`Load session: ${s.name}`}
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium truncate" style={{ color: "var(--text-primary)" }}>
                      {s.name}
                    </div>
                    <div className="flex items-center gap-2 mt-0.5 text-[10px]" style={{ color: "var(--text-muted)" }}>
                      <Clock size={9} aria-hidden="true" />
                      <span>{relativeTime(s.savedAt)}</span>
                      <span style={{ color: "var(--border)" }}>·</span>
                      <span className="font-mono truncate" style={{ color: "var(--text-code)", maxWidth: 90 }}>
                        {s.modelName}
                      </span>
                      {(s.fullResult || s.codeResult) && (
                        <>
                          <span style={{ color: "var(--border)" }}>·</span>
                          <span style={{ color: s.fullResult?.stage1_detection?.predicted_positive ? "var(--fault-red)" : "var(--safe-green)" }}>
                            {s.fullResult?.stage1_detection?.predicted_positive ? "fault" : "ok"}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                  <button
                    className="flex-shrink-0 p-1 rounded opacity-0 session-delete transition-opacity"
                    onClick={(e) => handleDelete(s.id, e)}
                    aria-label={`Delete session ${s.name}`}
                    title="Delete session"
                  >
                    <Trash2 size={11} style={{ color: "var(--text-muted)" }} aria-hidden="true" />
                  </button>
                </button>
              ))
            )}
          </div>

          {/* Footer hint */}
          <div
            className="px-3 py-2 text-[9px]"
            style={{
              borderTop: "1px solid var(--border-subtle)",
              color: "var(--text-muted)",
            }}
          >
            ⌘S save · click row to restore · {sessions.length}/{10} slots used
          </div>
        </div>
      )}
    </div>
  );
}
