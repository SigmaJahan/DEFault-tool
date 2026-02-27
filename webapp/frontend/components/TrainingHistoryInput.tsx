"use client";

import { useState, useCallback } from "react";
import { ChevronDown, ChevronRight, AlertCircle } from "lucide-react";
import type { TrainingHistory } from "@/types/analysis";

interface TrainingHistoryInputProps {
  value: TrainingHistory | null;
  onChange: (history: TrainingHistory | null) => void;
  disabled?: boolean;
}

const PLACEHOLDER = `# Paste your Keras history.history dict values, one array per line.
# Example: copy from your training output:
loss      = [0.65, 0.52, 0.44, 0.38, 0.33]
val_loss  = [0.68, 0.58, 0.54, 0.57, 0.60]
train_acc = [0.61, 0.72, 0.79, 0.83, 0.87]
val_acc   = [0.59, 0.69, 0.73, 0.72, 0.71]`;

type FieldKey = "loss" | "val_loss" | "train_acc" | "val_acc";

const FIELD_LABELS: { key: FieldKey; label: string; placeholder: string }[] = [
  { key: "loss",      label: "Train Loss",    placeholder: "0.65, 0.52, 0.44, ..." },
  { key: "val_loss",  label: "Val Loss",      placeholder: "0.68, 0.58, 0.54, ..." },
  { key: "train_acc", label: "Train Acc",     placeholder: "0.61, 0.72, 0.79, ..." },
  { key: "val_acc",   label: "Val Acc",       placeholder: "0.59, 0.69, 0.73, ..." },
];

function parseArray(s: string): number[] | null {
  const cleaned = s.replace(/[\[\](){}]/g, "").trim();
  if (!cleaned) return null;
  const parts = cleaned.split(/[\s,]+/).filter(Boolean);
  const nums = parts.map(Number);
  if (nums.some(isNaN)) return null;
  return nums;
}

function parseBulkText(text: string): Partial<Record<FieldKey, number[]>> {
  const result: Partial<Record<FieldKey, number[]>> = {};
  const keyMap: Record<string, FieldKey> = {
    loss: "loss", train_loss: "loss",
    val_loss: "val_loss", validation_loss: "val_loss",
    acc: "train_acc", train_acc: "train_acc", accuracy: "train_acc",
    val_acc: "val_acc", val_accuracy: "val_acc", validation_acc: "val_acc",
  };

  for (const line of text.split("\n")) {
    const m = line.match(/^\s*#?([a-z_A-Z]+)\s*[=:]\s*(.+)/);
    if (!m) continue;
    const key = keyMap[m[1].toLowerCase()];
    if (!key) continue;
    const arr = parseArray(m[2]);
    if (arr) result[key] = arr;
  }
  return result;
}

export function TrainingHistoryInput({ value, onChange, disabled }: TrainingHistoryInputProps) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"fields" | "bulk">("fields");
  const [fields, setFields] = useState<Record<FieldKey, string>>({
    loss: "", val_loss: "", train_acc: "", val_acc: "",
  });
  const [bulkText, setBulkText] = useState("");
  const [error, setError] = useState<string | null>(null);

  const hasValue = value != null;

  function updateField(key: FieldKey, raw: string) {
    setFields((prev) => ({ ...prev, [key]: raw }));
    setError(null);
  }

  const applyFields = useCallback(() => {
    const parsed: Partial<Record<FieldKey, number[]>> = {};
    for (const { key } of FIELD_LABELS) {
      const arr = parseArray(fields[key]);
      if (!arr) {
        setError(`Invalid numbers in "${key}": separate values with commas.`);
        return;
      }
      parsed[key] = arr;
    }
    const lengths = FIELD_LABELS.map(({ key }) => parsed[key]!.length);
    if (new Set(lengths).size !== 1) {
      setError(
        `All arrays must have the same length. Got: loss=${lengths[0]}, val_loss=${lengths[1]}, train_acc=${lengths[2]}, val_acc=${lengths[3]}`
      );
      return;
    }
    if (lengths[0] === 0) {
      setError("Arrays cannot be empty.");
      return;
    }
    setError(null);
    onChange({
      loss:      parsed.loss!,
      val_loss:  parsed.val_loss!,
      train_acc: parsed.train_acc!,
      val_acc:   parsed.val_acc!,
    });
  }, [fields, onChange]);

  const applyBulk = useCallback(() => {
    const parsed = parseBulkText(bulkText);
    const missing = FIELD_LABELS.filter(({ key }) => !parsed[key]).map(({ label }) => label);
    if (missing.length) {
      setError(`Could not parse: ${missing.join(", ")}. Check format: key = [v1, v2, ...]`);
      return;
    }
    const lengths = FIELD_LABELS.map(({ key }) => parsed[key]!.length);
    if (new Set(lengths).size !== 1) {
      setError(`Array lengths don't match: ${lengths.join(", ")}`);
      return;
    }
    setError(null);
    onChange({
      loss:      parsed.loss!,
      val_loss:  parsed.val_loss!,
      train_acc: parsed.train_acc!,
      val_acc:   parsed.val_acc!,
    });
  }, [bulkText, onChange]);

  function handleClear() {
    onChange(null);
    setFields({ loss: "", val_loss: "", train_acc: "", val_acc: "" });
    setBulkText("");
    setError(null);
  }

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ border: "1px solid var(--border)" }}
    >
      {/* Header toggle */}
      <button
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left"
        style={{ background: "var(--bg-card)" }}
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
      >
        {open ? (
          <ChevronDown size={12} style={{ color: "var(--text-muted)" }} />
        ) : (
          <ChevronRight size={12} style={{ color: "var(--text-muted)" }} />
        )}
        <span className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
          Training History
        </span>
        {hasValue && value && (
          <span
            className="ml-auto text-[10px] font-mono px-1.5 py-0.5 rounded"
            style={{
              background: "var(--safe-green-bg)",
              color: "var(--safe-green)",
              border: "1px solid var(--safe-green)",
            }}
          >
            {value.loss.length} epochs ✓
          </span>
        )}
        {!hasValue && (
          <span
            className="ml-auto text-[10px]"
            style={{ color: "var(--text-muted)" }}
          >
            optional: enables full 3-stage diagnosis
          </span>
        )}
      </button>

      {/* Expanded body */}
      {open && (
        <div
          className="px-3 pb-3 pt-2 space-y-3"
          style={{ background: "var(--bg-overlay)", borderTop: "1px solid var(--border-subtle)" }}
        >
          {/* Mode toggle */}
          <div className="flex gap-0 rounded overflow-hidden" style={{ border: "1px solid var(--border)" }}>
            {(["fields", "bulk"] as const).map((m) => (
              <button
                key={m}
                className="flex-1 text-[10px] py-1 uppercase tracking-wide"
                style={{
                  background: mode === m ? "var(--accent)" : "var(--bg-card)",
                  color: mode === m ? "#fff" : "var(--text-muted)",
                }}
                onClick={() => setMode(m)}
              >
                {m === "fields" ? "Fields" : "Paste / Bulk"}
              </button>
            ))}
          </div>

          {mode === "fields" ? (
            /* Individual field inputs */
            <div className="space-y-2">
              {FIELD_LABELS.map(({ key, label, placeholder }) => (
                <div key={key}>
                  <label
                    className="text-[9px] uppercase tracking-wider block mb-0.5"
                    style={{ color: "var(--text-muted)" }}
                  >
                    {label}
                  </label>
                  <input
                    className="w-full text-[11px] font-mono px-2 py-1.5 rounded"
                    style={{
                      background: "var(--bg-input)",
                      border: "1px solid var(--border)",
                      color: "var(--text-primary)",
                      outline: "none",
                    }}
                    placeholder={placeholder}
                    value={fields[key]}
                    onChange={(e) => updateField(key, e.target.value)}
                    disabled={disabled}
                  />
                </div>
              ))}
            </div>
          ) : (
            /* Bulk paste area */
            <textarea
              className="w-full text-[10px] font-mono px-2 py-2 rounded resize-none"
              style={{
                background: "var(--bg-input)",
                border: "1px solid var(--border)",
                color: "var(--text-primary)",
                outline: "none",
                height: 130,
              }}
              placeholder={PLACEHOLDER}
              value={bulkText}
              onChange={(e) => { setBulkText(e.target.value); setError(null); }}
              disabled={disabled}
            />
          )}

          {/* Error message */}
          {error && (
            <div
              className="flex items-start gap-1.5 text-[10px] px-2 py-1.5 rounded"
              style={{
                background: "var(--fault-red-bg)",
                border: "1px solid var(--fault-red)",
                color: "var(--fault-red)",
              }}
            >
              <AlertCircle size={11} className="flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              className="flex-1 text-[11px] py-1.5 rounded font-medium"
              style={{
                background: "var(--accent)",
                color: "#fff",
                opacity: disabled ? 0.5 : 1,
              }}
              disabled={disabled}
              onClick={mode === "fields" ? applyFields : applyBulk}
            >
              Apply
            </button>
            {hasValue && (
              <button
                className="text-[11px] py-1.5 px-3 rounded"
                style={{
                  background: "var(--bg-card)",
                  border: "1px solid var(--border)",
                  color: "var(--text-muted)",
                }}
                onClick={handleClear}
              >
                Clear
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
