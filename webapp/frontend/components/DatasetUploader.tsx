"use client";

import { useRef, useState } from "react";
import { ChevronDown, ChevronRight, Upload, X, AlertTriangle } from "lucide-react";
import type { DataMode } from "@/types/analysis";

interface DatasetUploaderProps {
  dataMode: DataMode;
  uploadedFile: File | null;
  epochs: number;
  numSamples: number;
  disabled?: boolean;
  onDataModeChange: (mode: DataMode) => void;
  onFileChange: (file: File | null) => void;
  onEpochsChange: (n: number) => void;
  onNumSamplesChange: (n: number) => void;
}

export function DatasetUploader({
  dataMode,
  uploadedFile,
  epochs,
  numSamples,
  disabled = false,
  onDataModeChange,
  onFileChange,
  onEpochsChange,
  onNumSamplesChange,
}: DatasetUploaderProps) {
  const [open, setOpen] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleFile(file: File) {
    const name = file.name.toLowerCase();
    if (!name.endsWith(".csv") && !name.endsWith(".npy") && !name.endsWith(".npz")) {
      alert("Please upload a .csv, .npy, or .npz file.");
      return;
    }
    onFileChange(file);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function formatBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  }

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ border: "1px solid var(--border-subtle)", background: "var(--bg-card)" }}
    >
      {/* Header toggle */}
      <button
        className="w-full flex items-center justify-between px-3 py-2 text-left"
        style={{ background: "transparent", cursor: "pointer" }}
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
      >
        <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: "var(--text-muted)" }}>
          Training Configuration
        </span>
        {open
          ? <ChevronDown size={12} style={{ color: "var(--text-muted)" }} />
          : <ChevronRight size={12} style={{ color: "var(--text-muted)" }} />}
      </button>

      {open && (
        <div className="px-3 pb-3 flex flex-col gap-3">

          {/* Mode tabs */}
          <div
            className="flex rounded-md overflow-hidden"
            style={{ border: "1px solid var(--border-subtle)" }}
          >
            {(["dummy", "uploaded"] as DataMode[]).map((m) => (
              <button
                key={m}
                className="flex-1 text-[11px] py-1.5 font-medium transition-all"
                style={{
                  background: dataMode === m ? "var(--accent)" : "var(--bg-overlay)",
                  color:      dataMode === m ? "#fff" : "var(--text-muted)",
                  cursor:     disabled ? "not-allowed" : "pointer",
                }}
                onClick={() => !disabled && onDataModeChange(m)}
                disabled={disabled}
              >
                {m === "dummy" ? "Dummy Data" : "Upload Dataset"}
              </button>
            ))}
          </div>

          {/* Dummy mode: samples slider + warning */}
          {dataMode === "dummy" && (
            <>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>Samples</span>
                  <span className="text-[10px] font-mono" style={{ color: "var(--text-code)" }}>{numSamples}</span>
                </div>
                <input
                  type="range"
                  min={50}
                  max={2000}
                  step={50}
                  value={numSamples}
                  disabled={disabled}
                  onChange={(e) => onNumSamplesChange(Number(e.target.value))}
                  className="w-full accent-[var(--accent)]"
                  style={{ cursor: disabled ? "not-allowed" : "pointer" }}
                />
              </div>
              <div
                className="flex items-start gap-1.5 text-[10px] px-2 py-1.5 rounded"
                style={{
                  background: "var(--warn-yellow-bg)",
                  border: "1px solid var(--warn-yellow)",
                  color: "var(--warn-yellow)",
                }}
              >
                <AlertTriangle size={10} className="flex-shrink-0 mt-0.5" />
                <span>Results will be labeled <strong>SYNTHETIC DATA</strong></span>
              </div>
            </>
          )}

          {/* Upload mode: drag & drop zone */}
          {dataMode === "uploaded" && (
            <>
              {uploadedFile ? (
                <div
                  className="flex items-center justify-between px-2 py-1.5 rounded text-[11px]"
                  style={{ background: "var(--bg-overlay)", border: "1px solid var(--border-subtle)" }}
                >
                  <span style={{ color: "var(--text-secondary)" }} className="truncate mr-2">
                    {uploadedFile.name}
                    <span className="ml-1" style={{ color: "var(--text-muted)" }}>
                      ({formatBytes(uploadedFile.size)})
                    </span>
                  </span>
                  <button
                    onClick={() => !disabled && onFileChange(null)}
                    disabled={disabled}
                    style={{ color: "var(--text-muted)", cursor: disabled ? "not-allowed" : "pointer" }}
                  >
                    <X size={12} />
                  </button>
                </div>
              ) : (
                <div
                  className="flex flex-col items-center justify-center gap-1.5 py-4 rounded-lg text-center"
                  style={{
                    border: `1.5px dashed ${dragOver ? "var(--accent)" : "var(--border-subtle)"}`,
                    background: dragOver ? "rgba(88,166,255,0.05)" : "transparent",
                    cursor: disabled ? "not-allowed" : "pointer",
                    transition: "all 0.15s ease",
                  }}
                  onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={onDrop}
                  onClick={() => !disabled && fileInputRef.current?.click()}
                >
                  <Upload size={14} style={{ color: "var(--text-muted)" }} />
                  <span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>
                    Drop file or click to upload
                  </span>
                  <span className="text-[9px]" style={{ color: "var(--text-muted)" }}>
                    .csv · .npy · .npz
                  </span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.npy,.npz"
                    className="hidden"
                    disabled={disabled}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleFile(f);
                      e.target.value = "";
                    }}
                  />
                </div>
              )}
              <div className="text-[9px] leading-tight" style={{ color: "var(--text-muted)" }}>
                CSV: last column = labels. NPZ: use <code className="font-mono">X</code>/<code className="font-mono">y</code> keys.
              </div>
            </>
          )}

          {/* Epochs slider — both modes */}
          <div>
            <div className="flex justify-between mb-1">
              <span className="text-[10px]" style={{ color: "var(--text-secondary)" }}>Epochs</span>
              <span className="text-[10px] font-mono" style={{ color: "var(--text-code)" }}>{epochs}</span>
            </div>
            <input
              type="range"
              min={5}
              max={50}
              step={1}
              value={epochs}
              disabled={disabled}
              onChange={(e) => onEpochsChange(Number(e.target.value))}
              className="w-full accent-[var(--accent)]"
              style={{ cursor: disabled ? "not-allowed" : "pointer" }}
            />
            <div className="flex justify-between text-[9px] mt-0.5" style={{ color: "var(--text-muted)" }}>
              <span>5 (quick)</span>
              <span>50 (thorough)</span>
            </div>
          </div>

        </div>
      )}
    </div>
  );
}
