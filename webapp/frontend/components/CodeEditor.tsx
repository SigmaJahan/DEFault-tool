"use client";

import { useRef } from "react";
import dynamic from "next/dynamic";
import { Upload, FileCode } from "lucide-react";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), { ssr: false });

interface CodeEditorProps {
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
}

export function CodeEditor({ value, onChange, disabled = false }: CodeEditorProps) {
  const fileRef = useRef<HTMLInputElement>(null);

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      onChange(text);
    };
    reader.readAsText(file);
    e.target.value = "";
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div
        className="flex items-center justify-between px-3 py-2 border-b"
        style={{ borderColor: "var(--border)", background: "var(--bg-overlay)" }}
      >
        <div className="flex items-center gap-2">
          <FileCode size={14} style={{ color: "var(--text-code)" }} />
          <span className="text-xs font-medium" style={{ color: "var(--text-secondary)", fontFamily: "var(--font-mono)" }}>
            model.py
          </span>
        </div>
        <button
          onClick={() => fileRef.current?.click()}
          disabled={disabled}
          className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md transition-colors"
          style={{
            color: "var(--accent)",
            border: "1px solid var(--border)",
            background: "transparent",
            cursor: disabled ? "not-allowed" : "pointer",
            opacity: disabled ? 0.5 : 1,
          }}
        >
          <Upload size={12} />
          Upload .py
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".py"
          className="hidden"
          onChange={handleFile}
        />
      </div>

      {/* Editor */}
      <div className="flex-1 overflow-hidden">
        <MonacoEditor
          language="python"
          theme="vs-dark"
          value={value}
          onChange={(v) => onChange(v ?? "")}
          options={{
            fontSize: 13,
            fontFamily: "JetBrains Mono, Fira Code, monospace",
            fontLigatures: true,
            minimap: { enabled: false },
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            renderLineHighlight: "line",
            wordWrap: "on",
            padding: { top: 12, bottom: 12 },
            readOnly: disabled,
            smoothScrolling: true,
            cursorBlinking: "smooth",
            bracketPairColorization: { enabled: true },
            "semanticHighlighting.enabled": true,
          }}
        />
      </div>
    </div>
  );
}
