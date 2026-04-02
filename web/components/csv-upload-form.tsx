"use client";

import { useState } from "react";

const TIMEFRAME_OPTIONS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];

const COL_FIELDS: { key: string; label: string; required: boolean; aliases: string[] }[] = [
  { key: "close",    label: "Close",    required: true,  aliases: ["close", "adj close", "adjusted close", "price"] },
  { key: "open",     label: "Open",     required: false, aliases: ["open"] },
  { key: "high",     label: "High",     required: false, aliases: ["high"] },
  { key: "low",      label: "Low",      required: false, aliases: ["low"] },
  { key: "volume",   label: "Volume",   required: false, aliases: ["volume", "vol"] },
  { key: "datetime", label: "Datetime", required: false, aliases: ["datetime", "date", "time", "timestamp"] },
];

export interface ColMap {
  close?: string;
  open?: string;
  high?: string;
  low?: string;
  volume?: string;
  datetime?: string;
}

function parseCSVHeaders(file: File): Promise<string[]> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = (e.target?.result as string) ?? "";
      const firstLine = text.split(/\r?\n/)[0] ?? "";
      // Handle quoted headers
      const headers = firstLine.split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
      resolve(headers.filter(Boolean));
    };
    reader.readAsText(file);
  });
}

function autoDetect(headers: string[]): ColMap {
  const lowerHeaders = headers.map((h) => h.toLowerCase());
  const map: ColMap = {};
  for (const field of COL_FIELDS) {
    for (const alias of field.aliases) {
      const idx = lowerHeaders.indexOf(alias);
      if (idx >= 0) {
        (map as any)[field.key] = headers[idx];
        break;
      }
    }
  }
  return map;
}

interface Props {
  uploading: boolean;
  onUpload: (file: File, symbol: string, timeframe: string, colMap: ColMap) => Promise<void>;
}

export function CsvUploadForm({ uploading, onUpload }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [colMap, setColMap] = useState<ColMap>({});
  const [symbol, setSymbol] = useState("");
  const [timeframe, setTimeframe] = useState("H1");

  async function handleFileChange(f: File | null) {
    setFile(f);
    if (!f) { setColumns([]); setColMap({}); return; }
    const headers = await parseCSVHeaders(f);
    setColumns(headers);
    setColMap(autoDetect(headers));
  }

  async function handleSubmit() {
    if (!file) return;
    await onUpload(file, symbol, timeframe, colMap);
    setFile(null);
    setColumns([]);
    setColMap({});
  }

  const missingClose = columns.length > 0 && !colMap.close;

  return (
    <div className="space-y-4">
      {/* File picker */}
      <div className="space-y-1">
        <label className="text-xs text-gray-400 uppercase">CSV File</label>
        <input
          type="file"
          accept=".csv"
          className="md-input w-full cursor-pointer file:mr-3 file:rounded file:border-0 file:bg-gray-700 file:px-3 file:py-1 file:text-xs file:text-white hover:file:bg-gray-600"
          onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)}
        />
      </div>

      {/* Column mapping — shown after headers are parsed */}
      {columns.length > 0 && (
        <div className="rounded border border-gray-700 bg-gray-800/50 p-3 space-y-3">
          <p className="text-xs font-medium text-gray-400 uppercase">Column Mapping</p>
          <div className="grid grid-cols-2 gap-3">
            {COL_FIELDS.map((f) => (
              <div key={f.key} className="space-y-1">
                <label className="text-xs text-gray-400 uppercase">
                  {f.label}{f.required && <span className="text-red-400 ml-0.5">*</span>}
                </label>
                <select
                  className="md-input w-full"
                  value={(colMap as any)[f.key] ?? ""}
                  onChange={(e) => setColMap((prev) => ({ ...prev, [f.key]: e.target.value || undefined }))}
                >
                  {!f.required && <option value="">(none)</option>}
                  {f.required && <option value="">-- select --</option>}
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            ))}
          </div>
          {missingClose && (
            <p className="text-xs text-red-400">Please select a column for Close.</p>
          )}
        </div>
      )}

      {/* Symbol / timeframe */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1">
          <label className="text-xs text-gray-400 uppercase">Symbol</label>
          <input
            className="md-input w-full"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            placeholder="USDJPY"
          />
        </div>
        <div className="space-y-1">
          <label className="text-xs text-gray-400 uppercase">Timeframe</label>
          <select className="md-input w-full" value={timeframe} onChange={(e) => setTimeframe(e.target.value)}>
            {TIMEFRAME_OPTIONS.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
          </select>
        </div>
      </div>

      <button
        type="button"
        disabled={uploading || !file || missingClose}
        onClick={handleSubmit}
        className="rounded bg-brand-500 px-4 py-2 text-xs text-white hover:bg-sky-400 disabled:opacity-40"
      >
        {uploading ? "Uploading…" : "Upload"}
      </button>
    </div>
  );
}
