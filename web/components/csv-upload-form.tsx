"use client";

import { useState } from "react";

const TIMEFRAME_OPTIONS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];

const OHLC_FIELDS: { key: string; label: string; required: boolean; aliases: string[] }[] = [
  { key: "close",    label: "Close",    required: true,  aliases: ["close", "adj close", "adjusted close"] },
  { key: "open",     label: "Open",     required: false, aliases: ["open"] },
  { key: "high",     label: "High",     required: false, aliases: ["high"] },
  { key: "low",      label: "Low",      required: false, aliases: ["low"] },
  { key: "volume",   label: "Volume",   required: false, aliases: ["volume", "vol"] },
  { key: "spread",   label: "Spread",   required: false, aliases: ["spread"] },
  { key: "datetime", label: "Datetime", required: false, aliases: ["datetime", "date", "time", "timestamp"] },
];

const TICK_ALIASES = ["price", "bid", "ask", "last", "mid", "tick"];

export interface ColMap {
  close?: string;
  open?: string;
  high?: string;
  low?: string;
  volume?: string;
  spread?: string;
  datetime?: string;
}

export interface UploadOptions {
  merge: boolean;
}

function parseCSVHeaders(file: File): Promise<string[]> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = (e.target?.result as string) ?? "";
      const firstLine = text.split(/\r?\n/)[0] ?? "";
      const headers = firstLine.split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
      resolve(headers.filter(Boolean));
    };
    reader.readAsText(file);
  });
}

function autoDetect(headers: string[]): ColMap {
  const lower = headers.map((h) => h.toLowerCase());
  const map: ColMap = {};
  for (const field of OHLC_FIELDS) {
    for (const alias of field.aliases) {
      const idx = lower.indexOf(alias);
      if (idx >= 0) { (map as any)[field.key] = headers[idx]; break; }
    }
  }
  return map;
}

function detectTickColumn(headers: string[]): string | null {
  const lower = headers.map((h) => h.toLowerCase());
  for (const alias of TICK_ALIASES) {
    const idx = lower.indexOf(alias);
    if (idx >= 0) return headers[idx];
  }
  return null;
}

interface Props {
  uploading: boolean;
  onUpload: (files: File[], symbol: string, timeframe: string, colMap: ColMap, options: UploadOptions) => Promise<void>;
}

export function CsvUploadForm({ uploading, onUpload }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [columns, setColumns] = useState<string[]>([]);
  const [colMap, setColMap] = useState<ColMap>({});
  const [isTick, setIsTick] = useState(false);
  const [symbol, setSymbol] = useState("");
  const [timeframe, setTimeframe] = useState("H1");
  const [merge, setMerge] = useState(true);

  async function handleFileChange(selected: FileList | null) {
    const arr = selected ? Array.from(selected) : [];
    setFiles(arr);
    setColumns([]);
    setColMap({});
    setIsTick(false);
    if (arr.length === 0) return;

    // Try to read headers from the first non-ZIP file for column mapping
    const firstCsv = arr.find((f) => !f.name.toLowerCase().endsWith(".zip"));
    if (!firstCsv) return; // all ZIPs — skip header parsing

    const headers = await parseCSVHeaders(firstCsv);
    setColumns(headers);
    const detected = autoDetect(headers);
    setColMap(detected);
    if (!detected.close) setIsTick(detectTickColumn(headers) !== null);
  }

  async function handleSubmit() {
    if (files.length === 0) return;
    await onUpload(files, symbol, timeframe, colMap, { merge });
    setFiles([]);
    setColumns([]);
    setColMap({});
    setIsTick(false);
  }

  const allZips = files.length > 0 && files.every((f) => f.name.toLowerCase().endsWith(".zip"));
  const multiFile = files.length > 1;
  const missingClose = !allZips && !isTick && columns.length > 0 && !colMap.close;
  const canSubmit = files.length > 0 && !missingClose && !uploading;

  // Summary label
  const fileSummary = files.length === 0 ? null
    : files.length === 1 ? files[0].name
    : `${files.length} files selected`;

  return (
    <div className="space-y-4">
      {/* File picker */}
      <div className="space-y-1">
        <label className="text-xs text-gray-400 uppercase">File(s)</label>
        <input
          type="file"
          accept=".csv,.zip"
          multiple
          className="md-input w-full cursor-pointer file:mr-3 file:rounded file:border-0 file:bg-gray-700 file:px-3 file:py-1 file:text-xs file:text-white hover:file:bg-gray-600"
          onChange={(e) => handleFileChange(e.target.files)}
        />
        <p className="text-xs text-gray-500">
          Accepts CSV or ZIP files. Select multiple to merge them together. OHLC and tick data supported.
        </p>
        {fileSummary && <p className="text-xs text-gray-300">{fileSummary}</p>}
      </div>

      {/* Merge toggle — shown when more than one file, or a single ZIP */}
      {(multiFile || allZips) && (
        <div className="rounded border border-gray-700 bg-gray-800/50 p-3 space-y-2">
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              className="accent-brand-500"
              checked={merge}
              onChange={(e) => setMerge(e.target.checked)}
            />
            <span className="text-xs text-gray-300">
              Merge all into one dataset
              <span className="ml-1 text-gray-500">(uncheck to create one dataset per file)</span>
            </span>
          </label>
        </div>
      )}

      {/* Tick mode notice */}
      {isTick && (
        <div className="rounded border border-yellow-700/50 bg-yellow-900/20 p-3">
          <p className="text-xs text-yellow-300 font-medium">Tick data detected</p>
          <p className="text-xs text-yellow-500 mt-0.5">
            No OHLC columns found. The price column will be stored as tick data and resampled to OHLC on preview.
          </p>
        </div>
      )}

      {/* OHLC column mapping — shown after CSV headers are parsed, non-tick only */}
      {columns.length > 0 && !isTick && (
        <div className="rounded border border-gray-700 bg-gray-800/50 p-3 space-y-3">
          <p className="text-xs font-medium text-gray-400 uppercase">
            Column Mapping
            {multiFile && <span className="ml-1 normal-case text-gray-500">(applied to all files)</span>}
          </p>
          <div className="grid grid-cols-2 gap-3">
            {OHLC_FIELDS.map((f) => (
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
          {missingClose && <p className="text-xs text-red-400">Please select a column for Close.</p>}
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
        disabled={!canSubmit}
        onClick={handleSubmit}
        className="rounded bg-brand-500 px-4 py-2 text-xs text-white hover:bg-sky-400 disabled:opacity-40"
      >
        {uploading ? "Uploading…" : "Upload"}
      </button>
    </div>
  );
}
