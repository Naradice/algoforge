"use client";

import { useState } from "react";
import useSWR, { mutate } from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";
import { useParams } from "next/navigation";
import { useToast } from "@/lib/toast";
import { CsvUploadForm, type ColMap, type UploadOptions } from "@/components/csv-upload-form";

type TestResult = {
  success: boolean;
  status_code?: number;
  content_type?: string;
  size_bytes?: number;
  title?: string;
  links?: { href: string; text: string; filename: string; matches_ext: boolean }[];
  error?: string;
};

// ── Config field definitions (shared with new-datasource page) ────────────────

const TIMEFRAME_OPTIONS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];

interface FieldDef {
  key: string;
  label: string;
  type: "text" | "number" | "select" | "date";
  options?: string[];
  optionDescriptions?: Record<string, string>;
  placeholder?: string;
  hint?: string;
}

const TYPE_FIELD_DEFS: Record<string, FieldDef[]> = {
  ohlc_download: [
    { key: "client", label: "Provider", type: "select", options: ["yfinance", "vantage"] },
    {
      key: "symbol", label: "Symbol", type: "text", placeholder: "USDJPY=X",
      hint: "yfinance: USDJPY=X, EURUSD=X. Alpha Vantage: USD/JPY",
    },
    { key: "timeframe", label: "Timeframe", type: "select", options: TIMEFRAME_OPTIONS },
    {
      key: "from_ts", label: "From Date", type: "date",
      hint: "yfinance H1/M data is limited to ~730 days lookback",
    },
    { key: "to_ts", label: "To Date", type: "date", hint: "Leave blank for today" },
  ],
  ddm_simulation: [
    { key: "timeframe", label: "Timeframe", type: "select", options: TIMEFRAME_OPTIONS },
    { key: "initial_price", label: "Initial Price", type: "number", placeholder: "100.0" },
    { key: "num_agent", label: "Number of Agents", type: "number", placeholder: "50" },
    { key: "seed", label: "Random Seed", type: "number", placeholder: "42" },
  ],
  web_report: [
    {
      key: "url", label: "URL", type: "text",
      placeholder: "https://www.example.com/reports",
      hint: "Landing page containing report links",
    },
    {
      key: "ext", label: "File Type", type: "select",
      options: ["pdf", "html", "mp3", "txt"],
    },
    {
      key: "subfolder", label: "Subfolder", type: "text",
      placeholder: "mizuho",
      hint: "Output directory name (under artifacts/web_reports/)",
    },
    {
      key: "filename", label: "Filename Template", type: "text",
      placeholder: "{YYYYMMDD}.pdf",
      hint: "Placeholders: {YYYYMMDD} {YYMMDD} {YYYYMM} {YYMM} {filename} {basefilename}",
    },
    {
      key: "type", label: "Fetch Method", type: "select",
      options: ["load", "goto_load", "goto_download", "load_rep"],
      optionDescriptions: {
        load:          "Direct HTTP download (httpx). Fast and simple — use for public links with no bot protection. Will get 403 on Akamai/CDN-protected sites.",
        goto_load:     "Opens the URL in a real browser and saves the rendered page as a PDF. Use when the target is an HTML page you want to archive as PDF, not a file download.",
        goto_download: "Uses the browser's fetch() to download the file in-page, carrying real browser headers and cookies. Required for Akamai/CDN-protected PDFs — most Japanese broker sites (Mizuho, Sony Finance, MUFG, etc.) need this.",
        load_rep:      "Plain HTTP download saved as HTML source. Use to archive a page's raw HTML. Fast, but shares the same CDN limitations as load — won't work on bot-protected sites.",
      },
    },
    {
      key: "unique", label: "Deduplication", type: "select",
      options: ["segment", "checksum", "text"],
      hint: "segment: skip if file exists. checksum: skip if content unchanged. text: skip if link text unchanged.",
    },
    {
      key: "interval_days", label: "Interval (days)", type: "number",
      placeholder: "1",
      hint: "Minimum days between downloads. Leave blank to always run.",
    },
  ],
  manual_upload: [],
};

function configToValues(config: Record<string, unknown>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(config)) {
    out[k] = String(v ?? "");
  }
  return out;
}

function valuesToConfig(type: string, values: Record<string, string>): Record<string, unknown> {
  const fields = TYPE_FIELD_DEFS[type] ?? [];
  const cfg: Record<string, unknown> = {};
  for (const f of fields) {
    const raw = values[f.key] ?? "";
    if (raw === "") continue;
    if (f.type === "number") {
      const n = Number(raw);
      if (!isNaN(n)) cfg[f.key] = n;
    } else {
      cfg[f.key] = raw;
    }
  }
  return cfg;
}

// ── Page component ────────────────────────────────────────────────────────────

export default function DatasourceDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { toast } = useToast();

  const { data: ds } = useSWR(`/api/v1/datasources/${id}`, fetcher);
  const { data: jobs, isLoading } = useSWR(
    ds?.type !== "manual_upload" ? `/api/v1/collection-jobs?datasource_id=${id}` : null,
    fetcher,
    { refreshInterval: (data) => (data?.some?.((j: any) => j.status === "running") || Date.now() < pollUntil ? 2000 : 0) }
  );
  const { data: webReportFiles, mutate: mutateFiles } = useSWR(
    ds?.type === "web_report" ? `/api/v1/datasources/${id}/web-report/files` : null,
    fetcher,
    { refreshInterval: (data) => (jobs?.some?.((j: any) => j.status === "running") ? 5000 : 0) }
  );
  const { data: uploadedDatasets, mutate: mutateDatasets } = useSWR(
    ds?.type === "manual_upload" ? `/api/v1/datasets?datasource_id=${id}` : null,
    fetcher,
  );

  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editValues, setEditValues] = useState<Record<string, string>>({});
  const [webReportCustom, setWebReportCustom] = useState("");
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [pollUntil, setPollUntil] = useState(0);
  const [runForever, setRunForever] = useState(false);
  const [lengthValue, setLengthValue] = useState("1000");
  const [uploading, setUploading] = useState(false);

  // Test-fetch state (web_report only)
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [testLoading, setTestLoading] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const [testShowLinks, setTestShowLinks] = useState(false);

  async function handleTestFetch() {
    const cfg = ds?.config ?? {};
    const url = cfg.url;
    const fetch_type = cfg.type ?? "load";
    const ext = cfg.ext ?? undefined;
    if (!url) return;
    setTestLoading(true);
    setTestError(null);
    setTestResult(null);
    setTestShowLinks(false);
    try {
      const res = await fetch("/api/v1/datasources/web-report/test-fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, fetch_type, ext }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        setTestError(body.error?.message ?? body.detail?.message ?? "Test failed");
        return;
      }
      setTestResult(body.data ?? null);
    } finally {
      setTestLoading(false);
    }
  }

  async function applyLinkParseConfig(matchCount: number) {
    const cfg = ds?.config ?? {};
    const ext = cfg.ext ?? "pdf";
    const linkFetchType = cfg.type === "goto_load" ? "goto_download" : (cfg.type ?? "load");
    const steps = [{ type: "link_parse", targets: [{ value: `.*\\.${ext}`, ext, filename: `{YYYYMMDD}_{filename}`, type: linkFetchType, unique: "text", interval_days: cfg.interval_days ?? 1 }] }];
    const newConfig = { ...cfg, custom: steps };
    const res = await fetch(`/api/v1/datasources/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: newConfig }),
    });
    if (res.ok) {
      toast(`Configured to download all ${matchCount} files`, "success");
      mutate(`/api/v1/datasources/${id}`);
      setTestResult(null);
    } else {
      const body = await res.json().catch(() => ({}));
      toast(body.error?.message ?? "Failed to update", "error");
    }
  }

  function startEdit() {
    setEditName(ds?.name ?? "");
    setEditValues(configToValues(ds?.config ?? {}));
    const storedLength = ds?.config?.length;
    setRunForever(storedLength == null);
    setLengthValue(storedLength != null ? String(storedLength) : "1000");
    setWebReportCustom(
      ds?.type === "web_report" && ds?.config?.custom != null
        ? JSON.stringify(ds.config.custom, null, 2)
        : ""
    );
    setEditing(true);
  }

  function cancelEdit() {
    setEditing(false);
  }

  async function saveEdit() {
    if (ds?.type === "web_report" && webReportCustom.trim()) {
      try { JSON.parse(webReportCustom); } catch { toast("Custom Steps JSON is invalid", "error"); return; }
    }
    setSaving(true);
    try {
      const baseConfig = valuesToConfig(ds?.type ?? "", editValues);
      if (ds?.type === "web_report" && webReportCustom.trim()) {
        baseConfig["custom"] = JSON.parse(webReportCustom);
      }
      if (ds?.type === "ddm_simulation") {
        if (!runForever) {
          const n = Number(lengthValue);
          if (!isNaN(n) && n > 0) baseConfig["length"] = n;
        }
        // runForever → no length key → backend runs endlessly
      }
      const res = await fetch(`/api/v1/datasources/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: editName.trim(), config: baseConfig }),
      });
      if (res.ok) {
        toast("Datasource saved", "success");
        mutate(`/api/v1/datasources/${id}`);
        setEditing(false);
      } else {
        const body = await res.json().catch(() => ({}));
        toast(body.error?.message ?? "Failed to save", "error");
      }
    } finally {
      setSaving(false);
    }
  }

  async function deleteDatasource() {
    if (!confirm(`Delete datasource "${ds?.name}"? This cannot be undone.`)) return;
    setDeleting(true);
    const res = await fetch(`/api/v1/datasources/${id}`, { method: "DELETE" });
    if (res.ok) {
      window.location.href = "/data";
    } else {
      const body = await res.json().catch(() => ({}));
      toast(body.error?.message ?? "Failed to delete", "error");
      setDeleting(false);
    }
  }

  async function handleUpload(uploadFiles: File[], symbol: string, timeframe: string, colMap: ColMap, options: UploadOptions) {
    setUploading(true);
    try {
      const form = new FormData();
      uploadFiles.forEach((f) => form.append("files", f));
      form.append("datasource_id", id);
      if (symbol)          form.append("symbol",       symbol);
      if (timeframe)       form.append("timeframe",    timeframe);
      if (colMap.close)    form.append("close_col",    colMap.close);
      if (colMap.open)     form.append("open_col",     colMap.open);
      if (colMap.high)     form.append("high_col",     colMap.high);
      if (colMap.low)      form.append("low_col",      colMap.low);
      if (colMap.volume)   form.append("volume_col",   colMap.volume);
      if (colMap.datetime) form.append("datetime_col", colMap.datetime);
      form.append("merge", String(options.merge));
      const res = await fetch("/api/v1/datasets/upload", { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        toast(body.error?.message ?? body.detail ?? "Upload failed", "error");
        return;
      }
      toast("File(s) uploaded successfully", "success");
      mutateDatasets();
    } finally {
      setUploading(false);
    }
  }

  function triggerPoll() {
    setPollUntil(Date.now() + 300_000); // poll for up to 5 minutes after queuing
  }

  function extractApiError(body: any, fallback: string): string {
    return body?.detail?.message ?? body?.error?.message ?? body?.detail ?? fallback;
  }

  // Refresh the files list immediately, then again after 5 s and 15 s so fast
  // web_report jobs (which complete before the next status poll) are visible.
  function scheduleFilesRefresh() {
    setTimeout(() => mutateFiles?.(), 5_000);
    setTimeout(() => mutateFiles?.(), 15_000);
  }

  async function startCollection() {
    const res = await fetch(`/api/v1/datasources/${id}/collect`, { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (res.ok) {
      toast("Collection queued", "success");
      triggerPoll();
      scheduleFilesRefresh();
    } else {
      toast(extractApiError(body, "Failed to start collection"), "error");
    }
    mutate(`/api/v1/collection-jobs?datasource_id=${id}`);
    mutateFiles?.();
  }

  async function runJob(jobId: number) {
    const res = await fetch(`/api/v1/collection-jobs/${jobId}/run`, { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (res.ok) {
      toast("Collection queued", "success");
      triggerPoll();
      scheduleFilesRefresh();
    } else {
      toast(extractApiError(body, "Failed to queue job"), "error");
    }
    mutate(`/api/v1/collection-jobs?datasource_id=${id}`);
    mutateFiles?.();
  }

  async function resetJob(jobId: number) {
    const res = await fetch(`/api/v1/collection-jobs/${jobId}/reset`, { method: "POST" });
    const body = await res.json().catch(() => ({}));
    if (res.ok) {
      toast("Job reset — you can now run it again", "success");
    } else {
      toast(extractApiError(body, "Failed to reset job"), "error");
    }
    mutate(`/api/v1/collection-jobs?datasource_id=${id}`);
  }

  const hasJobs = jobs && jobs.length > 0;
  const fields = TYPE_FIELD_DEFS[ds?.type ?? ""] ?? [];

  return (
    <div className="space-y-6 max-w-3xl">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <a href="/data" className="text-xs text-gray-500 hover:text-white">← Data</a>
          <h1 className="mt-1 text-2xl font-semibold text-white">{ds?.name ?? "…"}</h1>
          <p className="text-xs text-gray-400">{ds?.type}</p>
        </div>
        {ds && !editing && (
          <div className="flex gap-2">
            <button
              onClick={startEdit}
              className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:border-gray-400 hover:text-white"
            >
              Edit
            </button>
            <button
              onClick={deleteDatasource}
              disabled={deleting}
              className="rounded border border-red-800 px-3 py-1.5 text-xs text-red-400 hover:bg-red-900/20 disabled:opacity-50"
            >
              {deleting ? "Deleting…" : "Delete"}
            </button>
          </div>
        )}
      </div>

      {/* Config — view mode */}
      {ds && !editing && (
        <section className="rounded border border-gray-800 bg-gray-900 p-4 space-y-3">
          <h2 className="text-xs font-medium text-gray-400 uppercase">Config</h2>
          {fields.length > 0 || ds.type === "ddm_simulation" ? (
            <div className="grid grid-cols-2 gap-2">
              {ds.type === "ddm_simulation" && (
                <div>
                  <p className="text-xs text-gray-500 uppercase">Output Candles</p>
                  <p className="text-sm text-white font-mono">
                    {ds.config.length != null ? ds.config.length : "∞ (run forever)"}
                  </p>
                </div>
              )}
              {fields.map((f) => {
                const val = ds.config[f.key];
                if (val == null) return null;
                return (
                  <div key={f.key}>
                    <p className="text-xs text-gray-500 uppercase">{f.label}</p>
                    <p className="text-sm text-white font-mono">{String(val)}</p>
                  </div>
                );
              })}
              {ds.type === "web_report" && ds.config.custom != null && (
                <div className="col-span-2">
                  <p className="text-xs text-gray-500 uppercase">Link following</p>
                  <p className="text-sm text-green-400">✓ Configured to follow and download matching links</p>
                </div>
              )}
            </div>
          ) : (
            <pre className="text-xs text-gray-300 overflow-x-auto">{JSON.stringify(ds.config, null, 2)}</pre>
          )}
        </section>
      )}

      {/* Config — edit mode */}
      {ds && editing && (
        <section className="rounded border border-brand-500/40 bg-gray-900 p-4 space-y-4">
          <h2 className="text-xs font-medium text-gray-400 uppercase">Edit Datasource</h2>
          <div className="space-y-1">
            <label className="text-xs text-gray-400 uppercase">Name</label>
            <input
              className="md-input w-full"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
            />
          </div>
          {fields.map((f) => (
            <div key={f.key} className="space-y-1">
              <label className="text-xs text-gray-400 uppercase">{f.label}</label>
              {f.type === "select" ? (
                <>
                  <select
                    className="md-input w-full"
                    value={editValues[f.key] ?? ""}
                    onChange={(e) => setEditValues((prev) => ({ ...prev, [f.key]: e.target.value }))}
                  >
                    {f.options?.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                  {f.optionDescriptions?.[editValues[f.key]] && (
                    <p className="text-xs text-gray-400 bg-gray-800/60 rounded px-2 py-1.5 mt-1">
                      {f.optionDescriptions[editValues[f.key]]}
                    </p>
                  )}
                </>
              ) : f.type === "date" ? (
                <input
                  type="date"
                  className="md-input w-full"
                  value={editValues[f.key] ?? ""}
                  onChange={(e) => setEditValues((prev) => ({ ...prev, [f.key]: e.target.value }))}
                />
              ) : (
                <input
                  type={f.type === "number" ? "number" : "text"}
                  step={f.type === "number" ? "any" : undefined}
                  className="md-input w-full"
                  value={editValues[f.key] ?? ""}
                  onChange={(e) => setEditValues((prev) => ({ ...prev, [f.key]: e.target.value }))}
                  placeholder={f.placeholder}
                />
              )}
              {f.hint && <p className="text-xs text-gray-500">{f.hint}</p>}
            </div>
          ))}
          {ds?.type === "ddm_simulation" && (
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={runForever}
                  onChange={(e) => setRunForever(e.target.checked)}
                  className="w-4 h-4 accent-brand-500"
                />
                <span className="text-sm text-white">Run forever</span>
              </label>
              {!runForever && (
                <div className="space-y-1">
                  <label className="text-xs text-gray-400 uppercase">Output Candles</label>
                  <input
                    type="number"
                    className="md-input w-full"
                    value={lengthValue}
                    onChange={(e) => setLengthValue(e.target.value)}
                    placeholder="1000"
                  />
                </div>
              )}
            </div>
          )}
          {fields.length === 0 && ds?.type !== "ddm_simulation" && ds?.type !== "web_report" && (
            <p className="text-xs text-gray-500">No configurable fields for this datasource type.</p>
          )}
          <div className="flex gap-2 pt-2">
            <button
              onClick={saveEdit}
              disabled={saving}
              className="rounded bg-brand-500 px-3 py-1.5 text-xs text-white hover:bg-sky-400 disabled:opacity-40"
            >
              {saving ? "Saving…" : "Save"}
            </button>
            <button onClick={cancelEdit} className="rounded bg-gray-700 px-3 py-1.5 text-xs text-white hover:bg-gray-600">
              Cancel
            </button>
          </div>
        </section>
      )}

      {/* Test Fetch (web_report, view mode) */}
      {ds?.type === "web_report" && !editing && (
        <section className="rounded border border-gray-800 bg-gray-900 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xs font-medium text-gray-400 uppercase">Test Fetch Method</h2>
              <p className="text-xs text-gray-600 mt-0.5">Verify the configured fetch method can access the URL</p>
            </div>
            <button
              onClick={handleTestFetch}
              disabled={testLoading || !ds?.config?.url}
              className="rounded border border-gray-700 px-3 py-1 text-xs text-gray-300 hover:border-gray-400 hover:text-white disabled:opacity-40 shrink-0 ml-3"
            >
              {testLoading ? "Testing…" : "Test"}
            </button>
          </div>
          {testError && <p className="text-xs text-red-400">{testError}</p>}
          {testResult && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className={`inline-flex items-center gap-1 rounded px-2 py-0.5 font-semibold ${testResult.success ? "bg-green-900/50 text-green-300" : "bg-red-900/50 text-red-300"}`}>
                  {testResult.success ? "✓ Success" : "✗ Failed"}
                </span>
                {testResult.status_code !== undefined && <span className="text-gray-400">HTTP {testResult.status_code}</span>}
                {testResult.content_type && <span className="text-gray-400">{testResult.content_type.split(";")[0]}</span>}
                {testResult.size_bytes !== undefined && (
                  <span className="text-gray-400">
                    {testResult.size_bytes >= 1048576
                      ? `${(testResult.size_bytes / 1048576).toFixed(1)} MB`
                      : testResult.size_bytes >= 1024
                      ? `${(testResult.size_bytes / 1024).toFixed(1)} KB`
                      : `${testResult.size_bytes} B`}
                  </span>
                )}
                {testResult.title && <span className="text-gray-400 truncate max-w-[240px]" title={testResult.title}>&ldquo;{testResult.title}&rdquo;</span>}
              </div>
              {testResult.error && <p className="text-xs text-red-400">{testResult.error}</p>}
              {testResult.links && testResult.links.length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-gray-400">
                      <span className={testResult.links.filter(l => l.matches_ext).length === 0 ? "text-amber-400" : "text-green-400"}>
                        {testResult.links.filter(l => l.matches_ext).length} matching .{ds?.config?.ext}
                      </span>
                      {" · "}{testResult.links.length} total links
                    </p>
                    {testResult.links.filter(l => l.matches_ext).length > 0 && !ds?.config?.custom && (
                      <button
                        onClick={() => applyLinkParseConfig(testResult!.links!.filter(l => l.matches_ext).length)}
                        className="rounded bg-brand-500 px-2 py-1 text-xs text-white hover:bg-sky-400 shrink-0 ml-2"
                      >
                        Configure to download all {testResult.links.filter(l => l.matches_ext).length} files
                      </button>
                    )}
                    {ds?.config?.custom && testResult.links.filter(l => l.matches_ext).length > 0 && (
                      <span className="text-xs text-green-400 shrink-0 ml-2">✓ Already configured</span>
                    )}
                  </div>
                  <div className="rounded border border-gray-700 overflow-hidden text-xs">
                    <table className="w-full">
                      <thead>
                        <tr className="bg-gray-800/60 text-left">
                          <th className="px-3 py-1.5 text-gray-400 font-medium">Filename / URL</th>
                          <th className="px-3 py-1.5 text-gray-400 font-medium">Link text</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(testShowLinks
                          ? testResult.links
                          : testResult.links.filter(l => l.matches_ext).length > 0
                            ? testResult.links.filter(l => l.matches_ext)
                            : testResult.links
                        ).slice(0, testShowLinks ? 300 : 20).map((link, i) => (
                          <tr key={i} className={`border-t border-gray-700/50 hover:bg-gray-800/20 ${link.matches_ext ? "" : "opacity-50"}`}>
                            <td className="px-3 py-1.5 font-mono">
                              <a href={link.href} target="_blank" rel="noopener noreferrer"
                                className="text-brand-400 hover:underline truncate block max-w-[220px]" title={link.href}>
                                {link.filename || link.href}
                              </a>
                            </td>
                            <td className="px-3 py-1.5 text-gray-300 truncate max-w-[160px]">{link.text || "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <div className="px-3 py-1.5 border-t border-gray-700/50">
                      {!testShowLinks && testResult.links.length > 20 && (
                        <button type="button" onClick={() => setTestShowLinks(true)} className="text-xs text-gray-500 hover:text-white">
                          Show all {testResult.links.length} links
                        </button>
                      )}
                      {testShowLinks && (
                        <button type="button" onClick={() => setTestShowLinks(false)} className="text-xs text-gray-500 hover:text-white">
                          Collapse
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {/* Manual Upload section */}
      {ds?.type === "manual_upload" && (
        <section className="space-y-4">
          <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide">Upload CSV</h2>
          <div className="rounded border border-gray-800 bg-gray-900 p-4">
            <CsvUploadForm uploading={uploading} onUpload={handleUpload} />
          </div>

          {uploadedDatasets && uploadedDatasets.length > 0 && (
            <div className="rounded border border-gray-800 overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-left">
                    <th className="px-4 py-2 text-xs text-gray-400 font-medium">Name</th>
                    <th className="px-4 py-2 text-xs text-gray-400 font-medium">Symbol</th>
                    <th className="px-4 py-2 text-xs text-gray-400 font-medium">Timeframe</th>
                    <th className="px-4 py-2 text-xs text-gray-400 font-medium">Rows</th>
                  </tr>
                </thead>
                <tbody>
                  {uploadedDatasets.map((d: any) => (
                    <tr key={d.id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                      <td className="px-4 py-2">
                        <a href={`/data/datasets/${d.id}`} className="text-brand-400 hover:text-brand-300 hover:underline">{d.name}</a>
                      </td>
                      <td className="px-4 py-2 text-gray-300 font-mono text-xs">{d.symbol ?? "—"}</td>
                      <td className="px-4 py-2">{d.timeframe ? <span className="md-chip">{d.timeframe}</span> : <span className="text-gray-500">—</span>}</td>
                      <td className="px-4 py-2 tabular-nums text-gray-200">{d.row_count?.toLocaleString() ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {uploadedDatasets && uploadedDatasets.length === 0 && (
            <p className="text-sm text-gray-500">No datasets yet. Upload a CSV file above.</p>
          )}
        </section>
      )}

      {/* Downloaded Files (web_report only) */}
      {ds?.type === "web_report" && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide">Downloaded Files</h2>
            <button
              onClick={() => mutateFiles?.()}
              className="rounded border border-gray-700 px-2 py-1 text-xs text-gray-400 hover:text-white"
            >
              Refresh
            </button>
          </div>

          {!webReportFiles && (
            <p className="text-sm text-gray-500">No files downloaded yet. Run a collection job to download reports.</p>
          )}
          {webReportFiles && webReportFiles.length === 0 && (
            <p className="text-sm text-gray-500">No files downloaded yet.</p>
          )}
          {webReportFiles && webReportFiles.length > 0 && (
            <div className="rounded border border-gray-800 overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-left">
                    <th className="px-4 py-2 text-xs text-gray-400 font-medium">Filename</th>
                    <th className="px-4 py-2 text-xs text-gray-400 font-medium">Size</th>
                    <th className="px-4 py-2 text-xs text-gray-400 font-medium">Downloaded</th>
                    <th className="px-4 py-2 text-xs text-gray-400 font-medium"></th>
                  </tr>
                </thead>
                <tbody>
                  {webReportFiles.map((f: any) => (
                    <tr key={f.path} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                      <td className="px-4 py-2 font-mono text-xs text-white">{f.name}</td>
                      <td className="px-4 py-2 text-gray-400 text-xs tabular-nums">
                        {f.size_bytes >= 1024 * 1024
                          ? `${(f.size_bytes / 1024 / 1024).toFixed(1)} MB`
                          : f.size_bytes >= 1024
                          ? `${(f.size_bytes / 1024).toFixed(1)} KB`
                          : `${f.size_bytes} B`}
                      </td>
                      <td className="px-4 py-2 text-gray-400 text-xs">
                        {new Date(f.modified_at).toLocaleString()}
                      </td>
                      <td className="px-4 py-2 text-right">
                        <a
                          href={`/api/v1/datasources/${id}/web-report/files/${f.path}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-xs text-brand-400 hover:text-brand-300 hover:underline"
                        >
                          Open
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {/* Collection Jobs (non-manual types) */}
      {ds?.type !== "manual_upload" && (
        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide">Collection Jobs</h2>
            {!hasJobs && (
              <button
                onClick={startCollection}
                className="rounded bg-brand-500 px-2 py-1 text-xs text-white hover:bg-sky-400"
              >
                Start Collection
              </button>
            )}
          </div>

          {isLoading && <p className="text-gray-400 text-sm">Loading…</p>}

          {!isLoading && !hasJobs && (
            <p className="text-sm text-gray-500">No collection jobs yet. Click "Start Collection" to run now.</p>
          )}

          {jobs?.map((job: any) => (
            <div key={job.id} className="flex items-center justify-between rounded border border-gray-800 bg-gray-900 px-4 py-3">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <StatusBadge status={job.status} />
                  <span className="text-xs text-gray-400">{job.schedule_cron ?? "one-off"}</span>
                </div>
                {job.last_run_at && (
                  <p className="text-xs text-gray-500">Last run: {new Date(job.last_run_at).toLocaleString()}</p>
                )}
                {job.last_error && (
                  <p className="text-xs text-red-400 max-w-sm">{job.last_error}</p>
                )}
              </div>
              <div className="flex gap-2">
                {job.status === "running" && (
                  <button
                    onClick={() => resetJob(job.id)}
                    className="rounded border border-red-800 px-3 py-1.5 text-xs text-red-400 hover:bg-red-900/20"
                    title="Force-reset a stuck job after an ungraceful worker restart"
                  >
                    Reset
                  </button>
                )}
                <button
                  onClick={() => runJob(job.id)}
                  disabled={job.status === "running"}
                  className="rounded bg-gray-700 px-3 py-1.5 text-xs text-white hover:bg-gray-600 disabled:opacity-40"
                >
                  {job.status === "running" ? "Running…" : "Run now"}
                </button>
              </div>
            </div>
          ))}
        </section>
      )}

    </div>
  );
}
