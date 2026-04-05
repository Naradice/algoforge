"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { CsvUploadForm, type ColMap } from "@/components/csv-upload-form";

// ── Field definitions per datasource type ────────────────────────────────────

interface FieldDef {
  key: string;
  label: string;
  type: "text" | "number" | "select" | "date";
  options?: string[];
  optionDescriptions?: Record<string, string>; // shown below select when an option is chosen
  placeholder?: string;
  hint?: string;
}

const TIMEFRAME_OPTIONS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];

const TYPE_FIELD_DEFS: Record<string, FieldDef[]> = {
  ohlc_download: [
    {
      key: "client", label: "Provider", type: "select",
      options: ["yfinance", "vantage"],
      hint: "yfinance is free; vantage requires an API key",
    },
    {
      key: "symbol", label: "Symbol", type: "text",
      placeholder: "USDJPY=X",
      hint: "yfinance: USDJPY=X, EURUSD=X, AAPL. Alpha Vantage: USD/JPY",
    },
    { key: "timeframe", label: "Timeframe", type: "select", options: TIMEFRAME_OPTIONS },
    {
      key: "from_ts", label: "From Date", type: "date",
      hint: "yfinance H1/M data is limited to ~730 days lookback — older dates are auto-adjusted",
    },
    { key: "to_ts", label: "To Date", type: "date", placeholder: "", hint: "Leave blank for today" },
  ],
  ddm_simulation: [
    { key: "model", label: "Model Version", type: "select", options: ["v3", "v1"], hint: "V3 adds WMA trend-following feedback (original paper). V1 is the simpler base model — use to diagnose drift issues." },
    { key: "timeframe", label: "Timeframe", type: "select", options: TIMEFRAME_OPTIONS },
    { key: "initial_price", label: "Initial Price", type: "number", placeholder: "100.0" },
    { key: "spread", label: "Spread", type: "number", placeholder: "1.0", hint: "Bid-ask spread in price units" },
    { key: "num_agent", label: "Number of Agents", type: "number", placeholder: "300", hint: "More agents = lower volatility (original default: 300)" },
    { key: "max_volatility", label: "Max Volatility", type: "number", placeholder: "0.02", hint: "Upper bound of per-agent price tendency per step" },
    { key: "min_volatility", label: "Min Volatility", type: "number", placeholder: "0.01", hint: "Lower bound of per-agent price tendency per step" },
    { key: "trade_unit", label: "Trade Unit", type: "number", placeholder: "0.001", hint: "Minimum price increment (pips)" },
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

const TYPE_DEFAULTS: Record<string, Record<string, string>> = {
  ohlc_download: {
    client: "yfinance",
    symbol: "USDJPY=X",
    timeframe: "H1",
    from_ts: "2022-01-01",
    to_ts: "",
  },
  ddm_simulation: {
    model: "v3",
    timeframe: "M1",
    initial_price: "100",
    spread: "1",
    num_agent: "300",
    max_volatility: "0.02",
    min_volatility: "0.01",
    trade_unit: "0.001",
    seed: "42",
  },
  web_report: {
    url: "",
    ext: "pdf",
    subfolder: "",
    filename: "{YYYYMMDD}.pdf",
    type: "load",
    unique: "segment",
    interval_days: "1",
  },
  manual_upload: {},
};

const TYPE_DESCRIPTIONS: Record<string, { label: string; description: string; status?: string }> = {
  ohlc_download: {
    label: "OHLC Download",
    description: "Download historical candle data from yfinance (free) or Alpha Vantage.",
  },
  ddm_simulation: {
    label: "DDM Simulation",
    description:
      "Generate synthetic OHLC data using a Deterministic Dealer Model. Tick data is stored and resampled to your chosen timeframe on demand.",
  },
  web_report: {
    label: "Web Report",
    description: "Download financial reports (PDF, HTML, audio) from institution websites using Playwright. Mirrors cyclic_downloader source.json schema.",
  },
  manual_upload: {
    label: "Manual Upload",
    description:
      "Upload a CSV file from your computer as a dataset. The CSV must contain a 'close' column. Optionally include open, high, low, volume.",
  },
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function NewDatasourcePage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [type, setType] = useState("ohlc_download");
  const [values, setValues] = useState<Record<string, string>>(TYPE_DEFAULTS["ohlc_download"]);
  const [runForever, setRunForever] = useState(false);
  const [lengthValue, setLengthValue] = useState("1000");
  const [webReportCustom, setWebReportCustom] = useState("");
  const [testResult, setTestResult] = useState<{
    success: boolean;
    status_code?: number;
    content_type?: string;
    size_bytes?: number;
    title?: string;
    links?: { href: string; text: string; filename: string; matches_ext: boolean }[];
    error?: string;
  } | null>(null);
  const [testLoading, setTestLoading] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);
  const [testShowLinks, setTestShowLinks] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleTypeChange(t: string) {
    setType(t);
    setValues(TYPE_DEFAULTS[t] ?? {});
    setRunForever(false);
    setLengthValue("1000");
    setWebReportCustom("");
    setTestResult(null);
    setTestError(null);
  }

  async function handleTestFetch() {
    const url = values["url"];
    const fetch_type = values["type"] || "load";
    const ext = values["ext"] || undefined;
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

  function handleFieldChange(key: string, value: string) {
    setValues((prev) => ({ ...prev, [key]: value }));
    if (key === "url" || key === "ext" || key === "type") {
      setTestResult(null);
      setTestError(null);
    }
  }

  function buildConfig(): Record<string, unknown> {
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
    if (type === "ddm_simulation" && !runForever) {
      const n = Number(lengthValue);
      if (!isNaN(n) && n > 0) cfg["length"] = n;
      // runForever → omit length → backend runs endlessly
    }
    if (type === "web_report" && webReportCustom.trim()) {
      cfg["custom"] = JSON.parse(webReportCustom);
    }
    return cfg;
  }

  async function createDatasource(): Promise<number> {
    const res = await fetch("/api/v1/datasources", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, type, config: buildConfig() }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error?.message ?? body.detail ?? "Failed to create datasource");
    }
    const body = await res.json();
    return (body.data ?? body).id;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    if (type === "web_report" && webReportCustom.trim()) {
      try { JSON.parse(webReportCustom); } catch { setError("Custom Steps JSON is invalid"); setSaving(false); return; }
    }
    try {
      const dsId = await createDatasource();
      router.push(`/data/datasources/${dsId}`);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleUpload(file: File, symbol: string, timeframe: string, colMap: ColMap) {
    setSaving(true);
    setError(null);
    try {
      const dsId = await createDatasource();
      const form = new FormData();
      form.append("file", file);
      form.append("datasource_id", String(dsId));
      if (symbol) form.append("symbol", symbol);
      if (timeframe) form.append("timeframe", timeframe);
      if (colMap.close)    form.append("close_col",    colMap.close);
      if (colMap.open)     form.append("open_col",     colMap.open);
      if (colMap.high)     form.append("high_col",     colMap.high);
      if (colMap.low)      form.append("low_col",      colMap.low);
      if (colMap.volume)   form.append("volume_col",   colMap.volume);
      if (colMap.datetime) form.append("datetime_col", colMap.datetime);
      const upRes = await fetch("/api/v1/datasets/upload", { method: "POST", body: form });
      if (!upRes.ok) {
        const upBody = await upRes.json().catch(() => ({}));
        throw new Error(upBody.error?.message ?? upBody.detail ?? "File upload failed");
      }
      router.push(`/data/datasources/${dsId}`);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  const typeInfo = TYPE_DESCRIPTIONS[type];
  const fields = TYPE_FIELD_DEFS[type] ?? [];

  return (
    <div className="max-w-lg space-y-6">
      <div>
        <a href="/data" className="text-xs text-gray-500 hover:text-white">← Data</a>
        <h1 className="mt-1 text-2xl font-semibold text-white">New Datasource</h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <Field label="Name">
          <input
            className="md-input w-full"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. USDJPY H1"
            required
          />
        </Field>

        <Field label="Type">
          <select className="md-input w-full" value={type} onChange={(e) => handleTypeChange(e.target.value)}>
            {Object.entries(TYPE_DESCRIPTIONS).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
        </Field>

        {/* Type description */}
        <div className={`rounded border px-3 py-2 text-xs ${typeInfo?.status === "not-implemented" ? "border-yellow-800 bg-yellow-900/20 text-yellow-300" : "border-gray-800 bg-gray-900 text-gray-400"}`}>
          {typeInfo?.status === "not-implemented" && <span className="font-semibold text-yellow-400">Not yet implemented. </span>}
          {typeInfo?.description}
        </div>

        {/* Type-specific config fields */}
        {(fields.length > 0 || type === "ddm_simulation") && (
          <div className="rounded border border-gray-800 bg-gray-900 p-4 space-y-4">
            <p className="text-xs font-medium text-gray-400 uppercase">Configuration</p>
            {type === "ddm_simulation" && (
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
                  <Field label="Output Candles" hint="Number of OHLC candles to generate">
                    <input
                      type="number"
                      className="md-input w-full"
                      value={lengthValue}
                      onChange={(e) => setLengthValue(e.target.value)}
                      placeholder="1000"
                    />
                  </Field>
                )}
              </div>
            )}
            {fields.map((f) => (
              <Field key={f.key} label={f.label} hint={f.hint}>
                {f.type === "select" ? (
                  <>
                    <select
                      className="md-input w-full"
                      value={values[f.key] ?? ""}
                      onChange={(e) => handleFieldChange(f.key, e.target.value)}
                    >
                      {f.options?.map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                    {f.optionDescriptions?.[values[f.key]] && (
                      <p className="text-xs text-gray-400 bg-gray-800/60 rounded px-2 py-1.5 mt-1">
                        {f.optionDescriptions[values[f.key]]}
                      </p>
                    )}
                  </>
                ) : f.type === "date" ? (
                  <input
                    type="date"
                    className="md-input w-full"
                    value={values[f.key] ?? ""}
                    onChange={(e) => handleFieldChange(f.key, e.target.value)}
                    placeholder={f.placeholder}
                  />
                ) : (
                  <input
                    type={f.type === "number" ? "number" : "text"}
                    step={f.type === "number" ? "any" : undefined}
                    className="md-input w-full"
                    value={values[f.key] ?? ""}
                    onChange={(e) => handleFieldChange(f.key, e.target.value)}
                    placeholder={f.placeholder}
                  />
                )}
              </Field>
            ))}

            {/* ── Test Fetch Method (web_report only) ── */}
            {type === "web_report" && (
              <div className="space-y-3 pt-1 border-t border-gray-700/60">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs font-medium text-gray-400 uppercase">Test Fetch Method</p>
                    <p className="text-xs text-gray-600 mt-0.5">
                      Verify the selected Fetch Method can access the URL
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleTestFetch}
                    disabled={testLoading || !values["url"]}
                    className="rounded border border-gray-700 px-3 py-1 text-xs text-gray-300 hover:border-gray-400 hover:text-white disabled:opacity-40 shrink-0 ml-3"
                  >
                    {testLoading ? "Testing…" : "Test"}
                  </button>
                </div>
                {!testResult && !testLoading && !testError && (
                  <p className="text-xs text-gray-500">
                    Uses the configured Fetch Method to try accessing the URL and reports HTTP status, content type, and file size.
                  </p>
                )}
                {testError && <p className="text-xs text-red-400">{testError}</p>}
                {testResult && (
                  <div className="space-y-2">
                    {/* Status badge + summary row */}
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                      <span className={`inline-flex items-center gap-1 rounded px-2 py-0.5 font-semibold ${testResult.success ? "bg-green-900/50 text-green-300" : "bg-red-900/50 text-red-300"}`}>
                        {testResult.success ? "✓ Success" : "✗ Failed"}
                      </span>
                      {testResult.status_code !== undefined && (
                        <span className="text-gray-400">HTTP {testResult.status_code}</span>
                      )}
                      {testResult.content_type && (
                        <span className="text-gray-400">{testResult.content_type.split(";")[0]}</span>
                      )}
                      {testResult.size_bytes !== undefined && (
                        <span className="text-gray-400">
                          {testResult.size_bytes >= 1024 * 1024
                            ? `${(testResult.size_bytes / 1024 / 1024).toFixed(1)} MB`
                            : testResult.size_bytes >= 1024
                            ? `${(testResult.size_bytes / 1024).toFixed(1)} KB`
                            : `${testResult.size_bytes} B`}
                        </span>
                      )}
                      {testResult.title && (
                        <span className="text-gray-400 truncate max-w-[240px]" title={testResult.title}>
                          &ldquo;{testResult.title}&rdquo;
                        </span>
                      )}
                    </div>
                    {testResult.error && (
                      <p className="text-xs text-red-400">{testResult.error}</p>
                    )}
                    {/* Links table (HTML responses) */}
                    {testResult.links && testResult.links.length > 0 && (
                      <div className="space-y-1">
                        <p className="text-xs text-gray-400">
                          <span className={testResult.links.filter(l => l.matches_ext).length === 0 ? "text-amber-400" : "text-green-400"}>
                            {testResult.links.filter(l => l.matches_ext).length} matching .{values["ext"]}
                          </span>
                          {" · "}{testResult.links.length} total links
                        </p>
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
                          <div className="px-3 py-1.5 border-t border-gray-700/50 flex gap-3">
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
              </div>
            )}

            {/* ── Custom Steps JSON (web_report only) ── */}
            {type === "web_report" && (
              <Field
                label="Custom Steps (JSON)"
                hint="Optional. Leave blank for a direct download. Use link_parse / element_parse for multi-step scraping — see cyclic_downloader source.json for examples."
              >
                <textarea
                  className="md-input w-full font-mono text-xs"
                  rows={8}
                  value={webReportCustom}
                  onChange={(e) => setWebReportCustom(e.target.value)}
                  placeholder={`[{\n  "type": "link_parse",\n  "targets": [{\n    "value": ".*\\.pdf",\n    "ext": "pdf",\n    "filename": "{YYMMDD}.pdf",\n    "type": "goto_download",\n    "unique": "text",\n    "interval_days": 1\n  }]\n}]`}
                />
              </Field>
            )}
          </div>
        )}

        {error && <p className="text-sm text-red-400">{error}</p>}

        {type === "manual_upload" ? (
          <div className="rounded border border-gray-800 bg-gray-900 p-4 space-y-4">
            <p className="text-xs font-medium text-gray-400 uppercase">Upload CSV</p>
            <CsvUploadForm uploading={saving} onUpload={handleUpload} />
          </div>
        ) : (
          <button
            type="submit"
            disabled={saving || typeInfo?.status === "not-implemented"}
            className="w-full rounded bg-brand-500 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-40"
          >
            {saving ? "Creating…" : "Create Datasource"}
          </button>
        )}
      </form>

    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <label className="text-xs text-gray-400 uppercase">{label}</label>
      {children}
      {hint && <p className="text-xs text-gray-500">{hint}</p>}
    </div>
  );
}
