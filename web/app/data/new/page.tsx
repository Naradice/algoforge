"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { CsvUploadForm, type ColMap, type UploadOptions } from "@/components/csv-upload-form";
import { apiFetch } from "@/lib/fetcher";
import { TYPE_FIELD_DEFS, TYPE_DEFAULTS, TYPE_DESCRIPTIONS } from "@/lib/datasource-types";

// ── Component ─────────────────────────────────────────────────────────────────

type TestResult = {
  success: boolean;
  status_code?: number;
  content_type?: string;
  size_bytes?: number;
  title?: string;
  links?: { href: string; text: string; filename: string; matches_ext: boolean }[];
  error?: string;
};

function getTestRecommendation(result: TestResult, fetchType: string): { message: string; suggestType?: string } | null {
  if (result.success) return null;
  if (result.status_code === 403) {
    if (fetchType === "load" || fetchType === "load_rep")
      return { message: "This site has CDN/bot protection. Try 'goto_download' instead.", suggestType: "goto_download" };
    return { message: "Access denied (403). The site may require login or actively block automated browsers." };
  }
  if (result.error?.includes("Failed to fetch"))
    return { message: "Browser fetch was blocked. Try 'goto_load' which captures the rendered page instead.", suggestType: "goto_load" };
  if (result.error?.includes("timed out"))
    return { message: "Page load timed out. If this is a direct file URL, try 'load' for a plain HTTP download.", suggestType: "load" };
  return null;
}

function autoSubfolder(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "").split(".")[0];
  } catch { return ""; }
}

export default function NewDatasourcePage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [type, setType] = useState("ohlc_download");
  const [values, setValues] = useState<Record<string, string>>(TYPE_DEFAULTS["ohlc_download"]);
  const [runForever, setRunForever] = useState(false);
  const [lengthValue, setLengthValue] = useState("1000");
  const [webReportCustom, setWebReportCustom] = useState("");
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [testPassed, setTestPassed] = useState(false);
  const [testLoading, setTestLoading] = useState(false);
  const [testShowLinks, setTestShowLinks] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleTypeChange(t: string) {
    setType(t);
    setValues(TYPE_DEFAULTS[t] ?? {});
    setRunForever(false);
    setLengthValue("1000");
    setWebReportCustom("");
    setTestResult(null);
    setTestPassed(false);
  }

  async function handleTestFetch() {
    const url = values["url"];
    const fetch_type = values["type"] || "load";
    const ext = values["ext"] || undefined;
    if (!url) return;
    setTestLoading(true);
    setTestResult(null);
    setTestPassed(false);
    setTestShowLinks(false);
    try {
      const res = await apiFetch("/api/v1/datasources/web-report/test-fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, fetch_type, ext }),
      });
      const body = await res.json().catch(() => ({}));
      const result: TestResult = res.ok ? (body.data ?? {}) : { success: false, error: body.error?.message ?? body.detail?.message ?? "Test failed" };
      setTestResult(result);

      if (result.success) {
        setTestPassed(true);
        // Auto-set subfolder from domain if not already set
        if (!values["subfolder"]) {
          setValues((prev) => ({ ...prev, subfolder: autoSubfolder(url) }));
        }
        // Auto-configure link_parse if matching links found (transparent to user)
        const matches = (result.links ?? []).filter((l: any) => l.matches_ext);
        if (matches.length > 0) {
          const linkFetchType = fetch_type === "goto_load" ? "goto_download" : fetch_type;
          const steps = [{ type: "link_parse", targets: [{ value: `.*\\.${ext ?? "pdf"}`, ext: ext ?? "pdf", filename: `{YYYYMMDD}_{filename}`, type: linkFetchType, unique: "text", interval_days: Number(values["interval_days"] ?? 1) || 1 }] }];
          setWebReportCustom(JSON.stringify(steps));
        } else {
          setWebReportCustom("");
        }
      }
    } finally {
      setTestLoading(false);
    }
  }

  function handleWebReportFieldChange(key: string, value: string) {
    setValues((prev) => ({ ...prev, [key]: value }));
    if (key === "url" || key === "ext" || key === "type") {
      // Changing URL/ext/method invalidates the test
      setTestResult(null);
      setTestPassed(false);
      setWebReportCustom("");
    }
  }

  function handleFieldChange(key: string, value: string) {
    setValues((prev) => ({ ...prev, [key]: value }));
  }

  function buildConfig(): Record<string, unknown> {
    if (type === "web_report") {
      const cfg: Record<string, unknown> = {
        url: values["url"],
        ext: values["ext"] || "pdf",
        type: values["type"] || "load",
        unique: "text",
        interval_days: Math.max(1, Number(values["interval_days"]) || 1),
      };
      if (values["subfolder"]) cfg["subfolder"] = values["subfolder"];
      if (values["filename"]) cfg["filename"] = values["filename"];
      if (values["download_time"]) cfg["download_time"] = values["download_time"];
      if (webReportCustom) cfg["custom"] = JSON.parse(webReportCustom);
      return cfg;
    }
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
    }
    if (type === "economic_calendar" && typeof cfg["indicators"] === "string") {
      cfg["indicators"] = (cfg["indicators"] as string).split(",").map((s) => s.trim()).filter(Boolean);
    }
    return cfg;
  }

  async function createDatasource(): Promise<number> {
    const res = await apiFetch("/api/v1/datasources", {
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
    if (type === "web_report" && !testPassed) return;
    setSaving(true);
    setError(null);
    try {
      const dsId = await createDatasource();
      router.push(`/data/datasources/${dsId}`);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function handleUpload(uploadFiles: File[], symbol: string, timeframe: string, colMap: ColMap, options: UploadOptions) {
    setSaving(true);
    setError(null);
    try {
      const dsId = await createDatasource();
      const form = new FormData();
      uploadFiles.forEach((f) => form.append("files", f));
      form.append("datasource_id", String(dsId));
      if (symbol)          form.append("symbol",       symbol);
      if (timeframe)       form.append("timeframe",    timeframe);
      if (colMap.close)    form.append("close_col",    colMap.close);
      if (colMap.open)     form.append("open_col",     colMap.open);
      if (colMap.high)     form.append("high_col",     colMap.high);
      if (colMap.low)      form.append("low_col",      colMap.low);
      if (colMap.volume)   form.append("volume_col",   colMap.volume);
      if (colMap.spread)   form.append("spread_col",   colMap.spread);
      if (colMap.datetime) form.append("datetime_col", colMap.datetime);
      form.append("merge", String(options.merge));
      const upRes = await apiFetch("/api/v1/datasets/upload", { method: "POST", body: form });
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

        {/* ── Web Report: guided wizard ── */}
        {type === "web_report" && (
          <div className="space-y-3">

            {/* Step 1 — Configure & Test */}
            <div className="rounded border border-gray-800 bg-gray-900 p-4 space-y-4">
              <p className="text-xs font-medium text-gray-400 uppercase tracking-wide">1 — Configure &amp; Test</p>

              <Field label="URL" hint="Landing page or direct file URL">
                <input
                  className="md-input w-full"
                  value={values["url"] ?? ""}
                  onChange={(e) => handleWebReportFieldChange("url", e.target.value)}
                  placeholder="https://www.example.com/reports"
                />
              </Field>

              <Field label="File Type">
                <select className="md-input w-full" value={values["ext"] ?? "pdf"} onChange={(e) => handleWebReportFieldChange("ext", e.target.value)}>
                  {["pdf", "html", "mp3", "txt"].map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
              </Field>

              <Field label="Fetch Method">
                <select className="md-input w-full" value={values["type"] ?? "load"} onChange={(e) => handleWebReportFieldChange("type", e.target.value)}>
                  {["load", "goto_load", "goto_download", "load_rep"].map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
                {({
                  load:          "Direct HTTP download. Fast — use for public links without bot protection.",
                  goto_load:     "Opens URL in a real browser and saves the rendered page as PDF.",
                  goto_download: "Uses browser fetch() to download files — bypasses CDN/Akamai protection. Required for most Japanese broker sites.",
                  load_rep:      "Plain HTTP download saved as raw HTML. Same CDN limitations as load.",
                } as Record<string, string>)[values["type"] ?? "load"] && (
                  <p className="text-xs text-gray-400 bg-gray-800/60 rounded px-2 py-1.5 mt-1">
                    {({
                      load:          "Direct HTTP download. Fast — use for public links without bot protection.",
                      goto_load:     "Opens URL in a real browser and saves the rendered page as PDF.",
                      goto_download: "Uses browser fetch() to download files — bypasses CDN/Akamai protection. Required for most Japanese broker sites.",
                      load_rep:      "Plain HTTP download saved as raw HTML. Same CDN limitations as load.",
                    } as Record<string, string>)[values["type"] ?? "load"]}
                  </p>
                )}
              </Field>

              <div className="flex justify-end pt-1">
                <button
                  type="button"
                  onClick={handleTestFetch}
                  disabled={testLoading || !values["url"]}
                  className="rounded bg-gray-700 px-4 py-1.5 text-xs text-white hover:bg-gray-600 disabled:opacity-40"
                >
                  {testLoading ? "Testing…" : testPassed ? "Re-test" : "Test"}
                </button>
              </div>

              {/* Test result */}
              {testResult && (
                <div className={`rounded border p-3 space-y-2 ${testResult.success ? "border-green-800 bg-green-900/10" : "border-red-800 bg-red-900/10"}`}>
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className={`font-semibold ${testResult.success ? "text-green-300" : "text-red-300"}`}>
                      {testResult.success ? "✓ Success" : "✗ Failed"}
                    </span>
                    {testResult.status_code !== undefined && <span className="text-gray-400">HTTP {testResult.status_code}</span>}
                    {testResult.content_type && <span className="text-gray-400">{testResult.content_type.split(";")[0]}</span>}
                    {testResult.size_bytes != null && testResult.size_bytes > 0 && (
                      <span className="text-gray-400">
                        {testResult.size_bytes >= 1048576 ? `${(testResult.size_bytes / 1048576).toFixed(1)} MB`
                          : testResult.size_bytes >= 1024 ? `${(testResult.size_bytes / 1024).toFixed(1)} KB`
                          : `${testResult.size_bytes} B`}
                      </span>
                    )}
                    {testResult.title && <span className="text-gray-400 truncate max-w-[240px]" title={testResult.title}>&ldquo;{testResult.title}&rdquo;</span>}
                  </div>

                  {/* Failure: error + recommendation */}
                  {!testResult.success && (() => {
                    const rec = getTestRecommendation(testResult, values["type"] ?? "load");
                    return (
                      <div className="space-y-1">
                        {testResult.error && <p className="text-xs text-red-400">{testResult.error}</p>}
                        {rec && (
                          <div className="flex items-start gap-2">
                            <p className="text-xs text-amber-300 flex-1">{rec.message}</p>
                            {rec.suggestType && (
                              <button type="button" onClick={() => handleWebReportFieldChange("type", rec.suggestType!)}
                                className="rounded border border-amber-700 px-2 py-0.5 text-xs text-amber-300 hover:bg-amber-900/30 shrink-0">
                                Switch to {rec.suggestType}
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })()}

                  {/* Success: matching files summary */}
                  {testResult.success && testResult.links && testResult.links.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-xs text-gray-300">
                        {testResult.links.filter(l => l.matches_ext).length > 0 ? (
                          <span className="text-green-400">
                            Found {testResult.links.filter(l => l.matches_ext).length} matching .{values["ext"]} files — will download all automatically
                          </span>
                        ) : (
                          <span className="text-amber-400">
                            No .{values["ext"]} links found on this page. Check the URL or file type.
                          </span>
                        )}
                        <span className="text-gray-500"> · {testResult.links.length} total links</span>
                      </p>
                      <div className="rounded border border-gray-700 overflow-hidden text-xs">
                        <table className="w-full">
                          <thead>
                            <tr className="bg-gray-800/60 text-left">
                              <th className="px-3 py-1.5 text-gray-400 font-medium">File</th>
                              <th className="px-3 py-1.5 text-gray-400 font-medium">Link text</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(testShowLinks ? testResult.links
                              : testResult.links.filter(l => l.matches_ext).length > 0
                                ? testResult.links.filter(l => l.matches_ext)
                                : testResult.links
                            ).slice(0, testShowLinks ? 300 : 10).map((link, i) => (
                              <tr key={i} className={`border-t border-gray-700/50 ${link.matches_ext ? "" : "opacity-40"}`}>
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
                        {testResult.links.length > 10 && (
                          <div className="px-3 py-1.5 border-t border-gray-700/50">
                            <button type="button" onClick={() => setTestShowLinks((v) => !v)} className="text-xs text-gray-500 hover:text-white">
                              {testShowLinks ? "Collapse" : `Show all ${testResult.links.length} links`}
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {testResult.success && (!testResult.links || testResult.links.length === 0) && (
                    <p className="text-xs text-green-400">Direct file — will download on each collection run.</p>
                  )}
                </div>
              )}
            </div>

            {/* Step 2 — Schedule (revealed after test passes) */}
            {testPassed && (
              <div className="rounded border border-gray-800 bg-gray-900 p-4 space-y-4">
                <p className="text-xs font-medium text-gray-400 uppercase tracking-wide">2 — Schedule</p>

                <Field label="Check every (days)" hint="How often to check for new files. Minimum 1 day.">
                  <input
                    type="number"
                    min="1"
                    className="md-input w-full"
                    value={values["interval_days"] ?? "1"}
                    onChange={(e) => handleFieldChange("interval_days", String(Math.max(1, Number(e.target.value) || 1)))}
                  />
                </Field>

                <Field label="Download at (UTC)" hint="Optional. Run at this exact time each day, e.g. 18:00. Leave blank to run immediately after the interval elapses.">
                  <input
                    type="time"
                    className="md-input w-full"
                    value={values["download_time"] ?? ""}
                    onChange={(e) => handleFieldChange("download_time", e.target.value)}
                  />
                </Field>

                {/* Advanced (subfolder / filename template) */}
                <div>
                  <button type="button" onClick={() => setAdvancedOpen((v) => !v)}
                    className="text-xs text-gray-500 hover:text-gray-300">
                    {advancedOpen ? "▾ Hide advanced options" : "▸ Advanced options"}
                  </button>
                  {advancedOpen && (
                    <div className="mt-3 space-y-3 pl-3 border-l border-gray-700">
                      <Field label="Subfolder" hint="Directory under artifacts/web_reports/ — auto-set from domain">
                        <input className="md-input w-full" value={values["subfolder"] ?? ""} onChange={(e) => handleFieldChange("subfolder", e.target.value)} placeholder="mizuho-sc" />
                      </Field>
                      <Field label="Filename template" hint="Placeholders: {YYYYMMDD} {YYMMDD} {filename} {basefilename}">
                        <input className="md-input w-full" value={values["filename"] ?? ""} onChange={(e) => handleFieldChange("filename", e.target.value)} placeholder="{YYYYMMDD}_{filename}" />
                      </Field>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Non-web_report config fields ── */}
        {type !== "web_report" && (fields.length > 0 || type === "ddm_simulation") && (
          <div className="rounded border border-gray-800 bg-gray-900 p-4 space-y-4">
            <p className="text-xs font-medium text-gray-400 uppercase">Configuration</p>
            {type === "ddm_simulation" && (
              <div className="space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={runForever} onChange={(e) => setRunForever(e.target.checked)} className="w-4 h-4 accent-brand-500" />
                  <span className="text-sm text-white">Run forever</span>
                </label>
                {!runForever && (
                  <Field label="Output Candles" hint="Number of OHLC candles to generate">
                    <input type="number" className="md-input w-full" value={lengthValue} onChange={(e) => setLengthValue(e.target.value)} placeholder="1000" />
                  </Field>
                )}
              </div>
            )}
            {fields.map((f) => (
              <Field key={f.key} label={f.label} hint={f.hint}>
                {f.type === "select" ? (
                  <>
                    <select className="md-input w-full" value={values[f.key] ?? ""} onChange={(e) => handleFieldChange(f.key, e.target.value)}>
                      {f.options?.map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                    {f.optionDescriptions?.[values[f.key]] && (
                      <p className="text-xs text-gray-400 bg-gray-800/60 rounded px-2 py-1.5 mt-1">{f.optionDescriptions[values[f.key]]}</p>
                    )}
                  </>
                ) : f.type === "date" ? (
                  <input type="date" className="md-input w-full" value={values[f.key] ?? ""} onChange={(e) => handleFieldChange(f.key, e.target.value)} />
                ) : (
                  <input type={f.type === "number" ? "number" : "text"} step={f.type === "number" ? "any" : undefined} className="md-input w-full" value={values[f.key] ?? ""} onChange={(e) => handleFieldChange(f.key, e.target.value)} placeholder={f.placeholder} />
                )}
              </Field>
            ))}
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
            disabled={saving || typeInfo?.status === "not-implemented" || (type === "web_report" && !testPassed)}
            className="w-full rounded bg-brand-500 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-40"
          >
            {saving ? "Creating…" : type === "web_report" && !testPassed ? "Test required before creating" : "Create Datasource"}
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
