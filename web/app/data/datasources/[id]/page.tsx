"use client";

import { useState } from "react";
import useSWR, { mutate } from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";
import { useParams } from "next/navigation";
import { useToast } from "@/lib/toast";

// ── Config field definitions (shared with new-datasource page) ────────────────

const TIMEFRAME_OPTIONS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];

interface FieldDef {
  key: string;
  label: string;
  type: "text" | "number" | "select" | "date";
  options?: string[];
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
    {
      key: "length", label: "Output Candles", type: "number", placeholder: "1000",
      hint: "Number of OHLC candles to generate",
    },
    { key: "initial_price", label: "Initial Price", type: "number", placeholder: "100.0" },
    {
      key: "tick_time", label: "Seconds per Tick", type: "number", placeholder: "1.0",
      hint: "Simulated time per iteration. A price is only recorded when a trade occurs, so tick density per candle depends on agent dynamics.",
    },
    { key: "num_agent", label: "Number of Agents", type: "number", placeholder: "50" },
    { key: "seed", label: "Random Seed", type: "number", placeholder: "42" },
  ],
  web_report: [
    { key: "url", label: "URL", type: "text", placeholder: "https://…" },
    { key: "selector", label: "CSS Selector", type: "text", placeholder: "table" },
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
    `/api/v1/collection-jobs?datasource_id=${id}`,
    fetcher,
    { refreshInterval: (data) => (data?.some?.((j: any) => j.status === "running") ? 3000 : 0) }
  );

  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editValues, setEditValues] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  function startEdit() {
    setEditName(ds?.name ?? "");
    setEditValues(configToValues(ds?.config ?? {}));
    setEditing(true);
  }

  function cancelEdit() {
    setEditing(false);
  }

  async function saveEdit() {
    setSaving(true);
    try {
      const res = await fetch(`/api/v1/datasources/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: editName.trim(), config: valuesToConfig(ds?.type ?? "", editValues) }),
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

  async function startCollection() {
    const res = await fetch(`/api/v1/datasources/${id}/collect`, { method: "POST" });
    if (res.ok) {
      toast("Collection queued", "success");
    } else {
      const body = await res.json().catch(() => ({}));
      toast(body.error?.message ?? "Failed to start collection", "error");
    }
    mutate(`/api/v1/collection-jobs?datasource_id=${id}`);
  }

  async function runJob(jobId: number) {
    const res = await fetch(`/api/v1/collection-jobs/${jobId}/run`, { method: "POST" });
    if (res.ok) {
      toast("Collection queued", "success");
    } else {
      const body = await res.json().catch(() => ({}));
      toast(body.error?.message ?? "Failed to queue job", "error");
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
          {fields.length > 0 ? (
            <div className="grid grid-cols-2 gap-2">
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
                <select
                  className="md-input w-full"
                  value={editValues[f.key] ?? ""}
                  onChange={(e) => setEditValues((prev) => ({ ...prev, [f.key]: e.target.value }))}
                >
                  {f.options?.map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
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
          {fields.length === 0 && (
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

      {/* Collection Jobs */}
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
            <button
              onClick={() => runJob(job.id)}
              disabled={job.status === "running"}
              className="rounded bg-gray-700 px-3 py-1.5 text-xs text-white hover:bg-gray-600 disabled:opacity-40"
            >
              {job.status === "running" ? "Running…" : "Run now"}
            </button>
          </div>
        ))}
      </section>

    </div>
  );
}
