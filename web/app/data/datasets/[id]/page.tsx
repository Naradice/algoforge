"use client";

import { useState } from "react";
import useSWR, { mutate } from "swr";
import { useParams } from "next/navigation";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";
import { ACFPlot } from "@/components/acf-plot";
import { CCDFPlot } from "@/components/ccdf-plot";
import { QQPlot } from "@/components/qq-plot";

type Tab = "overview" | "preview" | "characteristics";

export default function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [tab, setTab] = useState<Tab>("overview");
  const [deleting, setDeleting] = useState(false);
  const { data: dataset } = useSWR(`/api/v1/datasets/${id}`, fetcher);
  const { data: chars } = useSWR(tab === "characteristics" ? `/api/v1/datasets/${id}/characteristics` : null, fetcher);
  const { data: preview } = useSWR(tab === "preview" && dataset?.status === "ready" ? `/api/v1/datasets/${id}/preview?rows=200` : null, fetcher);

  async function computeChars() {
    await fetch(`/api/v1/datasets/${id}/characteristics/compute`, { method: "POST" });
    mutate(`/api/v1/datasets/${id}/characteristics`);
  }

  async function deleteDataset() {
    if (!confirm(`Delete dataset "${dataset?.name}"? This cannot be undone.`)) return;
    setDeleting(true);
    const res = await fetch(`/api/v1/datasets/${id}`, { method: "DELETE" });
    if (res.ok) {
      window.location.href = "/data";
    } else {
      const body = await res.json().catch(() => ({}));
      alert(body.error?.message ?? "Failed to delete");
      setDeleting(false);
    }
  }

  return (
    <div className="space-y-4 max-w-5xl">
      <div className="flex items-start justify-between">
        <div>
          <a href="/data" className="text-xs text-gray-500 hover:text-white">← Data</a>
          <h1 className="mt-1 text-2xl font-semibold text-white">{dataset?.name ?? "…"}</h1>
          <div className="flex gap-3 mt-1 text-xs text-gray-400">
            {dataset?.symbol && <span>{dataset.symbol}</span>}
            {dataset?.timeframe && <span>{dataset.timeframe}</span>}
            {dataset && <StatusBadge status={dataset.status} />}
            {dataset?.row_count && <span>{dataset.row_count.toLocaleString()} rows</span>}
          </div>
        </div>
        {dataset?.status === "ready" && (
          <div className="flex gap-2">
            <a
              href={`/api/v1/datasets/${id}/download`}
              className="rounded border border-gray-600 px-3 py-1.5 text-xs text-gray-300 hover:border-gray-400 hover:text-white"
            >
              Download CSV
            </a>
            <button
              onClick={deleteDataset}
              disabled={deleting}
              className="rounded border border-red-800 px-3 py-1.5 text-xs text-red-400 hover:bg-red-900/20 disabled:opacity-50"
            >
              {deleting ? "Deleting…" : "Delete"}
            </button>
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-800">
        {(["overview", "preview", "characteristics"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm border-b-2 -mb-px transition-colors ${
              tab === t ? "border-brand-500 text-brand-500" : "border-transparent text-gray-400 hover:text-white"
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Overview */}
      {tab === "overview" && dataset && (
        <div className="grid grid-cols-2 gap-4">
          <InfoCard label="From" value={dataset.from_ts ? new Date(dataset.from_ts).toLocaleDateString() : "—"} />
          <InfoCard label="To" value={dataset.to_ts ? new Date(dataset.to_ts).toLocaleDateString() : "—"} />
          <InfoCard label="Rows" value={dataset.row_count?.toLocaleString() ?? "—"} />
          <InfoCard label="Status" value={dataset.status} />
          <InfoCard label="Artifact" value={dataset.artifact_path ?? "—"} mono />
        </div>
      )}

      {/* Preview */}
      {tab === "preview" && (
        <div className="overflow-x-auto">
          {!preview && <p className="text-gray-400 text-sm">Loading preview…</p>}
          {preview && preview.length > 0 && (
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400 uppercase">
                  {Object.keys(preview[0]).map((col) => (
                    <th key={col} className="py-2 pr-4 whitespace-nowrap">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.slice(0, 50).map((row: any, i: number) => (
                  <tr key={i} className="border-b border-gray-800/40">
                    {Object.values(row).map((v: any, j: number) => (
                      <td key={j} className="py-1.5 pr-4 text-gray-300 whitespace-nowrap font-mono">
                        {typeof v === "number" ? v.toFixed(5) : String(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Characteristics */}
      {tab === "characteristics" && (
        <div className="space-y-4">
          {!chars && (
            <div className="flex items-center gap-3">
              <p className="text-gray-400 text-sm">No characteristics computed yet.</p>
              <button onClick={computeChars} className="rounded bg-brand-500 px-3 py-1.5 text-xs text-white hover:bg-sky-400">
                Compute now
              </button>
            </div>
          )}
          {chars && (
            <>
              <div className="flex justify-end">
                <button onClick={computeChars} className="rounded bg-gray-700 px-3 py-1.5 text-xs text-white hover:bg-gray-600">
                  Recompute
                </button>
              </div>
              <CharacteristicsPanel metrics={chars.metrics} />
            </>
          )}
        </div>
      )}
    </div>
  );
}

function InfoCard({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded border border-gray-800 bg-gray-900 p-3">
      <p className="text-xs text-gray-400 uppercase">{label}</p>
      <p className={`mt-1 text-white ${mono ? "font-mono text-xs" : "text-sm font-medium"}`}>{value}</p>
    </div>
  );
}

function CharacteristicsPanel({ metrics }: { metrics: Record<string, any> }) {
  const fullStats = metrics?.full?.stats;
  const plots = metrics?.plots;

  return (
    <div className="space-y-4">
      {fullStats && (
        <div className="grid grid-cols-3 gap-3">
          <StatCard label="Rows" value={fullStats.n?.toLocaleString()} />
          <StatCard label="Hurst" value={fullStats.hurst?.toFixed(4)} hint="0.5 = random, >0.5 = trending" />
          <StatCard label="Kurtosis" value={fullStats.kurtosis?.toFixed(4)} hint=">3 = fat tails" />
          <StatCard label="Skewness" value={fullStats.skewness?.toFixed(4)} />
          <StatCard label="Return std" value={fullStats.std?.toExponential(3)} />
          <StatCard label="Return mean" value={fullStats.mean?.toExponential(3)} />
        </div>
      )}
      {metrics?.basic_stats && !fullStats && (
        <div className="grid grid-cols-3 gap-3">
          <StatCard label="Rows" value={metrics.basic_stats.row_count?.toLocaleString()} />
          <StatCard label="Hurst" value={metrics.basic_stats.hurst?.toFixed(4)} />
          <StatCard label="Return std" value={metrics.basic_stats.return_std?.toExponential(3)} />
        </div>
      )}
      {plots && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {plots.acf && (
            <div className="rounded border border-gray-800 bg-gray-900 p-4">
              <ACFPlot data={plots.acf} title="ACF (Returns)" />
            </div>
          )}
          {plots.pacf && (
            <div className="rounded border border-gray-800 bg-gray-900 p-4">
              <ACFPlot data={plots.pacf} title="PACF (Returns)" />
            </div>
          )}
          {plots.ccdf && (
            <div className="rounded border border-gray-800 bg-gray-900 p-4">
              <CCDFPlot data={plots.ccdf} />
            </div>
          )}
          {plots.qq && (
            <div className="rounded border border-gray-800 bg-gray-900 p-4">
              <QQPlot data={plots.qq} />
            </div>
          )}
        </div>
      )}
      <div className="rounded border border-gray-800 bg-gray-900 p-4">
        <p className="text-xs text-gray-400 mb-2 uppercase">Raw metrics (JSON)</p>
        <pre className="text-xs text-gray-500 overflow-x-auto max-h-64">{JSON.stringify(metrics, null, 2)}</pre>
      </div>
    </div>
  );
}

function StatCard({ label, value, hint }: { label: string; value: string | undefined; hint?: string }) {
  return (
    <div className="rounded border border-gray-800 bg-gray-900 p-3">
      <p className="text-xs text-gray-400 uppercase">{label}</p>
      <p className="mt-1 text-lg font-bold text-white font-mono">{value ?? "—"}</p>
      {hint && <p className="text-xs text-gray-500 mt-0.5">{hint}</p>}
    </div>
  );
}
