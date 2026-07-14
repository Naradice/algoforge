"use client";

import { useState } from "react";
import useSWR from "swr";
import { apiFetch, fetcher } from "@/lib/fetcher";
import { MultiRunLossChart } from "@/components/multi-run-loss-chart";
import type { RunSeries } from "@/components/multi-run-loss-chart";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface TrainingRun {
  id: number;
  model_id: number;
  dataset_id: number;
  hyperparams: Record<string, unknown>;
  status: string;
  best_epoch: number | null;
  val_loss: number | null;
}

interface CompareResult {
  run_id: number;
  model_id: number;
  model_name: string | null;
  architecture: string | null;
  dataset_id: number;
  hyperparams: Record<string, unknown>;
  status: string;
  best_epoch: number | null;
  val_loss: number | null;
  num_params: number | null;
  validation: Record<string, number> | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  preprocessed_characteristics: Record<string, any> | null;
}

// Distinct palette — mirrors multi-run-loss-chart.tsx so run colors stay consistent across charts
const PALETTE = ["#0ea5e9", "#f97316", "#22c55e", "#a855f7", "#ec4899", "#eab308", "#14b8a6", "#f43f5e"];

function formatParams(n: number | null): string {
  if (n == null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

// ── Joint analysis: training-data characteristics × model size × performance ──

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function numOrNull(v: any): number | null {
  return typeof v === "number" && !isNaN(v) ? v : null;
}

const X_METRICS = [
  { key: "num_params", label: "Model size (params)" },
  { key: "data.hurst", label: "Data: Hurst exponent" },
  { key: "data.memory_length", label: "Data: Memory length" },
  { key: "data.periodicity_strength", label: "Data: Periodicity strength" },
  { key: "data.spectral_entropy", label: "Data: Spectral entropy" },
  { key: "data.flatness_score", label: "Data: Wavelet flatness (multiscale)" },
  { key: "data.permutation_entropy", label: "Data: Permutation entropy" },
  { key: "data.sample_entropy", label: "Data: Sample entropy" },
  { key: "data.n_changepoints", label: "Data: Regime changes (count)" },
] as const;

const Y_METRICS = [
  { key: "val_loss", label: "Validation loss" },
  { key: "validation.directional_accuracy", label: "Directional accuracy" },
  { key: "validation.mae", label: "MAE" },
  { key: "validation.rmse", label: "RMSE" },
  { key: "validation.sharpe_proxy", label: "Sharpe proxy" },
] as const;

interface AnalysisEntry {
  runId: number;
  label: string;
  architecture: string | null;
  color: string;
  numParams: number | null;
  valLoss: number | null;
  validation: Record<string, number> | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  datasetMetrics: Record<string, any> | null;
}

function metricValue(entry: AnalysisEntry, key: string): number | null {
  if (key === "num_params") return entry.numParams;
  if (key === "val_loss") return entry.valLoss;
  if (key.startsWith("validation.")) {
    return numOrNull(entry.validation?.[key.slice("validation.".length)]);
  }
  if (key.startsWith("data.")) {
    const m = entry.datasetMetrics;
    if (!m) return null;
    switch (key.slice("data.".length)) {
      case "hurst": return numOrNull(m.long_range_dependence?.hurst);
      case "memory_length": return numOrNull(m.long_range_dependence?.memory_length);
      case "periodicity_strength": return numOrNull(m.spectral_periodicity?.periodicity_strength);
      case "spectral_entropy": return numOrNull(m.spectral_periodicity?.spectral_entropy);
      case "flatness_score": return numOrNull(m.multiscale_wavelet?.flatness_score);
      case "permutation_entropy": return numOrNull(m.complexity_nonlinearity?.permutation_entropy);
      case "sample_entropy": return numOrNull(m.complexity_nonlinearity?.sample_entropy);
      case "n_changepoints": return numOrNull(m.regime_changes?.n_changepoints);
      default: return null;
    }
  }
  return null;
}

function AnalysisSection({ entries }: { entries: AnalysisEntry[] }) {
  const [xKey, setXKey] = useState<string>(X_METRICS[0].key);
  const [yKey, setYKey] = useState<string>(Y_METRICS[0].key);

  const xLabel = X_METRICS.find((m) => m.key === xKey)?.label ?? xKey;
  const yLabel = Y_METRICS.find((m) => m.key === yKey)?.label ?? yKey;

  const points = entries
    .map((e) => {
      const x = metricValue(e, xKey);
      const y = metricValue(e, yKey);
      return x != null && y != null ? { x, y, entry: e } : null;
    })
    .filter((p): p is { x: number; y: number; entry: AnalysisEntry } => p !== null);

  return (
    <div className="rounded border border-gray-700 bg-gray-900 p-4 space-y-3">
      <h2 className="text-sm font-medium text-gray-300">Data × Model Analysis</h2>
      <p className="text-xs text-gray-600">
        &quot;Data:&quot; metrics use each run&apos;s own as-trained characteristics (after its
        preprocessing) when available, falling back to the raw dataset&apos;s for older runs.
      </p>
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5 text-gray-400">
          X:
          <select value={xKey} onChange={(e) => setXKey(e.target.value)}
            className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-white">
            {X_METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-gray-400">
          Y:
          <select value={yKey} onChange={(e) => setYKey(e.target.value)}
            className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-white">
            {Y_METRICS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
        </label>
      </div>
      {points.length === 0 ? (
        <p className="text-xs text-gray-500">
          None of the {entries.length} compared runs have both metrics available.
        </p>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart margin={{ top: 8, right: 20, bottom: 24, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis type="number" dataKey="x" name={xLabel} tick={{ fontSize: 10, fill: "#6b7280" }}
                label={{ value: xLabel, position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
              <YAxis type="number" dataKey="y" name={yLabel} tick={{ fontSize: 10, fill: "#6b7280" }} width={64}
                label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
              <Tooltip
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                content={({ active, payload }: any) => {
                  if (!active || !payload?.length) return null;
                  const p = payload[0].payload as { x: number; y: number; entry: AnalysisEntry };
                  return (
                    <div className="rounded border border-gray-700 bg-gray-900 px-2 py-1.5 text-xs">
                      <p className="text-white font-medium">{p.entry.label}</p>
                      <p className="text-gray-400">{p.entry.architecture ?? "—"}</p>
                      <p className="text-gray-300">{xLabel}: {p.x}</p>
                      <p className="text-gray-300">{yLabel}: {p.y}</p>
                    </div>
                  );
                }}
              />
              <Scatter data={points} shape={(props: unknown) => {
                const { cx, cy, payload } = props as { cx: number; cy: number; payload: { entry: AnalysisEntry } };
                return <circle cx={cx} cy={cy} r={5} fill={payload.entry.color} opacity={0.85} />;
              }} />
            </ScatterChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-600">
            {points.length} of {entries.length} runs plotted (runs missing either metric are omitted).
          </p>
        </>
      )}
    </div>
  );
}

/** Build a short human-readable label from a run's key hyperparams */
function runLabel(run: TrainingRun): string {
  const hp = run.hyperparams ?? {};
  const parts: string[] = [`Run #${run.id}`];
  if (hp.lr != null) parts.push(`lr=${hp.lr}`);
  if (hp.batch_size != null) parts.push(`bs=${hp.batch_size}`);
  if (hp.epochs != null) parts.push(`ep=${hp.epochs}`);
  return parts.join(" ");
}

export default function ModelComparePage() {
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [comparing, setComparing] = useState(false);
  const [comparison, setComparison] = useState<CompareResult[] | null>(null);
  const [runSeries, setRunSeries] = useState<RunSeries[]>([]);

  const { data: models } = useSWR<{ id: number; name: string }[]>("/api/v1/models", fetcher);

  function toggleRun(run: TrainingRun) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(run.id)) next.delete(run.id);
      else next.add(run.id);
      return next;
    });
  }

  async function handleCompare() {
    if (selectedIds.size < 2) return;
    setComparing(true);
    try {
      const ids = Array.from(selectedIds);
      const idsParam = ids.join(",");

      // Fetch summary comparison + per-run epoch metrics in parallel
      const [compRes, ...metricResults] = await Promise.all([
        apiFetch(`/api/v1/training-runs/compare?run_ids=${idsParam}`).then((r) => r.json()),
        ...ids.map((id) =>
          apiFetch(`/api/v1/training-runs/${id}/metrics`).then((r) => r.json())
        ),
      ]);

      setComparison(compRes.data ?? compRes);

      // Build RunSeries for the chart
      // We need each run's label — pull from the comparison summary
      const summaryMap = new Map<number, CompareResult>(
        (compRes.data ?? compRes).map((r: CompareResult) => [r.run_id, r])
      );

      const series: RunSeries[] = ids.map((id, i) => {
        const summary = summaryMap.get(id);
        const hp = summary?.hyperparams ?? {};
        const parts = [`Run #${id}`];
        if (hp.lr != null) parts.push(`lr=${hp.lr}`);
        if (hp.batch_size != null) parts.push(`bs=${hp.batch_size}`);
        return {
          runId: id,
          label: parts.join(" "),
          metrics: (metricResults[i]?.data ?? []).map((m: { epoch: number; train_loss: number; val_loss: number }) => ({
            epoch: m.epoch,
            train_loss: m.train_loss,
            val_loss: m.val_loss,
          })),
        };
      });

      setRunSeries(series);
    } finally {
      setComparing(false);
    }
  }

  // Runs that resolved successfully (error entries carry {run_id, error} and no status)
  const validComparison = (comparison ?? []).filter((r) => r.status != null);
  const datasetIds = Array.from(new Set(validComparison.map((r) => r.dataset_id)));

  const { data: datasetCharsData } = useSWR(
    datasetIds.length ? ["compare-model-chars", ...datasetIds] : null,
    () => Promise.all(datasetIds.map((id) =>
      apiFetch(`/api/v1/datasets/${id}/characteristics`).then((r) => r.json()).then((j) => j.data)
    ))
  );
  const datasetCharsMap = new Map<number, Record<string, unknown> | null>(
    datasetIds.map((id, i) => [id, datasetCharsData?.[i]?.metrics ?? null])
  );

  const analysisEntries: AnalysisEntry[] = validComparison.map((r, i) => ({
    runId: r.run_id,
    label: runSeries.find((s) => s.runId === r.run_id)?.label ?? `Run #${r.run_id}`,
    architecture: r.architecture,
    color: PALETTE[i % PALETTE.length],
    numParams: r.num_params,
    valLoss: r.val_loss,
    validation: r.validation,
    // Prefer this run's own as-trained characteristics (post-preprocessing, pre-normalize);
    // fall back to the raw dataset's for older runs computed before that field existed.
    datasetMetrics: r.preprocessed_characteristics ?? datasetCharsMap.get(r.dataset_id) ?? null,
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-white">Compare Training Runs</h1>
        <button
          onClick={handleCompare}
          disabled={selectedIds.size < 2 || comparing}
          className="rounded bg-brand-500 px-4 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-50"
        >
          {comparing ? "Loading…" : `Compare ${selectedIds.size} run${selectedIds.size !== 1 ? "s" : ""}`}
        </button>
      </div>

      <p className="text-gray-400 text-sm">
        Select 2+ completed training runs to compare val_loss curves and metrics side-by-side.
      </p>

      {/* Run selectors grouped by model */}
      {models && (
        <div className="space-y-4">
          {models.map((model) => (
            <ModelRunSelector
              key={model.id}
              model={model}
              selectedIds={selectedIds}
              onToggle={toggleRun}
            />
          ))}
          {models.length === 0 && (
            <p className="text-gray-500 text-sm">No models found.</p>
          )}
        </div>
      )}

      {/* Overlaid val_loss chart */}
      {runSeries.length > 0 && (
        <div className="rounded border border-gray-700 bg-gray-900 p-4">
          <h2 className="mb-3 text-sm font-medium text-gray-300">Validation Loss — All Runs</h2>
          <MultiRunLossChart runs={runSeries} />
        </div>
      )}

      {/* Summary metrics table */}
      {comparison && (
        <div className="rounded border border-gray-700 bg-gray-900 p-4">
          <h2 className="mb-3 text-sm font-medium text-gray-300">Summary</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-400 border-b border-gray-700">
                  <th className="pb-2 pr-4">Run</th>
                  <th className="pb-2 pr-4">Architecture</th>
                  <th className="pb-2 pr-4">Params</th>
                  <th className="pb-2 pr-4">Best Epoch</th>
                  <th className="pb-2 pr-4">Best Val Loss</th>
                  <th className="pb-2 pr-4">Dir. Acc.</th>
                  <th className="pb-2 pr-4">Sharpe</th>
                  <th className="pb-2 pr-4">Status</th>
                  <th className="pb-2">Key Hyperparams</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((r) => (
                  <tr key={r.run_id} className="border-t border-gray-800">
                    <td className="py-2 pr-4">
                      <a
                        href={`/model/${r.model_id}/training-runs/${r.run_id}`}
                        className="text-brand-400 hover:text-brand-300 font-medium"
                      >
                        #{r.run_id}
                      </a>
                    </td>
                    <td className="py-2 pr-4 text-gray-300">{r.architecture ?? "—"}</td>
                    <td className="py-2 pr-4 text-gray-300 font-mono">{formatParams(r.num_params)}</td>
                    <td className="py-2 pr-4 text-gray-300">{r.best_epoch ?? "—"}</td>
                    <td className={`py-2 pr-4 font-mono ${r.val_loss != null ? "text-green-400" : "text-gray-400"}`}>
                      {r.val_loss != null ? r.val_loss.toFixed(6) : "—"}
                    </td>
                    <td className="py-2 pr-4 text-gray-300 font-mono">
                      {r.validation?.directional_accuracy != null ? `${(r.validation.directional_accuracy * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td className="py-2 pr-4 text-gray-300 font-mono">
                      {r.validation?.sharpe_proxy != null ? r.validation.sharpe_proxy.toFixed(3) : "—"}
                    </td>
                    <td className="py-2 pr-4 text-gray-400">{r.status}</td>
                    <td className="py-2 text-gray-400 text-xs font-mono">
                      {Object.entries(r.hyperparams ?? {})
                        .filter(([k]) => ["lr", "batch_size", "epochs"].includes(k))
                        .map(([k, v]) => `${k}=${v}`)
                        .join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Joint training-data characteristics × model size × performance analysis */}
      {analysisEntries.length > 0 && <AnalysisSection entries={analysisEntries} />}
    </div>
  );
}

function ModelRunSelector({
  model,
  selectedIds,
  onToggle,
}: {
  model: { id: number; name: string };
  selectedIds: Set<number>;
  onToggle: (run: TrainingRun) => void;
}) {
  const { data: runs } = useSWR<TrainingRun[]>(
    `/api/v1/models/${model.id}/training-runs`,
    fetcher
  );

  const completed = (runs ?? []).filter((r) => r.status === "completed");
  if (completed.length === 0) return null;

  return (
    <div className="rounded border border-gray-700 bg-gray-900 p-4">
      <h3 className="mb-3 text-sm font-medium text-white">{model.name}</h3>
      <div className="space-y-2">
        {completed.map((run) => (
          <label key={run.id} className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={selectedIds.has(run.id)}
              onChange={() => onToggle(run)}
              className="rounded border-gray-600 bg-gray-800 text-brand-500"
            />
            <span className="text-sm text-gray-300">{runLabel(run)}</span>
            <span className="text-xs text-gray-500">
              val_loss: {run.val_loss?.toFixed(6) ?? "—"} · best epoch: {run.best_epoch ?? "—"}
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}
