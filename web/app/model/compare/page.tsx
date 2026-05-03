"use client";

import { useState } from "react";
import useSWR from "swr";
import { apiFetch, fetcher } from "@/lib/fetcher";
import { MultiRunLossChart } from "@/components/multi-run-loss-chart";
import type { RunSeries } from "@/components/multi-run-loss-chart";

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
  dataset_id: number;
  hyperparams: Record<string, unknown>;
  status: string;
  best_epoch: number | null;
  val_loss: number | null;
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
                  <th className="pb-2 pr-4">Best Epoch</th>
                  <th className="pb-2 pr-4">Best Val Loss</th>
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
                    <td className="py-2 pr-4 text-gray-300">{r.best_epoch ?? "—"}</td>
                    <td className={`py-2 pr-4 font-mono ${r.val_loss != null ? "text-green-400" : "text-gray-400"}`}>
                      {r.val_loss != null ? r.val_loss.toFixed(6) : "—"}
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
