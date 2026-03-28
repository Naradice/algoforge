"use client";

import { useState } from "react";
import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { LossChart } from "@/components/loss-chart";

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

export default function ModelComparePage() {
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [comparing, setComparing] = useState(false);
  const [comparison, setComparison] = useState<CompareResult[] | null>(null);

  const { data: models } = useSWR("/api/v1/models", fetcher);

  function toggleRun(id: number) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleCompare() {
    if (selectedIds.size < 2) return;
    setComparing(true);
    try {
      const ids = Array.from(selectedIds).join(",");
      const res = await fetch(`/api/v1/training-runs/compare?run_ids=${ids}`);
      const body = await res.json();
      setComparison(body.data ?? body);
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
          {comparing ? "Comparing…" : `Compare ${selectedIds.size} runs`}
        </button>
      </div>

      <p className="text-gray-400 text-sm">Select 2+ completed training runs to compare side-by-side.</p>

      {/* Select runs from models */}
      {models && (
        <div className="space-y-4">
          {(models as { id: number; name: string }[]).map((model) => (
            <ModelRunSelector
              key={model.id}
              model={model}
              selectedIds={selectedIds}
              onToggle={toggleRun}
            />
          ))}
        </div>
      )}

      {/* Comparison results */}
      {comparison && (
        <div className="space-y-4">
          <h2 className="text-lg font-medium text-white">Comparison</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-400">
                  <th className="pb-2 pr-4">Run ID</th>
                  <th className="pb-2 pr-4">Model</th>
                  <th className="pb-2 pr-4">Dataset</th>
                  <th className="pb-2 pr-4">Best Epoch</th>
                  <th className="pb-2 pr-4">Best Val Loss</th>
                  <th className="pb-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((r) => (
                  <tr key={r.run_id} className="border-t border-gray-800">
                    <td className="py-2 pr-4 text-white">#{r.run_id}</td>
                    <td className="py-2 pr-4 text-gray-300">{r.model_id}</td>
                    <td className="py-2 pr-4 text-gray-300">{r.dataset_id}</td>
                    <td className="py-2 pr-4 text-gray-300">{r.best_epoch ?? "—"}</td>
                    <td className={`py-2 pr-4 ${r.val_loss != null ? "text-green-400" : "text-gray-400"}`}>
                      {r.val_loss != null ? r.val_loss.toFixed(6) : "—"}
                    </td>
                    <td className="py-2 text-gray-400">{r.status}</td>
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
  onToggle: (id: number) => void;
}) {
  const { data: runs } = useSWR<TrainingRun[]>(`/api/v1/models/${model.id}/training-runs`, fetcher);

  if (!runs || runs.length === 0) return null;

  return (
    <div className="rounded border border-gray-700 bg-gray-900 p-4">
      <h3 className="mb-3 text-sm font-medium text-white">{model.name}</h3>
      <div className="space-y-2">
        {runs.filter((r) => r.status === "completed").map((run) => (
          <label key={run.id} className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={selectedIds.has(run.id)}
              onChange={() => onToggle(run.id)}
              className="rounded border-gray-600 bg-gray-800 text-brand-500"
            />
            <span className="text-sm text-gray-300">
              Run #{run.id} · val_loss: {run.val_loss?.toFixed(6) ?? "—"} · best epoch: {run.best_epoch ?? "—"}
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}
