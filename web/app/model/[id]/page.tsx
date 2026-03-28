"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import useSWR, { mutate } from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";

const DEFAULT_HYPERPARAMS = {
  obs_len: 60,
  pred_len: 10,
  epochs: 50,
  batch_size: 32,
  lr: 0.001,
  feature_cols: ["close"],
  normalize: "returns",
  val_split: 0.2,
};

export default function ModelDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { data: model, isLoading, error } = useSWR(`/api/v1/models/${id}`, fetcher, { refreshInterval: 5000 });
  const { data: runs } = useSWR(`/api/v1/models/${id}/training-runs`, fetcher, { refreshInterval: 5000 });
  const { data: validations } = useSWR(`/api/v1/models/${id}/validations`, fetcher);
  const { data: datasets } = useSWR("/api/v1/datasets", fetcher);

  const [showTrainForm, setShowTrainForm] = useState(false);
  const [datasetId, setDatasetId] = useState("");
  const [hpText, setHpText] = useState(JSON.stringify(DEFAULT_HYPERPARAMS, null, 2));
  const [startingRun, setStartingRun] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);

  const [deploying, setDeploying] = useState<number | null>(null);
  const [deployError, setDeployError] = useState<string | null>(null);

  async function startTraining() {
    setTrainError(null);
    let hp: object;
    try {
      hp = JSON.parse(hpText);
    } catch {
      setTrainError("Hyperparams is not valid JSON");
      return;
    }
    if (!datasetId) {
      setTrainError("Select a dataset");
      return;
    }
    setStartingRun(true);
    try {
      const res = await fetch(`/api/v1/models/${id}/training-runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: parseInt(datasetId), hyperparams: hp }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setTrainError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      setShowTrainForm(false);
      mutate(`/api/v1/models/${id}/training-runs`);
      mutate(`/api/v1/models/${id}`);
    } finally {
      setStartingRun(false);
    }
  }

  async function deployRun(runId: number) {
    setDeployError(null);
    setDeploying(runId);
    try {
      const res = await fetch(`/api/v1/models/${id}/deploy?training_run_id=${runId}`, {
        method: "POST",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setDeployError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      mutate(`/api/v1/models/${id}`);
    } finally {
      setDeploying(null);
    }
  }

  if (isLoading) return <p className="text-gray-400">Loading…</p>;
  if (error || !model) return <p className="text-red-400">Model not found</p>;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold text-white">{model.name}</h1>
            <StatusBadge status={model.status} />
          </div>
          <p className="mt-1 text-sm text-gray-400">
            Architecture: <span className="text-gray-200">{model.architecture}</span>
            &nbsp;·&nbsp;Created {new Date(model.created_at).toLocaleDateString()}
          </p>
        </div>
        <a href="/model" className="text-sm text-gray-400 hover:text-white">
          ← Back
        </a>
      </div>

      {/* Architecture config */}
      <section>
        <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-gray-400">Config</h2>
        <pre className="rounded bg-gray-900 p-3 text-xs text-gray-300 overflow-auto">
          {JSON.stringify(model.config, null, 2)}
        </pre>
      </section>

      {/* Training runs */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-medium uppercase tracking-wide text-gray-400">Training Runs</h2>
          <button
            onClick={() => setShowTrainForm(!showTrainForm)}
            className="rounded bg-brand-500 px-3 py-1 text-xs text-white hover:bg-sky-400"
          >
            + Start Training
          </button>
        </div>

        {showTrainForm && (
          <div className="mb-4 rounded border border-gray-700 bg-gray-900 p-4 space-y-3">
            <div>
              <label className="mb-1 block text-xs text-gray-400">Dataset</label>
              <select
                value={datasetId}
                onChange={(e) => setDatasetId(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                <option value="">Select dataset…</option>
                {datasets?.map((d: any) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.row_count} rows)
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs text-gray-400">Hyperparameters (JSON)</label>
              <textarea
                value={hpText}
                onChange={(e) => setHpText(e.target.value)}
                rows={8}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 font-mono text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              />
            </div>
            {trainError && <p className="text-xs text-red-400">{trainError}</p>}
            <div className="flex gap-2">
              <button
                onClick={startTraining}
                disabled={startingRun}
                className="rounded bg-brand-500 px-3 py-1.5 text-xs text-white hover:bg-sky-400 disabled:opacity-50"
              >
                {startingRun ? "Queuing…" : "Start"}
              </button>
              <button
                onClick={() => setShowTrainForm(false)}
                className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-400 hover:text-white"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {deployError && <p className="mb-2 text-xs text-red-400">{deployError}</p>}

        {runs && runs.length === 0 && (
          <p className="text-sm text-gray-500">No training runs yet.</p>
        )}

        {runs && runs.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase">
                <th className="py-2 pr-4">Run</th>
                <th className="py-2 pr-4">Status</th>
                <th className="py-2 pr-4">Progress</th>
                <th className="py-2 pr-4">Val Loss</th>
                <th className="py-2 pr-4">Dataset</th>
                <th className="py-2" />
              </tr>
            </thead>
            <tbody>
              {runs.map((run: any) => (
                <tr key={run.id} className="border-b border-gray-800/50 hover:bg-gray-900">
                  <td className="py-2 pr-4 text-gray-300">#{run.id}</td>
                  <td className="py-2 pr-4">
                    <StatusBadge status={run.status} />
                  </td>
                  <td className="py-2 pr-4 text-gray-300">
                    {run.status === "running"
                      ? `Epoch ${run.current_epoch}${run.eta_seconds ? ` · ~${Math.round(run.eta_seconds / 60)}m left` : ""}`
                      : run.best_epoch != null
                      ? `Best epoch ${run.best_epoch}`
                      : "—"}
                  </td>
                  <td className="py-2 pr-4 text-gray-300">
                    {run.val_loss != null ? run.val_loss.toFixed(6) : "—"}
                  </td>
                  <td className="py-2 pr-4 text-gray-400">{run.dataset_id}</td>
                  <td className="py-2 text-right">
                    <div className="flex items-center justify-end gap-3">
                      <button
                        onClick={() => router.push(`/model/${id}/training-runs/${run.id}`)}
                        className="text-xs text-gray-400 hover:text-white"
                      >
                        Details
                      </button>
                      {run.status === "completed" && model.status !== "deployed" && (
                        <button
                          onClick={() => deployRun(run.id)}
                          disabled={deploying === run.id}
                          className="text-xs text-brand-500 hover:underline disabled:opacity-50"
                        >
                          {deploying === run.id ? "Deploying…" : "Deploy"}
                        </button>
                      )}
                      {run.status === "completed" && model.artifact_path === run.artifact_path && (
                        <span className="text-xs text-green-400">Deployed</span>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Validations */}
      {validations && validations.length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-gray-400">
            Validations
          </h2>
          <div className="space-y-3">
            {validations.map((v: any) => (
              <div key={v.id} className="rounded border border-gray-700 bg-gray-900 p-3">
                <p className="mb-1 text-xs text-gray-500">
                  Run #{v.training_run_id} · Dataset {v.dataset_id} ·{" "}
                  {new Date(v.computed_at).toLocaleString()}
                </p>
                <pre className="text-xs text-gray-300 overflow-auto">
                  {JSON.stringify(v.metrics, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
