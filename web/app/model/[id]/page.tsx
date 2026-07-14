"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import useSWR, { mutate } from "swr";
import { apiFetch, fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";
import { useToast } from "@/lib/toast";
import { summarizePreprocessing } from "@/lib/preprocessing";

// ── Types ──────────────────────────────────────────────────────────────────

interface TrainingParams {
  obs_len: number;
  pred_len: number;
  epochs: number;
  batch_size: number;
  lr: number;
  val_split: number;
}

// ── Defaults ───────────────────────────────────────────────────────────────

const DEFAULT_TRAINING: TrainingParams = {
  obs_len: 60, pred_len: 10, epochs: 50, batch_size: 32,
  lr: 0.001, val_split: 0.2,
};

const DEFAULT_SEARCH_GRID = JSON.stringify({ lr: [0.001, 0.0001], batch_size: [32, 64] }, null, 2);

// ── Component ──────────────────────────────────────────────────────────────

export default function ModelDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { toast } = useToast();

  const { data: model, isLoading, error } = useSWR(`/api/v1/models/${id}`, fetcher, { refreshInterval: 5000 });
  const { data: runs } = useSWR(`/api/v1/models/${id}/training-runs`, fetcher, { refreshInterval: 5000 });
  const { data: validations } = useSWR(`/api/v1/models/${id}/validations`, fetcher);
  const { data: datasets } = useSWR("/api/v1/datasets", fetcher);

  // ── Train form state ──
  const [showTrainForm, setShowTrainForm] = useState(false);
  const [datasetId, setDatasetId] = useState("");
  const [preprocessedDatasetId, setPreprocessedDatasetId] = useState("");
  const [trainingParams, setTrainingParams] = useState<TrainingParams>(DEFAULT_TRAINING);

  const { data: preprocessedDatasets } = useSWR(
    datasetId ? `/api/v1/preprocessed-datasets?dataset_id=${datasetId}&page_size=200` : null,
    fetcher
  );

  const [startingRun, setStartingRun] = useState(false);
  const [trainError, setTrainError] = useState<string | null>(null);

  // ── Deploy state ──
  const [deploying, setDeploying] = useState<number | null>(null);
  const [deployError, setDeployError] = useState<string | null>(null);

  // ── Hyperparam search state ──
  const [showSearchForm, setShowSearchForm] = useState(false);
  const [searchDatasetId, setSearchDatasetId] = useState("");
  const [searchGridText, setSearchGridText] = useState(DEFAULT_SEARCH_GRID);
  const [searchBaseHpText, setSearchBaseHpText] = useState(
    JSON.stringify({ obs_len: 60, pred_len: 10, epochs: 50, batch_size: 32, lr: 0.001, feature_cols: ["close"], normalize: "zscore", val_split: 0.2 }, null, 2)
  );
  const [startingSearch, setStartingSearch] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchResult, setSearchResult] = useState<{ run_ids: number[] } | null>(null);

  // ── Validate state ──
  const [showValidateForm, setShowValidateForm] = useState(false);
  const [validateRunId, setValidateRunId] = useState("");
  const [validateDatasetId, setValidateDatasetId] = useState("");
  const [validating, setValidating] = useState(false);
  const [validateError, setValidateError] = useState<string | null>(null);

  // ── Actions ──

  function selectDataset(newDatasetId: string) {
    setDatasetId(newDatasetId);
    setPreprocessedDatasetId(""); // recipe list is scoped to the chosen dataset
  }

  async function startTraining() {
    setTrainError(null);
    if (!datasetId) { setTrainError("Select a dataset"); return; }
    if (!preprocessedDatasetId) { setTrainError("Select a preprocessed dataset (or create one)"); return; }
    setStartingRun(true);
    try {
      const res = await apiFetch(`/api/v1/models/${id}/training-runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preprocessed_dataset_id: parseInt(preprocessedDatasetId),
          hyperparams: trainingParams,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setTrainError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      setShowTrainForm(false);
      toast("Training run queued", "success");
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
      const res = await apiFetch(`/api/v1/models/${id}/deploy?training_run_id=${runId}`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setDeployError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      toast("Model deployed", "success");
      mutate(`/api/v1/models/${id}`);
    } finally {
      setDeploying(null);
    }
  }

  async function startSearch() {
    setSearchError(null);
    setSearchResult(null);
    let grid: Record<string, unknown[]>;
    let baseHp: Record<string, unknown>;
    try {
      grid = JSON.parse(searchGridText);
      baseHp = JSON.parse(searchBaseHpText);
    } catch {
      setSearchError("Search grid or base hyperparams is not valid JSON");
      return;
    }
    if (!searchDatasetId) { setSearchError("Select a dataset"); return; }
    setStartingSearch(true);
    try {
      const res = await apiFetch("/api/v1/training-runs/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: parseInt(id), dataset_id: parseInt(searchDatasetId), search_grid: { ...baseHp, ...grid } }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setSearchError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      const body = await res.json();
      const result = body.data ?? body;
      setSearchResult(result);
      toast(`${result.run_ids?.length ?? "?"} training runs queued`, "success");
      mutate(`/api/v1/models/${id}/training-runs`);
    } finally {
      setStartingSearch(false);
    }
  }

  async function runValidation() {
    setValidateError(null);
    if (!validateRunId || !validateDatasetId) { setValidateError("Select a training run and dataset"); return; }
    setValidating(true);
    try {
      const res = await apiFetch(`/api/v1/models/${id}/validations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ training_run_id: parseInt(validateRunId), dataset_id: parseInt(validateDatasetId) }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setValidateError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      setShowValidateForm(false);
      toast("Validation queued", "success");
      mutate(`/api/v1/models/${id}/validations`);
    } finally {
      setValidating(false);
    }
  }

  // ── Render ──

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
        <a href="/model" className="text-sm text-gray-400 hover:text-white">← Back</a>
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
          <div className="mb-4 rounded border border-gray-700 bg-gray-900 p-4 space-y-4">
            {/* Dataset selector */}
            <div>
              <label className="mb-1 block text-xs text-gray-400">Dataset</label>
              <select
                value={datasetId}
                onChange={(e) => selectDataset(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                <option value="">Select dataset…</option>
                {datasets?.map((d: any) => (
                  <option key={d.id} value={d.id}>{d.name} ({d.row_count} rows)</option>
                ))}
              </select>
            </div>

            {/* Preprocessed dataset (recipe) selector */}
            {datasetId && (
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <label className="block text-xs text-gray-400">Preprocessed Dataset</label>
                  <a
                    href={`/data/preprocessed/new?dataset_id=${datasetId}`}
                    className="text-xs text-brand-500 hover:underline"
                  >
                    + New recipe
                  </a>
                </div>
                {preprocessedDatasets && preprocessedDatasets.length === 0 ? (
                  <p className="text-xs text-gray-500">
                    No preprocessing recipes for this dataset yet —{" "}
                    <a href={`/data/preprocessed/new?dataset_id=${datasetId}`} className="text-brand-500 hover:underline">create one</a>.
                  </p>
                ) : (
                  <select
                    value={preprocessedDatasetId}
                    onChange={(e) => setPreprocessedDatasetId(e.target.value)}
                    className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                  >
                    <option value="">Select recipe…</option>
                    {preprocessedDatasets?.map((r: any) => (
                      <option key={r.id} value={r.id}>
                        {r.name} — {summarizePreprocessing(r.preprocessing)}, {r.normalize}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}

            {/* Training hyperparameters */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {(
                [
                  { key: "obs_len",    label: "Observation Length", step: 1, min: 1 },
                  { key: "pred_len",   label: "Prediction Length",  step: 1, min: 1 },
                  { key: "epochs",     label: "Epochs",             step: 1, min: 1 },
                  { key: "batch_size", label: "Batch Size",         step: 1, min: 1 },
                  { key: "lr",         label: "Learning Rate",      step: 0.0001, min: 0 },
                  { key: "val_split",  label: "Validation Split",   step: 0.05, min: 0.05, max: 0.5 },
                ] as { key: keyof TrainingParams; label: string; step: number; min?: number; max?: number }[]
              ).map(({ key, label, step, min, max }) => (
                <div key={key}>
                  <label className="mb-1 block text-xs text-gray-400">{label}</label>
                  <input
                    type="number"
                    step={step}
                    min={min}
                    max={max}
                    value={trainingParams[key] as number}
                    onChange={(e) => setTrainingParams((p) => ({ ...p, [key]: Number(e.target.value) }))}
                    className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                  />
                </div>
              ))}
            </div>

            {trainError && <p className="text-xs text-red-400">{trainError}</p>}

            <div className="flex gap-2 pt-1">
              <button
                onClick={startTraining}
                disabled={startingRun}
                className="rounded bg-brand-500 px-3 py-1.5 text-xs text-white hover:bg-sky-400 disabled:opacity-50"
              >
                {startingRun ? "Queuing…" : "Start Training"}
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
          <div className="rounded-lg border border-dashed border-gray-700 px-6 py-10 text-center">
            <p className="text-gray-300 font-medium text-sm mb-1">No training runs yet</p>
            <p className="text-gray-500 text-xs">Use the Train button above to start the first run.</p>
          </div>
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
                  <td className="py-2 pr-4"><StatusBadge status={run.status} /></td>
                  <td className="py-2 pr-4 text-gray-300">
                    {run.status === "running"
                      ? `Epoch ${run.current_epoch}${run.eta_seconds ? ` · ~${Math.round(run.eta_seconds / 60)}m left` : ""}`
                      : run.best_epoch != null ? `Best epoch ${run.best_epoch}` : "—"}
                  </td>
                  <td className="py-2 pr-4 text-gray-300">{run.val_loss != null ? run.val_loss.toFixed(6) : "—"}</td>
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

      {/* Hyperparameter Search */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-medium uppercase tracking-wide text-gray-400">Hyperparameter Search</h2>
          <button
            onClick={() => { setShowSearchForm(!showSearchForm); setSearchResult(null); setSearchError(null); }}
            className="rounded border border-gray-600 px-3 py-1 text-xs text-gray-300 hover:border-gray-400 hover:text-white"
          >
            {showSearchForm ? "Cancel" : "+ New Search"}
          </button>
        </div>

        {showSearchForm && (
          <div className="rounded border border-gray-700 bg-gray-900 p-4 space-y-3">
            <p className="text-xs text-gray-400">
              Sweep multiple runs. Fixed values go in <em>Base Hyperparams</em> (JSON); arrays to sweep go in <em>Search Grid</em>.
              Include a <code className="text-gray-300">preprocessing</code> key in Base Hyperparams to use indicators or clustering across all runs.
            </p>
            <div>
              <label className="mb-1 block text-xs text-gray-400">Dataset</label>
              <select
                value={searchDatasetId}
                onChange={(e) => setSearchDatasetId(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                <option value="">Select dataset…</option>
                {datasets?.map((d: any) => (
                  <option key={d.id} value={d.id}>{d.name} ({d.row_count} rows)</option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="mb-1 block text-xs text-gray-400">Search Grid <span className="text-gray-500">(values to sweep)</span></label>
                <textarea
                  value={searchGridText}
                  onChange={(e) => setSearchGridText(e.target.value)}
                  rows={6}
                  className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 font-mono text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                />
              </div>
              <div>
                <label className="mb-1 block text-xs text-gray-400">Base Hyperparams <span className="text-gray-500">(fixed)</span></label>
                <textarea
                  value={searchBaseHpText}
                  onChange={(e) => setSearchBaseHpText(e.target.value)}
                  rows={6}
                  className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 font-mono text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                />
              </div>
            </div>
            {(() => {
              try {
                const grid = JSON.parse(searchGridText);
                const combos = Object.values(grid).reduce<number>((acc, vals) => acc * (vals as unknown[]).length, 1);
                return <p className="text-xs text-gray-500">{combos} run{combos !== 1 ? "s" : ""} will be queued.</p>;
              } catch { return null; }
            })()}
            {searchError && <p className="text-xs text-red-400">{searchError}</p>}
            {searchResult && (
              <p className="text-xs text-green-400">
                Queued {searchResult.run_ids.length} runs: #{searchResult.run_ids.join(", #")}
                {" — "}
                <a href="/model/compare" className="underline hover:text-green-300">Compare on results page →</a>
              </p>
            )}
            <button
              onClick={startSearch}
              disabled={startingSearch}
              className="rounded bg-brand-500 px-3 py-1.5 text-xs text-white hover:bg-sky-400 disabled:opacity-50"
            >
              {startingSearch ? "Queuing…" : "Start Search"}
            </button>
          </div>
        )}
      </section>

      {/* Validations */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-medium uppercase tracking-wide text-gray-400">Validations</h2>
          {runs && runs.some((r: any) => r.status === "completed") && (
            <button
              onClick={() => setShowValidateForm(!showValidateForm)}
              className="rounded border border-gray-600 px-3 py-1 text-xs text-gray-300 hover:border-gray-400 hover:text-white"
            >
              + Run Validation
            </button>
          )}
        </div>

        {showValidateForm && (
          <div className="mb-4 rounded border border-gray-700 bg-gray-900 p-4 space-y-3">
            <div>
              <label className="mb-1 block text-xs text-gray-400">Training Run</label>
              <select
                value={validateRunId}
                onChange={(e) => setValidateRunId(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                <option value="">Select run…</option>
                {runs?.filter((r: any) => r.status === "completed").map((r: any) => (
                  <option key={r.id} value={r.id}>Run #{r.id} · val_loss {r.val_loss?.toFixed(6) ?? "—"}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs text-gray-400">Validation Dataset</label>
              <select
                value={validateDatasetId}
                onChange={(e) => setValidateDatasetId(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                <option value="">Select dataset…</option>
                {datasets?.map((d: any) => (
                  <option key={d.id} value={d.id}>{d.name} ({d.row_count} rows)</option>
                ))}
              </select>
            </div>
            {validateError && <p className="text-xs text-red-400">{validateError}</p>}
            <div className="flex gap-2">
              <button
                onClick={runValidation}
                disabled={validating}
                className="rounded bg-brand-500 px-3 py-1.5 text-xs text-white hover:bg-sky-400 disabled:opacity-50"
              >
                {validating ? "Running…" : "Run"}
              </button>
              <button
                onClick={() => setShowValidateForm(false)}
                className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-400 hover:text-white"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {validations && validations.length === 0 && <p className="text-sm text-gray-500">No validations yet.</p>}
        {validations && validations.length > 0 && (
          <div className="space-y-3">
            {validations.map((v: any) => (
              <div key={v.id} className="rounded border border-gray-700 bg-gray-900 p-3">
                <p className="mb-2 text-xs text-gray-500">
                  Run #{v.training_run_id} · Dataset {v.dataset_id} · {new Date(v.computed_at).toLocaleString()}
                </p>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {Object.entries(v.metrics as Record<string, number>).map(([key, val]) => (
                    <div key={key} className="rounded bg-gray-800 px-3 py-2">
                      <p className="text-xs text-gray-500">{key.replace(/_/g, " ")}</p>
                      <p className="text-sm font-mono text-white">
                        {typeof val === "number" ? val.toFixed(4) : String(val)}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
