"use client";

import { useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import useSWR, { mutate } from "swr";
import { apiFetch, fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";
import { useToast } from "@/lib/toast";

// ── Types ──────────────────────────────────────────────────────────────────

type IndicatorType = "sma" | "ema" | "rsi" | "macd" | "bbands" | "atr" | "returns" | "volatility";

interface IndicatorCfg {
  type: IndicatorType;
  period?: number;
  fast?: number;
  slow?: number;
  signal?: number;
  std?: number;
}

interface ClusteringCfg {
  enabled: boolean;
  n_clusters: number;
  on_cols: string[];
}

interface TrainingParams {
  obs_len: number;
  pred_len: number;
  epochs: number;
  batch_size: number;
  lr: number;
  normalize: string;
  val_split: number;
}

// ── Indicator helpers ──────────────────────────────────────────────────────

const INDICATOR_DEFAULTS: Record<IndicatorType, Partial<IndicatorCfg>> = {
  sma:        { period: 20 },
  ema:        { period: 20 },
  rsi:        { period: 14 },
  macd:       { fast: 12, slow: 26, signal: 9 },
  bbands:     { period: 20, std: 2 },
  atr:        { period: 14 },
  returns:    { period: 1 },
  volatility: { period: 20 },
};

const INDICATOR_LABELS: Record<IndicatorType, string> = {
  sma: "SMA", ema: "EMA", rsi: "RSI", macd: "MACD",
  bbands: "Bollinger Bands", atr: "ATR", returns: "Returns", volatility: "Volatility",
};

function getOutputCols(cfg: IndicatorCfg): string[] {
  const p = cfg.period;
  switch (cfg.type) {
    case "sma":        return [`sma_${p ?? 20}`];
    case "ema":        return [`ema_${p ?? 20}`];
    case "rsi":        return [`rsi_${p ?? 14}`];
    case "macd":       return ["macd", "macd_signal", "macd_hist"];
    case "bbands":     return [`bb_upper_${p ?? 20}`, `bb_mid_${p ?? 20}`, `bb_lower_${p ?? 20}`, `bb_width_${p ?? 20}`];
    case "atr":        return [`atr_${p ?? 14}`];
    case "returns":    return [`returns_${p ?? 1}`];
    case "volatility": return [`vol_${p ?? 20}`];
    default:           return [];
  }
}

const BASE_COLS = ["open", "high", "low", "close", "volume"];

function getAllAvailableCols(indicators: IndicatorCfg[], clustering: ClusteringCfg): string[] {
  const cols = [...BASE_COLS];
  for (const ind of indicators) cols.push(...getOutputCols(ind));
  if (clustering.enabled) cols.push(`cluster_${clustering.n_clusters}`);
  return [...new Set(cols)];
}

// ── Defaults ───────────────────────────────────────────────────────────────

const DEFAULT_TRAINING: TrainingParams = {
  obs_len: 60, pred_len: 10, epochs: 50, batch_size: 32,
  lr: 0.001, normalize: "zscore", val_split: 0.2,
};

const DEFAULT_CLUSTERING: ClusteringCfg = { enabled: false, n_clusters: 5, on_cols: ["close"] };

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
  const [activeTab, setActiveTab] = useState<"preprocessing" | "features" | "training">("preprocessing");

  const [indicators, setIndicators] = useState<IndicatorCfg[]>([]);
  const [clustering, setClustering] = useState<ClusteringCfg>(DEFAULT_CLUSTERING);
  const [featureCols, setFeatureCols] = useState<string[]>(["close"]);
  const [trainingParams, setTrainingParams] = useState<TrainingParams>(DEFAULT_TRAINING);

  // new indicator mini-form
  const [newIndType, setNewIndType] = useState<IndicatorType>("rsi");
  const [newIndParams, setNewIndParams] = useState<Record<string, number>>(INDICATOR_DEFAULTS["rsi"] as Record<string, number>);
  const [showNewInd, setShowNewInd] = useState(false);

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

  // ── Derived ──

  const availableCols = getAllAvailableCols(indicators, clustering);

  const toggleFeatureCol = useCallback((col: string) => {
    setFeatureCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  }, []);

  function addIndicator() {
    const cfg: IndicatorCfg = { type: newIndType, ...newIndParams } as IndicatorCfg;
    setIndicators((prev) => [...prev, cfg]);
    // auto-select output columns
    setFeatureCols((prev) => [...new Set([...prev, ...getOutputCols(cfg)])]);
    setShowNewInd(false);
  }

  function removeIndicator(idx: number) {
    const removed = indicators[idx];
    const removedCols = getOutputCols(removed);
    const remaining = indicators.filter((_, i) => i !== idx);
    const kept = new Set([
      ...BASE_COLS,
      ...remaining.flatMap(getOutputCols),
      ...(clustering.enabled ? [`cluster_${clustering.n_clusters}`] : []),
    ]);
    setIndicators(remaining);
    setFeatureCols((fc) => fc.filter((c) => !removedCols.includes(c) || kept.has(c)));
  }

  function changeNewIndType(t: IndicatorType) {
    setNewIndType(t);
    setNewIndParams(INDICATOR_DEFAULTS[t] as Record<string, number>);
  }

  function buildHyperparams() {
    return {
      ...trainingParams,
      feature_cols: featureCols.length ? featureCols : ["close"],
      preprocessing: {
        indicators,
        clustering: clustering.enabled ? clustering : undefined,
      },
    };
  }

  // ── Actions ──

  async function startTraining() {
    setTrainError(null);
    if (!datasetId) { setTrainError("Select a dataset"); return; }
    if (!featureCols.length) { setTrainError("Select at least one feature column"); return; }
    setStartingRun(true);
    try {
      const res = await apiFetch(`/api/v1/models/${id}/training-runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: parseInt(datasetId), hyperparams: buildHyperparams() }),
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
                onChange={(e) => setDatasetId(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                <option value="">Select dataset…</option>
                {datasets?.map((d: any) => (
                  <option key={d.id} value={d.id}>{d.name} ({d.row_count} rows)</option>
                ))}
              </select>
            </div>

            {/* Tabs */}
            <div>
              <div className="flex gap-1 border-b border-gray-700 mb-4">
                {(["preprocessing", "features", "training"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-3 py-1.5 text-xs font-medium capitalize transition-colors ${
                      activeTab === tab
                        ? "border-b-2 border-brand-500 text-white"
                        : "text-gray-500 hover:text-gray-300"
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              {/* ── Preprocessing tab ── */}
              {activeTab === "preprocessing" && (
                <div className="space-y-4">
                  {/* Indicator list */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs font-medium text-gray-300">Technical Indicators</span>
                      <button
                        onClick={() => setShowNewInd(!showNewInd)}
                        className="text-xs text-brand-500 hover:underline"
                      >
                        + Add
                      </button>
                    </div>

                    {indicators.length === 0 && !showNewInd && (
                      <p className="text-xs text-gray-500">No indicators added. Click + Add to compute features from price data.</p>
                    )}

                    {indicators.map((ind, i) => (
                      <div key={i} className="flex items-center justify-between rounded bg-gray-800 px-3 py-2 mb-1.5">
                        <div>
                          <span className="text-xs font-medium text-white">{INDICATOR_LABELS[ind.type]}</span>
                          <span className="ml-2 text-xs text-gray-400">
                            {Object.entries(ind)
                              .filter(([k]) => k !== "type")
                              .map(([k, v]) => `${k}=${v}`)
                              .join(" ")}
                          </span>
                          <span className="ml-2 text-xs text-gray-600">→ {getOutputCols(ind).join(", ")}</span>
                        </div>
                        <button onClick={() => removeIndicator(i)} className="text-xs text-gray-500 hover:text-red-400">✕</button>
                      </div>
                    ))}

                    {showNewInd && (
                      <div className="rounded border border-gray-700 bg-gray-800 p-3 space-y-3">
                        <div className="flex gap-2 flex-wrap">
                          <div>
                            <label className="mb-1 block text-xs text-gray-400">Type</label>
                            <select
                              value={newIndType}
                              onChange={(e) => changeNewIndType(e.target.value as IndicatorType)}
                              className="rounded border border-gray-600 bg-gray-700 px-2 py-1 text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                            >
                              {(Object.keys(INDICATOR_LABELS) as IndicatorType[]).map((t) => (
                                <option key={t} value={t}>{INDICATOR_LABELS[t]}</option>
                              ))}
                            </select>
                          </div>
                          {Object.entries(newIndParams).map(([param, val]) => (
                            <div key={param}>
                              <label className="mb-1 block text-xs text-gray-400">{param}</label>
                              <input
                                type="number"
                                value={val}
                                onChange={(e) => setNewIndParams((p) => ({ ...p, [param]: Number(e.target.value) }))}
                                className="w-20 rounded border border-gray-600 bg-gray-700 px-2 py-1 text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                              />
                            </div>
                          ))}
                        </div>
                        <div className="text-xs text-gray-500">
                          Produces: <span className="text-gray-300">{getOutputCols({ type: newIndType, ...newIndParams } as IndicatorCfg).join(", ")}</span>
                        </div>
                        <div className="flex gap-2">
                          <button onClick={addIndicator} className="rounded bg-brand-500 px-3 py-1 text-xs text-white hover:bg-sky-400">Add</button>
                          <button onClick={() => setShowNewInd(false)} className="text-xs text-gray-500 hover:text-white">Cancel</button>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Clustering */}
                  <div>
                    <span className="text-xs font-medium text-gray-300">Clustering</span>
                    <div className="mt-2 rounded bg-gray-800 p-3 space-y-3">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={clustering.enabled}
                          onChange={(e) => {
                            const enabled = e.target.checked;
                            setClustering((c) => ({ ...c, enabled }));
                            if (!enabled) setFeatureCols((fc) => fc.filter((c) => c !== `cluster_${clustering.n_clusters}`));
                            else setFeatureCols((fc) => [...new Set([...fc, `cluster_${clustering.n_clusters}`])]);
                          }}
                          className="accent-brand-500"
                        />
                        <span className="text-xs text-white">Enable K-Means clustering</span>
                      </label>
                      {clustering.enabled && (
                        <div className="space-y-2 pl-4">
                          <div className="flex items-center gap-3">
                            <label className="text-xs text-gray-400 w-20">K clusters</label>
                            <input
                              type="number"
                              min={2}
                              max={20}
                              value={clustering.n_clusters}
                              onChange={(e) => {
                                const old = clustering.n_clusters;
                                const n = Math.max(2, Number(e.target.value));
                                setClustering((c) => ({ ...c, n_clusters: n }));
                                setFeatureCols((fc) => fc.map((c) => c === `cluster_${old}` ? `cluster_${n}` : c));
                              }}
                              className="w-20 rounded border border-gray-600 bg-gray-700 px-2 py-1 text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                            />
                            <span className="text-xs text-gray-500">→ adds column <code className="text-gray-300">cluster_{clustering.n_clusters}</code></span>
                          </div>
                          <div>
                            <label className="mb-1 block text-xs text-gray-400">Cluster on columns</label>
                            <div className="flex flex-wrap gap-2">
                              {[...BASE_COLS, ...indicators.flatMap(getOutputCols)].map((col) => (
                                <label key={col} className="flex items-center gap-1 cursor-pointer">
                                  <input
                                    type="checkbox"
                                    checked={clustering.on_cols.includes(col)}
                                    onChange={(e) => {
                                      setClustering((c) => ({
                                        ...c,
                                        on_cols: e.target.checked ? [...c.on_cols, col] : c.on_cols.filter((x) => x !== col),
                                      }));
                                    }}
                                    className="accent-brand-500"
                                  />
                                  <span className="text-xs text-gray-300">{col}</span>
                                </label>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* ── Features tab ── */}
              {activeTab === "features" && (
                <div className="space-y-3">
                  <p className="text-xs text-gray-400">Select which columns feed into the model. Add indicators in the Preprocessing tab to unlock more columns.</p>
                  <div className="flex flex-wrap gap-x-4 gap-y-2">
                    {availableCols.map((col) => (
                      <label key={col} className="flex items-center gap-1.5 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={featureCols.includes(col)}
                          onChange={() => toggleFeatureCol(col)}
                          className="accent-brand-500"
                        />
                        <span className="text-xs text-gray-200">{col}</span>
                      </label>
                    ))}
                  </div>
                  {featureCols.length > 0 && (
                    <p className="text-xs text-gray-500">
                      Selected ({featureCols.length}): <span className="text-gray-300">{featureCols.join(", ")}</span>
                    </p>
                  )}
                </div>
              )}

              {/* ── Training tab ── */}
              {activeTab === "training" && (
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
                  <div>
                    <label className="mb-1 block text-xs text-gray-400">Normalization</label>
                    <select
                      value={trainingParams.normalize}
                      onChange={(e) => setTrainingParams((p) => ({ ...p, normalize: e.target.value }))}
                      className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                    >
                      <option value="zscore">Z-Score (recommended)</option>
                      <option value="returns">Log Returns</option>
                      <option value="minmax">Min-Max [0, 1]</option>
                      <option value="robust">Robust (median/IQR)</option>
                      <option value="none">None</option>
                    </select>
                  </div>
                </div>
              )}
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
