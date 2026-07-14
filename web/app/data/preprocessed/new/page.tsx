"use client";

import { Suspense, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import useSWR from "swr";
import { apiFetch, fetcher } from "@/lib/fetcher";
import { useToast } from "@/lib/toast";
import {
  IndicatorType,
  IndicatorCfg,
  ClusteringCfg,
  INDICATOR_DEFAULTS,
  INDICATOR_LABELS,
  getOutputCols,
  getAllAvailableCols,
  BASE_COLS,
  DEFAULT_CLUSTERING,
} from "@/lib/preprocessing";

function NewPreprocessedDatasetContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { toast } = useToast();

  const { data: datasets } = useSWR("/api/v1/datasets?page_size=1000", fetcher);

  const [datasetId, setDatasetId] = useState(searchParams.get("dataset_id") ?? "");
  const [name, setName] = useState("");
  const [normalize, setNormalize] = useState("zscore");

  const [indicators, setIndicators] = useState<IndicatorCfg[]>([]);
  const [clustering, setClustering] = useState<ClusteringCfg>(DEFAULT_CLUSTERING);
  const [featureCols, setFeatureCols] = useState<string[]>(["close"]);

  const [newIndType, setNewIndType] = useState<IndicatorType>("rsi");
  const [newIndParams, setNewIndParams] = useState<Record<string, number>>(INDICATOR_DEFAULTS["rsi"] as Record<string, number>);
  const [showNewInd, setShowNewInd] = useState(false);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const availableCols = getAllAvailableCols(indicators, clustering);

  function toggleFeatureCol(col: string) {
    setFeatureCols((prev) => (prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]));
  }

  function addIndicator() {
    const cfg: IndicatorCfg = { type: newIndType, ...newIndParams } as IndicatorCfg;
    setIndicators((prev) => [...prev, cfg]);
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

  async function handleSubmit() {
    setError(null);
    if (!datasetId) { setError("Select a dataset"); return; }
    if (!name.trim()) { setError("Give this recipe a name"); return; }
    if (!featureCols.length) { setError("Select at least one feature column"); return; }
    setSubmitting(true);
    try {
      const res = await apiFetch("/api/v1/preprocessed-datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name.trim(),
          dataset_id: parseInt(datasetId),
          preprocessing: { indicators, clustering: clustering.enabled ? clustering : undefined },
          feature_cols: featureCols,
          normalize,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      const body = await res.json();
      const created = body.data ?? body;
      toast("Preprocessed dataset created — computing characteristics…", "success");
      router.push(`/data/preprocessed/${created.id}`);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <a href="/data/preprocessed" className="text-xs text-gray-500 hover:text-white">← Preprocessed Datasets</a>
        <h1 className="mt-1 text-2xl font-semibold text-white">New Preprocessed Dataset</h1>
        <p className="mt-1 text-sm text-gray-400">
          Save a preprocessing recipe once, then pick it from a list every time you start a
          training run instead of re-configuring indicators/features/normalization.
        </p>
      </div>

      <div className="rounded border border-gray-700 bg-gray-900 p-4 space-y-4">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="mb-1 block text-xs text-gray-400">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. USDJPY + RSI/MACD zscore"
              className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-gray-400">Dataset</label>
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            >
              <option value="">Select dataset…</option>
              {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
              {datasets?.map((d: any) => (
                <option key={d.id} value={d.id}>{d.name} ({d.row_count} rows)</option>
              ))}
            </select>
          </div>
        </div>

        {/* Technical indicators */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-300">Technical Indicators</span>
            <button onClick={() => setShowNewInd(!showNewInd)} className="text-xs text-brand-500 hover:underline">+ Add</button>
          </div>

          {indicators.length === 0 && !showNewInd && (
            <p className="text-xs text-gray-500">No indicators added. Click + Add to compute features from price data.</p>
          )}

          {indicators.map((ind, i) => (
            <div key={i} className="flex items-center justify-between rounded bg-gray-800 px-3 py-2 mb-1.5">
              <div>
                <span className="text-xs font-medium text-white">{INDICATOR_LABELS[ind.type]}</span>
                <span className="ml-2 text-xs text-gray-400">
                  {Object.entries(ind).filter(([k]) => k !== "type").map(([k, v]) => `${k}=${v}`).join(" ")}
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
                      setFeatureCols((fc) => fc.map((c) => (c === `cluster_${old}` ? `cluster_${n}` : c)));
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

        {/* Feature columns */}
        <div>
          <span className="text-xs font-medium text-gray-300">Feature Columns</span>
          <p className="mt-1 text-xs text-gray-500">Select which columns feed into the model. Add indicators above to unlock more columns.</p>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-2">
            {availableCols.map((col) => (
              <label key={col} className="flex items-center gap-1.5 cursor-pointer">
                <input type="checkbox" checked={featureCols.includes(col)} onChange={() => toggleFeatureCol(col)} className="accent-brand-500" />
                <span className="text-xs text-gray-200">{col}</span>
              </label>
            ))}
          </div>
          {featureCols.length > 0 && (
            <p className="mt-2 text-xs text-gray-500">
              Selected ({featureCols.length}): <span className="text-gray-300">{featureCols.join(", ")}</span>
            </p>
          )}
        </div>

        {/* Normalize */}
        <div className="max-w-xs">
          <label className="mb-1 block text-xs text-gray-400">Normalization</label>
          <select
            value={normalize}
            onChange={(e) => setNormalize(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
          >
            <option value="zscore">Z-Score (recommended)</option>
            <option value="returns">Log Returns</option>
            <option value="minmax">Min-Max [0, 1]</option>
            <option value="robust">Robust (median/IQR)</option>
            <option value="none">None</option>
          </select>
        </div>

        {error && <p className="text-xs text-red-400">{error}</p>}

        <button
          onClick={handleSubmit}
          disabled={submitting}
          className="rounded bg-brand-500 px-4 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-50"
        >
          {submitting ? "Creating…" : "Create"}
        </button>
      </div>
    </div>
  );
}

export default function NewPreprocessedDatasetPage() {
  return (
    <Suspense fallback={<div className="p-8 text-gray-400 text-sm">Loading…</div>}>
      <NewPreprocessedDatasetContent />
    </Suspense>
  );
}
