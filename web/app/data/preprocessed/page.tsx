"use client";

import { useMemo } from "react";
import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";
import { summarizePreprocessing } from "@/lib/preprocessing";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PreprocessedDataset = any;

export default function PreprocessedDatasetsPage() {
  const { data: recipes, isLoading } = useSWR<PreprocessedDataset[]>("/api/v1/preprocessed-datasets?page_size=200", fetcher, {
    refreshInterval: (data) => data?.some((r) => r.status === "pending") ? 3000 : 0,
  });
  const { data: datasets } = useSWR("/api/v1/datasets?page_size=1000", fetcher);

  const datasetNameById = useMemo(() => {
    const m = new Map<number, string>();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (datasets) for (const d of datasets as any[]) m.set(d.id, d.name);
    return m;
  }, [datasets]);

  return (
    <div className="space-y-8 max-w-5xl">
      <div className="md-page-header">
        <h1 className="md-title-lg">Preprocessed Datasets</h1>
        <a href="/data/preprocessed/new" className="md-btn md-btn-primary">+ New Recipe</a>
      </div>

      <p className="md-body-md text-gray-400">
        Named, reusable preprocessing recipes — indicators, clustering, feature columns, and
        normalization saved once and picked from a list when starting a training run, instead of
        re-configuring them every time.
      </p>

      <section className="md-section">
        {isLoading && <p className="md-body-md">Loading…</p>}

        {recipes && recipes.length === 0 && (
          <div className="md-empty-state">
            <p className="text-gray-200 font-medium">No preprocessed datasets yet.</p>
            <a href="/data/preprocessed/new" className="md-btn-text mt-2">Create your first recipe →</a>
          </div>
        )}

        {recipes && recipes.length > 0 && (
          <div className="md-card overflow-hidden">
            <table className="md-table">
              <thead>
                <tr>
                  <th className="pl-5">Name</th>
                  <th>Base Dataset</th>
                  <th>Preprocessing</th>
                  <th>Features</th>
                  <th>Normalize</th>
                  <th>Hurst</th>
                  <th>Status</th>
                  <th className="pr-5">Created</th>
                </tr>
              </thead>
              <tbody>
                {recipes.map((r) => {
                  const hurst = r.characteristics?.long_range_dependence?.hurst;
                  return (
                    <tr key={r.id}>
                      <td className="pl-5">
                        <a href={`/data/preprocessed/${r.id}`} className="text-brand-400 hover:text-brand-300 font-medium hover:underline">
                          {r.name}
                        </a>
                      </td>
                      <td>
                        <a href={`/data/datasets/${r.dataset_id}`} className="text-gray-300 hover:text-white hover:underline">
                          {datasetNameById.get(r.dataset_id) ?? `Dataset ${r.dataset_id}`}
                        </a>
                      </td>
                      <td className="text-gray-300 text-xs">{summarizePreprocessing(r.preprocessing)}</td>
                      <td className="text-gray-400 text-xs font-mono">{(r.feature_cols ?? []).join(", ")}</td>
                      <td><span className="md-chip">{r.normalize}</span></td>
                      <td className="tabular-nums text-gray-200">
                        {typeof hurst === "number" && !isNaN(hurst) ? hurst.toFixed(3) : "—"}
                      </td>
                      <td><StatusBadge status={r.status} /></td>
                      <td className="pr-5 text-gray-400">{new Date(r.created_at).toLocaleDateString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
