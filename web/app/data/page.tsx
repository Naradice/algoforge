"use client";

import { useState, useMemo } from "react";
import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";

export default function DataPage() {
  const { data: datasources, isLoading } = useSWR("/api/v1/datasources?page_size=1000", fetcher);
  const { data: datasets } = useSWR("/api/v1/datasets?page_size=1000", fetcher, {
    refreshInterval: (data) => data?.some?.((d: any) => d.status === "running") ? 3000 : 10000,
  });
  const { data: allJobs } = useSWR("/api/v1/collection-jobs?page_size=100", fetcher, {
    refreshInterval: (data) => data?.some?.((j: any) => j.status === "running") ? 3000 : 10000,
  });

  // ── Filters ────────────────────────────────────────────────────────────────
  const [search, setSearch] = useState("");
  const [filterTimeframe, setFilterTimeframe] = useState("");
  const [filterSourceType, setFilterSourceType] = useState("");

  const jobByDatasource = new Map<number, { status: string; last_error?: string; last_run_at?: string }>();
  if (allJobs) {
    for (const job of allJobs as { datasource_id: number; status: string; last_error?: string; last_run_at?: string; id: number }[]) {
      if (!jobByDatasource.has(job.datasource_id)) jobByDatasource.set(job.datasource_id, job);
    }
  }

  // datasource_id → type lookup
  const dsTypeById = useMemo(() => {
    const m = new Map<number, string>();
    if (datasources) for (const ds of datasources as { id: number; type: string }[]) m.set(ds.id, ds.type);
    return m;
  }, [datasources]);

  // Derive available filter options from data
  const timeframeOptions = useMemo(() => {
    if (!datasets) return [];
    const seen = new Set<string>();
    for (const d of datasets as any[]) if (d.timeframe) seen.add(d.timeframe);
    return Array.from(seen).sort();
  }, [datasets]);

  const sourceTypeOptions = useMemo(() => {
    if (!datasets || !datasources) return [];
    const seen = new Set<string>();
    for (const d of datasets as any[]) {
      const t = dsTypeById.get(d.datasource_id);
      if (t) seen.add(t);
    }
    return Array.from(seen).sort();
  }, [datasets, datasources, dsTypeById]);

  // Apply filters
  const filteredDatasets = useMemo(() => {
    if (!datasets) return [];
    const q = search.trim().toLowerCase();
    return (datasets as any[]).filter((d) => {
      if (q && !d.name?.toLowerCase().includes(q) && !d.symbol?.toLowerCase().includes(q)) return false;
      if (filterTimeframe && d.timeframe !== filterTimeframe) return false;
      if (filterSourceType) {
        const t = dsTypeById.get(d.datasource_id);
        if (t !== filterSourceType) return false;
      }
      return true;
    });
  }, [datasets, search, filterTimeframe, filterSourceType, dsTypeById]);

  const hasFilters = search || filterTimeframe || filterSourceType;

  return (
    <div className="space-y-8 max-w-5xl">
      {/* Header */}
      <div className="md-page-header">
        <h1 className="md-title-lg">Data Management</h1>
        <div className="flex items-center gap-3">
          <a href="/data/preprocessed" className="md-btn-text">Preprocessed Datasets →</a>
          <a href="/data/new" className="md-btn md-btn-primary">+ New Datasource</a>
        </div>
      </div>

      {/* Datasources */}
      <section className="md-section">
        <h2 className="md-label-md">Datasources</h2>
        {isLoading && <p className="md-body-md">Loading…</p>}

        {datasources && datasources.length === 0 && (
          <div className="md-empty-state">
            <p className="text-gray-200 font-medium">No datasources yet.</p>
            <a href="/data/new" className="md-btn-text mt-2">Create your first datasource →</a>
          </div>
        )}

        {datasources && datasources.length > 0 && (
          <div className="space-y-2">
            {datasources.map((ds: any) => {
              const job = jobByDatasource.get(ds.id);
              return (
                <a key={ds.id} href={`/data/datasources/${ds.id}`}
                  className="md-list-item">
                  <div className="space-y-1 min-w-0">
                    <p className="md-label-lg">{ds.name}</p>
                    <div className="flex items-center gap-2">
                      <span className="md-chip">{ds.type}</span>
                      {job?.last_error && (
                        <span className="text-xs text-danger truncate max-w-xs">{job.last_error}</span>
                      )}
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2 shrink-0 ml-4">
                    {job && <StatusBadge status={job.status} />}
                    <span className="md-body-sm">{new Date(ds.created_at).toLocaleDateString()}</span>
                  </div>
                </a>
              );
            })}
          </div>
        )}
      </section>

      {/* Datasets */}
      <section className="md-section">
        <div className="flex items-center justify-between mb-3">
          <h2 className="md-label-md">
            Datasets{" "}
            <span className="text-gray-500 font-normal">
              ({hasFilters ? `${filteredDatasets.length} of ${datasets?.length ?? "…"}` : (datasets?.length ?? "…")})
            </span>
          </h2>
        </div>

        {/* Filter bar */}
        {datasets && datasets.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            <input
              type="search"
              placeholder="Search name or symbol…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="md-input text-sm flex-1 min-w-[160px] max-w-xs"
            />
            {timeframeOptions.length > 0 && (
              <select
                value={filterTimeframe}
                onChange={(e) => setFilterTimeframe(e.target.value)}
                className="md-input text-sm w-auto"
              >
                <option value="">All timeframes</option>
                {timeframeOptions.map((tf) => (
                  <option key={tf} value={tf}>{tf}</option>
                ))}
              </select>
            )}
            {sourceTypeOptions.length > 0 && (
              <select
                value={filterSourceType}
                onChange={(e) => setFilterSourceType(e.target.value)}
                className="md-input text-sm w-auto"
              >
                <option value="">All source types</option>
                {sourceTypeOptions.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            )}
            {hasFilters && (
              <button
                type="button"
                onClick={() => { setSearch(""); setFilterTimeframe(""); setFilterSourceType(""); }}
                className="text-xs text-gray-500 hover:text-white px-2"
              >
                Clear
              </button>
            )}
          </div>
        )}

        {datasets && datasets.length === 0 && (
          <p className="md-body-md text-gray-400">No datasets yet. Run a collection job to generate one.</p>
        )}

        {datasets && datasets.length > 0 && filteredDatasets.length === 0 && (
          <p className="md-body-md text-gray-400">No datasets match the current filters.</p>
        )}

        {filteredDatasets.length > 0 && (
          <div className="md-card overflow-hidden">
            <table className="md-table">
              <thead>
                <tr>
                  <th className="pl-5">Name</th>
                  <th>Symbol</th>
                  <th>Timeframe</th>
                  <th>Source type</th>
                  <th>Rows</th>
                  <th>Status</th>
                  <th className="pr-5">Created</th>
                </tr>
              </thead>
              <tbody>
                {filteredDatasets.map((d: any) => (
                  <tr key={d.id}>
                    <td className="pl-5">
                      <a href={`/data/datasets/${d.id}`} className="text-brand-400 hover:text-brand-300 font-medium hover:underline">
                        {d.name}
                      </a>
                    </td>
                    <td className="text-gray-300 font-mono text-xs">{d.symbol ?? "—"}</td>
                    <td>{d.timeframe ? <span className="md-chip">{d.timeframe}</span> : <span className="text-gray-500">—</span>}</td>
                    <td>
                      {dsTypeById.get(d.datasource_id)
                        ? <span className="md-chip">{dsTypeById.get(d.datasource_id)}</span>
                        : <span className="text-gray-500">—</span>}
                    </td>
                    <td className="tabular-nums text-gray-200">
                      {d.status === "running" ? (
                        <span className="flex items-center gap-1">
                          <span className="inline-block w-2 h-2 rounded-full bg-brand-400 animate-pulse" />
                          {d.row_count?.toLocaleString() ?? 0}
                        </span>
                      ) : (
                        d.row_count?.toLocaleString() ?? "—"
                      )}
                    </td>
                    <td><StatusBadge status={d.status} /></td>
                    <td className="pr-5 text-gray-400">{new Date(d.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
