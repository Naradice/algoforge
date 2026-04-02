"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";

export default function DataPage() {
  const { data: datasources, isLoading } = useSWR("/api/v1/datasources", fetcher);
  const { data: datasets } = useSWR("/api/v1/datasets", fetcher, {
    refreshInterval: (data) => data?.some?.((d: any) => d.status === "running") ? 3000 : 10000,
  });
  const { data: allJobs } = useSWR("/api/v1/collection-jobs?page_size=100", fetcher, {
    refreshInterval: (data) => data?.some?.((j: any) => j.status === "running") ? 3000 : 10000,
  });

  const jobByDatasource = new Map<number, { status: string; last_error?: string; last_run_at?: string }>();
  if (allJobs) {
    for (const job of allJobs as { datasource_id: number; status: string; last_error?: string; last_run_at?: string; id: number }[]) {
      if (!jobByDatasource.has(job.datasource_id)) jobByDatasource.set(job.datasource_id, job);
    }
  }

  return (
    <div className="space-y-8 max-w-5xl">
      {/* Header */}
      <div className="md-page-header">
        <h1 className="md-title-lg">Data Management</h1>
        <a href="/data/new" className="md-btn md-btn-primary">+ New Datasource</a>
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
        <h2 className="md-label-md">Datasets ({datasets?.length ?? "…"})</h2>

        {datasets && datasets.length === 0 && (
          <p className="md-body-md text-gray-400">No datasets yet. Run a collection job to generate one.</p>
        )}

        {datasets && datasets.length > 0 && (
          <div className="md-card overflow-hidden">
            <table className="md-table">
              <thead>
                <tr>
                  <th className="pl-5">Name</th>
                  <th>Symbol</th>
                  <th>Timeframe</th>
                  <th>Rows</th>
                  <th>Status</th>
                  <th className="pr-5">Created</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((d: any) => (
                  <tr key={d.id}>
                    <td className="pl-5">
                      <a href={`/data/datasets/${d.id}`} className="text-brand-400 hover:text-brand-300 font-medium hover:underline">
                        {d.name}
                      </a>
                    </td>
                    <td className="text-gray-300 font-mono text-xs">{d.symbol ?? "—"}</td>
                    <td>{d.timeframe ? <span className="md-chip">{d.timeframe}</span> : <span className="text-gray-500">—</span>}</td>
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
