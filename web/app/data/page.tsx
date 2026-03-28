"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";

export default function DataPage() {
  const { data: datasources, isLoading } = useSWR("/api/v1/datasources", fetcher);
  const { data: datasets } = useSWR("/api/v1/datasets", fetcher);
  const { data: allJobs } = useSWR("/api/v1/collection-jobs?page_size=100", fetcher);

  // Build a map: datasource_id -> most recent job
  const jobByDatasource = new Map<number, { status: string; last_error?: string; last_run_at?: string }>();
  if (allJobs) {
    for (const job of allJobs as { datasource_id: number; status: string; last_error?: string; last_run_at?: string; id: number }[]) {
      if (!jobByDatasource.has(job.datasource_id)) {
        jobByDatasource.set(job.datasource_id, job);
      }
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-white">Data Management</h1>
        <a href="/data/new" className="rounded bg-brand-500 px-3 py-1.5 text-sm text-white hover:bg-sky-400">
          New Datasource
        </a>
      </div>

      <section className="space-y-2">
        <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide">Datasources</h2>
        {isLoading && <p className="text-gray-400">Loading…</p>}
        {datasources && datasources.length === 0 && (
          <div className="rounded border border-gray-800 bg-gray-900 px-6 py-10 text-center">
            <p className="text-gray-400">No datasources yet.</p>
            <a href="/data/new" className="mt-2 inline-block text-sm text-brand-400 hover:underline">Create your first datasource →</a>
          </div>
        )}
        {datasources && datasources.length > 0 && (
          <div className="grid gap-3">
            {datasources.map((ds: any) => {
              const job = jobByDatasource.get(ds.id);
              return (
                <a
                  key={ds.id}
                  href={`/data/datasources/${ds.id}`}
                  className="flex items-center justify-between rounded border border-gray-800 bg-gray-900 px-4 py-3 hover:border-brand-500"
                >
                  <div className="space-y-0.5">
                    <p className="font-medium text-white">{ds.name}</p>
                    <p className="text-xs text-gray-400">{ds.type}</p>
                    {job?.last_error && (
                      <p className="text-xs text-red-400 truncate max-w-xs">{job.last_error}</p>
                    )}
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    {job && (
                      <span className={`rounded-full px-2 py-0.5 text-xs ${
                        job.status === "idle" ? "bg-gray-700 text-gray-400" :
                        job.status === "running" ? "bg-blue-500/20 text-blue-400" :
                        job.status === "error" ? "bg-red-500/20 text-red-400" :
                        "bg-green-500/20 text-green-400"
                      }`}>
                        {job.status}
                      </span>
                    )}
                    <span className="text-xs text-gray-500">{new Date(ds.created_at).toLocaleDateString()}</span>
                  </div>
                </a>
              );
            })}
          </div>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
          Datasets ({datasets?.length ?? "…"})
        </h2>
        {datasets && datasets.length === 0 && (
          <p className="text-sm text-gray-500">No datasets yet. Run a collection job to generate one.</p>
        )}
        {datasets && datasets.length > 0 && (
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase">
                <th className="py-2 pr-4">Name</th>
                <th className="py-2 pr-4">Symbol</th>
                <th className="py-2 pr-4">Timeframe</th>
                <th className="py-2 pr-4">Rows</th>
                <th className="py-2 pr-4">Status</th>
              </tr>
            </thead>
            <tbody>
              {datasets.map((d: any) => (
                <tr key={d.id} className="border-b border-gray-800/50 hover:bg-gray-900">
                  <td className="py-2 pr-4">
                    <a href={`/data/datasets/${d.id}`} className="text-brand-500 hover:underline">
                      {d.name}
                    </a>
                  </td>
                  <td className="py-2 pr-4 text-gray-300">{d.symbol ?? "—"}</td>
                  <td className="py-2 pr-4 text-gray-300">{d.timeframe ?? "—"}</td>
                  <td className="py-2 pr-4 text-gray-300">{d.row_count?.toLocaleString() ?? "—"}</td>
                  <td className="py-2 pr-4">
                    <span className={`rounded-full px-2 py-0.5 text-xs ${d.status === "ready" ? "bg-green-500/20 text-green-400" : d.status === "error" ? "bg-red-500/20 text-red-400" : "bg-gray-700 text-gray-400"}`}>{d.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
