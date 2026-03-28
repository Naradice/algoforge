"use client";

import { use } from "react";
import useSWR, { mutate } from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";

export default function DatasourceDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { data: ds } = useSWR(`/api/v1/datasources/${id}`, fetcher);
  const { data: jobs, isLoading } = useSWR(`/api/v1/collection-jobs?datasource_id=${id}`, fetcher);

  async function runJob(jobId: number) {
    await fetch(`/api/v1/collection-jobs/${jobId}/run`, { method: "POST" });
    mutate(`/api/v1/collection-jobs?datasource_id=${id}`);
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <a href="/data" className="text-xs text-gray-500 hover:text-white">← Data</a>
        <h1 className="mt-1 text-2xl font-semibold text-white">{ds?.name ?? "…"}</h1>
        <p className="text-xs text-gray-400">{ds?.type}</p>
      </div>

      {ds && (
        <section className="rounded border border-gray-800 bg-gray-900 p-4 space-y-2">
          <h2 className="text-xs font-medium text-gray-400 uppercase">Config</h2>
          <pre className="text-xs text-gray-300 overflow-x-auto">{JSON.stringify(ds.config, null, 2)}</pre>
        </section>
      )}

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide">Collection Jobs</h2>
          <a
            href={`/data/datasources/${id}/new-job`}
            className="rounded bg-brand-500 px-2 py-1 text-xs text-white hover:bg-sky-400"
          >
            New Job
          </a>
        </div>
        {isLoading && <p className="text-gray-400 text-sm">Loading…</p>}
        {jobs?.map((job: any) => (
          <div key={job.id} className="flex items-center justify-between rounded border border-gray-800 bg-gray-900 px-4 py-3">
            <div className="space-y-0.5">
              <div className="flex items-center gap-2">
                <StatusBadge status={job.status} />
                <span className="text-xs text-gray-400">{job.schedule_cron ?? "one-off"}</span>
              </div>
              {job.last_run_at && (
                <p className="text-xs text-gray-500">Last run: {new Date(job.last_run_at).toLocaleString()}</p>
              )}
              {job.last_error && (
                <p className="text-xs text-red-400 truncate max-w-sm">{job.last_error}</p>
              )}
            </div>
            <button
              onClick={() => runJob(job.id)}
              disabled={job.status === "running"}
              className="rounded bg-gray-700 px-3 py-1.5 text-xs text-white hover:bg-gray-600 disabled:opacity-40"
            >
              {job.status === "running" ? "Running…" : "Run now"}
            </button>
          </div>
        ))}
      </section>
    </div>
  );
}
