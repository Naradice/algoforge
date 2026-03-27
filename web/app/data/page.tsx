"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";

export default function DataPage() {
  const { data: datasources, isLoading } = useSWR("/api/v1/data/datasources", fetcher);
  const { data: datasets } = useSWR("/api/v1/data/datasets", fetcher);

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
        {datasources && (
          <div className="grid gap-3">
            {datasources.map((ds: any) => (
              <a
                key={ds.id}
                href={`/data/datasources/${ds.id}`}
                className="flex items-center justify-between rounded border border-gray-800 bg-gray-900 px-4 py-3 hover:border-brand-500"
              >
                <div>
                  <p className="font-medium text-white">{ds.name}</p>
                  <p className="text-xs text-gray-400">{ds.type}</p>
                </div>
                <span className="text-xs text-gray-500">{new Date(ds.created_at).toLocaleDateString()}</span>
              </a>
            ))}
          </div>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-sm font-medium text-gray-400 uppercase tracking-wide">
          Datasets ({datasets?.length ?? "…"})
        </h2>
        {datasets && (
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
                    <span className="rounded-full bg-gray-700 px-2 py-0.5 text-xs">{d.status}</span>
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
