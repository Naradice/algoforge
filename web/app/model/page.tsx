"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";

export default function ModelListPage() {
  const { data: models, isLoading } = useSWR("/api/v1/models", fetcher);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-white">ML Models</h1>
        <a href="/model/new" className="rounded bg-brand-500 px-3 py-1.5 text-sm text-white hover:bg-sky-400">
          New Model
        </a>
      </div>

      {isLoading && <p className="text-gray-400">Loading…</p>}

      {models && (
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase">
              <th className="py-2 pr-4">Name</th>
              <th className="py-2 pr-4">Architecture</th>
              <th className="py-2 pr-4">Status</th>
              <th className="py-2 pr-4">Created</th>
              <th className="py-2" />
            </tr>
          </thead>
          <tbody>
            {models.map((m: any) => (
              <tr key={m.id} className="border-b border-gray-800/50 hover:bg-gray-900">
                <td className="py-2 pr-4">
                  <a href={`/model/${m.id}`} className="text-brand-500 hover:underline">
                    {m.name}
                  </a>
                </td>
                <td className="py-2 pr-4 text-gray-300">{m.architecture}</td>
                <td className="py-2 pr-4">
                  <StatusBadge status={m.status} />
                </td>
                <td className="py-2 pr-4 text-gray-400">
                  {new Date(m.created_at).toLocaleDateString()}
                </td>
                <td className="py-2 text-right">
                  <a href={`/model/${m.id}`} className="text-xs text-gray-400 hover:text-white">
                    View →
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
