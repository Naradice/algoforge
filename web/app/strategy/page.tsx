"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";

export default function StrategyListPage() {
  const { data: strategies, isLoading } = useSWR("/api/v1/strategies", fetcher);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-white">Strategies</h1>
        <a href="/strategy/new" className="rounded bg-brand-500 px-3 py-1.5 text-sm text-white hover:bg-sky-400">
          New Strategy
        </a>
      </div>

      {isLoading && <p className="text-gray-400">Loading…</p>}

      {strategies && strategies.length === 0 && (
        <div className="rounded border border-gray-800 bg-gray-900 px-6 py-12 text-center">
          <p className="text-gray-400">No strategies yet.</p>
          <a href="/strategy/new" className="mt-3 inline-block text-sm text-brand-400 hover:underline">Create your first strategy →</a>
        </div>
      )}

      {strategies && strategies.length > 0 && (
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase">
              <th className="py-2 pr-4">Name</th>
              <th className="py-2 pr-4">Status</th>
              <th className="py-2 pr-4">Created</th>
              <th className="py-2" />
            </tr>
          </thead>
          <tbody>
            {strategies.map((s: any) => (
              <tr key={s.id} className="border-b border-gray-800/50 hover:bg-gray-900">
                <td className="py-2 pr-4">
                  <a href={`/strategy/${s.id}`} className="text-brand-500 hover:underline">
                    {s.name}
                  </a>
                </td>
                <td className="py-2 pr-4">
                  <StatusBadge status={s.status} />
                </td>
                <td className="py-2 pr-4 text-gray-400">
                  {new Date(s.created_at).toLocaleDateString()}
                </td>
                <td className="py-2 text-right">
                  <a href={`/strategy/${s.id}`} className="text-xs text-gray-400 hover:text-white">
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

