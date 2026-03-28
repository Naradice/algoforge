"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";

export default function StrategyListPage() {
  const { data: strategies, isLoading } = useSWR("/api/v1/strategies", fetcher);

  return (
    <div className="space-y-8 max-w-5xl">
      <div className="md-page-header">
        <h1 className="md-title-lg">Strategies</h1>
        <a href="/strategy/new" className="md-btn md-btn-primary">+ New Strategy</a>
      </div>

      {isLoading && <p className="md-body-md">Loading…</p>}

      {strategies && strategies.length === 0 && (
        <div className="md-empty-state">
          <p className="text-gray-200 font-medium">No strategies yet.</p>
          <a href="/strategy/new" className="md-btn-text mt-2">Create your first strategy →</a>
        </div>
      )}

      {strategies && strategies.length > 0 && (
        <div className="md-card overflow-hidden">
          <table className="md-table">
            <thead>
              <tr>
                <th className="pl-5">Name</th>
                <th>Status</th>
                <th>Created</th>
                <th className="pr-5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((s: any) => (
                <tr key={s.id}>
                  <td className="pl-5">
                    <a href={`/strategy/${s.id}`} className="text-brand-400 hover:text-brand-300 font-medium hover:underline">
                      {s.name}
                    </a>
                    {s.description && <p className="md-body-sm mt-0.5">{s.description}</p>}
                  </td>
                  <td><StatusBadge status={s.status} /></td>
                  <td className="text-gray-400">{new Date(s.created_at).toLocaleDateString()}</td>
                  <td className="pr-5 text-right">
                    <a href={`/strategy/${s.id}`} className="text-xs text-brand-400 hover:text-brand-300 font-medium">
                      Open →
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
