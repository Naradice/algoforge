"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";

export default function ModelListPage() {
  const { data: models, isLoading } = useSWR("/api/v1/models", fetcher);

  return (
    <div className="space-y-8 max-w-5xl">
      <div className="md-page-header">
        <h1 className="md-title-lg">ML Models</h1>
        <div className="flex gap-2">
          <a href="/model/compare" className="md-btn md-btn-outlined">Compare Runs</a>
          <a href="/model/new" className="md-btn md-btn-primary">+ New Model</a>
        </div>
      </div>

      {isLoading && <p className="md-body-md">Loading…</p>}

      {models && models.length === 0 && (
        <div className="md-empty-state">
          <p className="text-gray-200 font-medium">No models yet.</p>
          <a href="/model/new" className="md-btn-text mt-2">Create your first model →</a>
        </div>
      )}

      {models && models.length > 0 && (
        <div className="md-card overflow-hidden">
          <table className="md-table">
            <thead>
              <tr>
                <th className="pl-5">Name</th>
                <th>Architecture</th>
                <th>Status</th>
                <th>Created</th>
                <th className="pr-5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m: any) => (
                <tr key={m.id}>
                  <td className="pl-5">
                    <a href={`/model/${m.id}`} className="text-brand-400 hover:text-brand-300 font-medium hover:underline">
                      {m.name}
                    </a>
                  </td>
                  <td><span className="md-chip">{m.architecture}</span></td>
                  <td><StatusBadge status={m.status} /></td>
                  <td className="text-gray-400">{new Date(m.created_at).toLocaleDateString()}</td>
                  <td className="pr-5 text-right">
                    <a href={`/model/${m.id}`} className="text-xs text-brand-400 hover:text-brand-300 font-medium">
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
