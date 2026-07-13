"use client";

import { useState } from "react";
import useSWR, { mutate } from "swr";
import { fetcher, apiFetch } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";
import { useToast } from "@/lib/toast";

export default function ModelListPage() {
  const { data: models, isLoading } = useSWR("/api/v1/models", fetcher);
  const { toast } = useToast();
  const [deletingId, setDeletingId] = useState<number | null>(null);

  async function deleteModel(id: number, name: string) {
    if (!confirm(`Delete model "${name}"? This will remove all training runs and cannot be undone.`)) return;
    setDeletingId(id);
    try {
      const res = await apiFetch(`/api/v1/models/${id}`, { method: "DELETE" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        toast(body.error?.message ?? body.detail ?? `Error ${res.status}`, "error");
        return;
      }
      toast(`Model "${name}" deleted`, "success");
      mutate("/api/v1/models");
    } finally {
      setDeletingId(null);
    }
  }

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
                    <div className="flex items-center justify-end gap-3">
                      <a href={`/model/${m.id}`} className="text-xs text-brand-400 hover:text-brand-300 font-medium">
                        Open →
                      </a>
                      <button
                        onClick={() => deleteModel(m.id, m.name)}
                        disabled={deletingId === m.id}
                        className="text-xs text-gray-500 hover:text-red-400 disabled:opacity-50"
                      >
                        {deletingId === m.id ? "Deleting…" : "Delete"}
                      </button>
                    </div>
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
