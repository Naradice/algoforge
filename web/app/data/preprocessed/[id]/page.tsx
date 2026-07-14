"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import useSWR, { mutate } from "swr";
import { apiFetch, fetcher } from "@/lib/fetcher";
import { useToast } from "@/lib/toast";
import { StatusBadge } from "@/components/status-badge";
import { StructureStatGrid } from "@/components/structure-stat-grid";
import { summarizePreprocessing } from "@/lib/preprocessing";

export default function PreprocessedDatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { toast } = useToast();

  const url = `/api/v1/preprocessed-datasets/${id}`;
  const { data: recipe, isLoading } = useSWR(url, fetcher, {
    refreshInterval: (data) => (data?.status === "pending" ? 3000 : 0),
  });
  const { data: dataset } = useSWR(recipe ? `/api/v1/datasets/${recipe.dataset_id}` : null, fetcher);

  const [renaming, setRenaming] = useState(false);
  const [nameDraft, setNameDraft] = useState("");
  const [recomputing, setRecomputing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  async function saveRename() {
    if (!nameDraft.trim()) return;
    const res = await apiFetch(url, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: nameDraft.trim() }),
    });
    if (res.ok) {
      toast("Renamed", "success");
      mutate(url);
    }
    setRenaming(false);
  }

  async function recompute() {
    setActionError(null);
    setRecomputing(true);
    try {
      const res = await apiFetch(`${url}/characteristics/compute`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setActionError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      toast("Recomputing characteristics…", "success");
      mutate(url);
    } finally {
      setRecomputing(false);
    }
  }

  async function handleDelete() {
    if (!confirm(`Delete "${recipe.name}"? Training runs that already used it keep their own recorded characteristics.`)) return;
    setActionError(null);
    setDeleting(true);
    try {
      const res = await apiFetch(url, { method: "DELETE" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setActionError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      toast("Deleted", "success");
      router.push("/data/preprocessed");
    } finally {
      setDeleting(false);
    }
  }

  if (isLoading) return <p className="text-gray-400">Loading…</p>;
  if (!recipe) return <p className="text-red-400">Not found</p>;

  return (
    <div className="space-y-6 max-w-3xl">
      <div>
        <a href="/data/preprocessed" className="text-xs text-gray-500 hover:text-white">← Preprocessed Datasets</a>
        <div className="mt-1 flex items-center gap-3">
          {renaming ? (
            <input
              autoFocus
              value={nameDraft}
              onChange={(e) => setNameDraft(e.target.value)}
              onBlur={saveRename}
              onKeyDown={(e) => e.key === "Enter" && saveRename()}
              className="text-2xl font-semibold bg-gray-800 border border-gray-700 rounded px-2 text-white"
            />
          ) : (
            <h1
              className="text-2xl font-semibold text-white cursor-pointer hover:underline decoration-dashed"
              title="Click to rename"
              onClick={() => { setNameDraft(recipe.name); setRenaming(true); }}
            >
              {recipe.name}
            </h1>
          )}
          <StatusBadge status={recipe.status} />
        </div>
        <p className="mt-1 text-sm text-gray-400">
          Base dataset:{" "}
          <a href={`/data/datasets/${recipe.dataset_id}`} className="text-brand-400 hover:text-brand-300">
            {dataset?.name ?? `Dataset ${recipe.dataset_id}`}
          </a>
          {" · "}Created {new Date(recipe.created_at).toLocaleDateString()}
        </p>
      </div>

      <div className="rounded border border-gray-700 bg-gray-900 p-4 space-y-2">
        <h2 className="text-sm font-medium text-gray-300">Recipe</h2>
        <p className="text-xs text-gray-400 font-mono">
          indicators: {summarizePreprocessing(recipe.preprocessing)}
        </p>
        <p className="text-xs text-gray-400 font-mono">features: [{(recipe.feature_cols ?? []).join(", ")}]</p>
        <p className="text-xs text-gray-400 font-mono">normalize: {recipe.normalize}</p>
      </div>

      <div className="rounded border border-gray-700 bg-gray-900 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium text-gray-300">Data Characteristics</h2>
          <button
            onClick={recompute}
            disabled={recomputing || recipe.status === "pending"}
            className="text-xs text-brand-400 hover:text-brand-300 disabled:opacity-50"
          >
            {recomputing ? "Recomputing…" : "Recompute"}
          </button>
        </div>
        {recipe.status === "pending" && <p className="text-xs text-gray-500">Computing…</p>}
        {recipe.status === "error" && (
          <p className="text-xs text-red-400">{recipe.characteristics?.error ?? "Computation failed."}</p>
        )}
        {recipe.status === "ready" && <StructureStatGrid characteristics={recipe.characteristics} />}
      </div>

      {actionError && <p className="text-xs text-red-400">{actionError}</p>}

      <button
        onClick={handleDelete}
        disabled={deleting}
        className="rounded border border-red-800 px-3 py-1.5 text-xs text-red-400 hover:bg-red-950 disabled:opacity-50"
      >
        {deleting ? "Deleting…" : "Delete recipe"}
      </button>
    </div>
  );
}
