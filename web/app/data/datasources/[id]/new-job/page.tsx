"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";

export default function NewJobPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [scheduleCron, setScheduleCron] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/collection-jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          datasource_id: Number(id),
          schedule_cron: scheduleCron.trim() || null,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error?.message ?? body.detail ?? "Failed to create job");
      }
      router.push(`/data/datasources/${id}`);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="max-w-lg space-y-6">
      <div>
        <a href={`/data/datasources/${id}`} className="text-xs text-gray-500 hover:text-white">
          ← Datasource
        </a>
        <h1 className="mt-1 text-2xl font-semibold text-white">New Collection Job</h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-1">
          <label className="text-xs text-gray-400 uppercase">Schedule (cron, optional)</label>
          <input
            className="input w-full"
            value={scheduleCron}
            onChange={(e) => setScheduleCron(e.target.value)}
            placeholder="e.g. 0 * * * *  (leave blank for one-off)"
          />
          <p className="text-xs text-gray-500">Leave blank to create a one-off job you can trigger manually.</p>
        </div>

        {error && <p className="text-sm text-red-400">{error}</p>}

        <button
          type="submit"
          disabled={saving}
          className="w-full rounded bg-brand-500 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-40"
        >
          {saving ? "Creating…" : "Create Job"}
        </button>
      </form>

      <style>{`.input { @apply rounded border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white focus:border-brand-500 focus:outline-none; }`}</style>
    </div>
  );
}
