"use client";

import { useState } from "react";
import useSWR from "swr";
import { apiFetch, fetcher } from "@/lib/fetcher";

interface Webhook {
  id: number;
  url: string;
  events: string[];
  active: boolean;
  last_fired_at: string | null;
  last_status: number | null;
}

const ALL_EVENTS = [
  "strategy.signal",
  "trade.opened",
  "trade.closed",
  "strategy.run.completed",
  "strategy.error",
  "training.completed",
  "dataset.ready",
];

export default function WebhooksPage() {
  const { data: webhooks, mutate } = useSWR<Webhook[]>("/api/v1/webhooks", fetcher);
  const [showForm, setShowForm] = useState(false);
  const [url, setUrl] = useState("");
  const [secret, setSecret] = useState("");
  const [events, setEvents] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<number, string>>({});

  function toggleEvent(evt: string) {
    setEvents((prev) => prev.includes(evt) ? prev.filter((e) => e !== evt) : [...prev, evt]);
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!url || !secret || events.length === 0) {
      setError("URL, secret, and at least one event are required");
      return;
    }
    setSaving(true);
    try {
      const res = await apiFetch("/api/v1/webhooks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, events, secret }),
      });
      if (!res.ok) {
        const b = await res.json().catch(() => ({}));
        setError(b.error?.message ?? `Error ${res.status}`);
        return;
      }
      setShowForm(false);
      setUrl("");
      setSecret("");
      setEvents([]);
      mutate();
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: number) {
    await apiFetch(`/api/v1/webhooks/${id}`, { method: "DELETE" });
    mutate();
  }

  async function handleTest(id: number) {
    setTestResults((prev) => ({ ...prev, [id]: "Testing…" }));
    const res = await apiFetch(`/api/v1/webhooks/${id}/test`, { method: "POST" });
    const body = await res.json().catch(() => ({}));
    const data = body.data ?? body;
    setTestResults((prev) => ({ ...prev, [id]: `HTTP ${data.status_code}` }));
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-medium text-white">Webhook Registrations</h2>
        <button
          onClick={() => setShowForm((v) => !v)}
          className="rounded bg-brand-500 px-3 py-1.5 text-sm text-white hover:bg-sky-400"
        >
          {showForm ? "Cancel" : "Add Webhook"}
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleCreate} className="rounded border border-gray-700 bg-gray-900 p-4 space-y-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">URL</label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://your-server.com/webhook"
              required
              className="w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Secret</label>
            <input
              type="password"
              value={secret}
              onChange={(e) => setSecret(e.target.value)}
              placeholder="HMAC signing secret"
              required
              className="w-full rounded border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Events</label>
            <div className="grid grid-cols-2 gap-2">
              {ALL_EVENTS.map((evt) => (
                <label key={evt} className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
                  <input type="checkbox" checked={events.includes(evt)} onChange={() => toggleEvent(evt)} className="rounded border-gray-600" />
                  {evt}
                </label>
              ))}
            </div>
          </div>
          {error && <p className="text-sm text-red-400">{error}</p>}
          <button type="submit" disabled={saving} className="rounded bg-brand-500 px-4 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-50">
            {saving ? "Creating…" : "Create Webhook"}
          </button>
        </form>
      )}

      {!webhooks || webhooks.length === 0 ? (
        <p className="text-gray-500 text-sm">No webhooks registered.</p>
      ) : (
        <div className="space-y-3">
          {webhooks.map((wh) => (
            <div key={wh.id} className="rounded border border-gray-700 bg-gray-900 p-4">
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-sm font-medium text-white">{wh.url}</div>
                  <div className="text-xs text-gray-400 mt-1">{wh.events.join(", ")}</div>
                  {wh.last_fired_at && (
                    <div className="text-xs text-gray-500 mt-1">
                      Last fired: {new Date(wh.last_fired_at).toLocaleString()} · HTTP {wh.last_status}
                    </div>
                  )}
                  {testResults[wh.id] && (
                    <div className="text-xs text-yellow-400 mt-1">Test: {testResults[wh.id]}</div>
                  )}
                </div>
                <div className="flex gap-2">
                  <button onClick={() => handleTest(wh.id)} className="rounded border border-gray-600 px-3 py-1 text-xs text-gray-300 hover:text-white">
                    Test
                  </button>
                  <button onClick={() => handleDelete(wh.id)} className="rounded border border-red-800 px-3 py-1 text-xs text-red-400 hover:text-red-300">
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
