"use client";

import { useState, useEffect, useRef } from "react";
import { useParams } from "next/navigation";
import useSWR, { mutate } from "swr";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";
import { useToast } from "@/lib/toast";

// ---------------------------------------------------------------------------
// Chat panel with WebSocket
// ---------------------------------------------------------------------------

type ChatMessage = { role: string; content: string; pending?: boolean };

function ChatPanel({ strategyId, runId }: { strategyId: string; runId: number }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const wsUrl =
      (typeof window !== "undefined"
        ? window.location.origin.replace(/^http/, "ws")
        : "ws://localhost:8000") +
      `/api/v1/ws/strategies/${strategyId}/runs/${runId}/chat`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setConnected(false);
    socket.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.error) {
          setMessages((prev) => [...prev, { role: "system", content: `Error: ${data.error}` }]);
          setWaiting(false);
          return;
        }
        if (data.is_final) {
          setWaiting(false);
          setMessages((prev) => {
            const updated = prev.filter((m) => !m.pending);
            return [...updated, { role: "agent", content: data.content }];
          });
        }
      } catch {
        /* ignore */
      }
    };

    setWs(socket);
    return () => socket.close();
  }, [strategyId, runId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function sendMessage() {
    if (!ws || !input.trim() || !connected || waiting) return;
    const msg = input.trim();
    setInput("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: msg },
      { role: "agent", content: "Thinking…", pending: true },
    ]);
    setWaiting(true);
    ws.send(JSON.stringify({ message: msg }));
  }

  return (
    <section className="space-y-2">
      <div className="flex items-center gap-2">
        <h2 className="text-sm font-medium uppercase tracking-wide text-gray-400">
          AI Chat
        </h2>
        <span
          className={`h-2 w-2 rounded-full ${connected ? "bg-green-400" : "bg-gray-600"}`}
          title={connected ? "Connected" : "Disconnected"}
        />
      </div>

      <div className="h-64 overflow-y-auto rounded border border-gray-700 bg-gray-950 p-3 space-y-2">
        {messages.length === 0 && (
          <p className="text-xs text-gray-600">
            Ask about strategy performance, suggest improvements, or explain signals…
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-xs rounded px-3 py-1.5 text-xs ${
                m.role === "user"
                  ? "bg-brand-500 text-white"
                  : m.role === "system"
                  ? "bg-red-900/40 text-red-300"
                  : m.pending
                  ? "bg-gray-800 text-gray-500 italic"
                  : "bg-gray-800 text-gray-200"
              }`}
              style={{ whiteSpace: "pre-wrap" }}
            >
              {m.content}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
          placeholder={connected ? "Ask something…" : "Connecting…"}
          disabled={!connected || waiting}
          className="flex-1 rounded border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-brand-500 disabled:opacity-50"
        />
        <button
          onClick={sendMessage}
          disabled={!connected || waiting || !input.trim()}
          className="rounded bg-brand-500 px-3 py-1.5 text-sm text-white hover:bg-sky-400 disabled:opacity-50"
        >
          Send
        </button>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function StrategyDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { toast } = useToast();
  const { data: strategy, isLoading, error } = useSWR(
    `/api/v1/strategies/${id}`,
    fetcher
  );
  const { data: runs } = useSWR(`/api/v1/strategies/${id}/runs`, fetcher, {
    refreshInterval: 3000,
  });
  const { data: datasets } = useSWR("/api/v1/datasets", fetcher);
  const { data: versions } = useSWR(`/api/v1/strategies/${id}/versions`, fetcher);
  const [showVersions, setShowVersions] = useState(false);

  const [showRunForm, setShowRunForm] = useState(false);
  const [mode, setMode] = useState("backtest");
  const [datasetId, setDatasetId] = useState("");
  const [starting, setStarting] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const [selectedRun, setSelectedRun] = useState<number | null>(null);
  const { data: runMetrics } = useSWR(
    selectedRun ? `/api/v1/strategies/${id}/runs/${selectedRun}/metrics` : null,
    fetcher,
    { refreshInterval: 5000 }
  );
  const { data: runTrades } = useSWR(
    selectedRun ? `/api/v1/strategies/${id}/runs/${selectedRun}/trades` : null,
    fetcher,
    { refreshInterval: 5000 }
  );

  const [stopping, setStopping] = useState<number | null>(null);

  const [editingDef, setEditingDef] = useState(false);
  const [defText, setDefText] = useState("");
  const [savingDef, setSavingDef] = useState(false);
  const [defError, setDefError] = useState<string | null>(null);

  function startEditDef() {
    setDefText(JSON.stringify(strategy.definition, null, 2));
    setDefError(null);
    setEditingDef(true);
  }

  async function saveDef() {
    setDefError(null);
    let parsed: object;
    try {
      parsed = JSON.parse(defText);
    } catch {
      setDefError("Invalid JSON");
      return;
    }
    setSavingDef(true);
    try {
      const res = await fetch(`/api/v1/strategies/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ definition: parsed }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setDefError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      setEditingDef(false);
      toast("Definition saved", "success");
      mutate(`/api/v1/strategies/${id}`);
      mutate(`/api/v1/strategies/${id}/versions`);
    } finally {
      setSavingDef(false);
    }
  }

  async function startRun() {
    setRunError(null);
    if (mode === "backtest" && !datasetId) {
      setRunError("Select a dataset for backtest");
      return;
    }
    setStarting(true);
    try {
      const res = await fetch(`/api/v1/strategies/${id}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          dataset_id: datasetId ? parseInt(datasetId) : null,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setRunError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      setShowRunForm(false);
      toast(`${mode} run queued`, "success");
      mutate(`/api/v1/strategies/${id}/runs`);
    } finally {
      setStarting(false);
    }
  }

  async function stopRun(runId: number) {
    setStopping(runId);
    try {
      await fetch(`/api/v1/strategies/${id}/runs/${runId}/stop`, { method: "POST" });
      toast("Stop requested", "info");
      mutate(`/api/v1/strategies/${id}/runs`);
    } finally {
      setStopping(null);
    }
  }

  if (isLoading) return <p className="text-gray-400">Loading…</p>;
  if (error || !strategy) return <p className="text-red-400">Strategy not found</p>;

  const activeRun = runs?.find((r: any) => ["pending", "running"].includes(r.status));

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-semibold text-white">{strategy.name}</h1>
            <StatusBadge status={strategy.status} />
          </div>
          {strategy.description && (
            <p className="mt-1 text-sm text-gray-400">{strategy.description}</p>
          )}
          <p className="mt-1 text-xs text-gray-500">
            Created {new Date(strategy.created_at).toLocaleDateString()}
          </p>
        </div>
        <a href="/strategy" className="text-sm text-gray-400 hover:text-white">
          ← Back
        </a>
      </div>

      {/* Definition */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-medium uppercase tracking-wide text-gray-400">Definition</h2>
          {versions && versions.length > 0 && (
            <button
              onClick={() => setShowVersions(!showVersions)}
              className="text-xs text-gray-400 hover:text-white"
            >
              {showVersions ? "Hide" : `Version history (${versions.length})`}
            </button>
          )}
        </div>
        {!editingDef && (
          <>
            <pre className="rounded bg-gray-900 p-3 text-xs text-gray-300 overflow-auto max-h-48">
              {JSON.stringify(strategy.definition, null, 2)}
            </pre>
            <button
              onClick={startEditDef}
              className="mt-2 text-xs text-gray-400 hover:text-white"
            >
              Edit definition
            </button>
          </>
        )}
        {editingDef && (
          <div className="space-y-2">
            <textarea
              value={defText}
              onChange={(e) => setDefText(e.target.value)}
              rows={14}
              className="w-full rounded border border-gray-700 bg-gray-900 p-3 font-mono text-xs text-gray-200 focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
            {defError && <p className="text-xs text-red-400">{defError}</p>}
            <div className="flex gap-2">
              <button
                onClick={saveDef}
                disabled={savingDef}
                className="rounded bg-brand-500 px-3 py-1.5 text-xs text-white hover:bg-sky-400 disabled:opacity-50"
              >
                {savingDef ? "Saving…" : "Save"}
              </button>
              <button
                onClick={() => setEditingDef(false)}
                className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-400 hover:text-white"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
        {showVersions && versions && (
          <div className="mt-3 space-y-2 max-h-64 overflow-y-auto">
            {(versions as { id: number; version: number; definition: object; created_at: string }[]).map((v) => (
              <div key={v.id} className="rounded border border-gray-800 bg-gray-900 p-3">
                <p className="mb-1 text-xs text-gray-500">
                  v{v.version} · {new Date(v.created_at).toLocaleString()}
                </p>
                <pre className="text-xs text-gray-400 overflow-auto max-h-32">
                  {JSON.stringify(v.definition, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Runs */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-medium uppercase tracking-wide text-gray-400">Runs</h2>
          <div className="flex gap-2">
            {activeRun && (
              <button
                onClick={() => stopRun(activeRun.id)}
                disabled={stopping === activeRun.id}
                className="rounded border border-red-700 px-3 py-1 text-xs text-red-400 hover:bg-red-900/30 disabled:opacity-50"
              >
                {stopping === activeRun.id ? "Stopping…" : "Stop"}
              </button>
            )}
            <button
              onClick={() => setShowRunForm(!showRunForm)}
              disabled={!!activeRun}
              className="rounded bg-brand-500 px-3 py-1 text-xs text-white hover:bg-sky-400 disabled:opacity-40"
              title={activeRun ? "A run is already active" : undefined}
            >
              + New Run
            </button>
          </div>
        </div>

        {showRunForm && (
          <div className="mb-4 rounded border border-gray-700 bg-gray-900 p-4 space-y-3">
            <div>
              <label className="mb-1 block text-xs text-gray-400">Mode</label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                <option value="backtest">Backtest</option>
                <option value="paper">Paper (live yfinance, simulated orders)</option>
                <option value="live">Live (broker — not implemented)</option>
              </select>
            </div>
            {mode === "backtest" && (
              <div>
                <label className="mb-1 block text-xs text-gray-400">Dataset</label>
                <select
                  value={datasetId}
                  onChange={(e) => setDatasetId(e.target.value)}
                  className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                >
                  <option value="">Select dataset…</option>
                  {datasets?.map((d: any) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.row_count} rows)
                    </option>
                  ))}
                </select>
              </div>
            )}
            {runError && <p className="text-xs text-red-400">{runError}</p>}
            <div className="flex gap-2">
              <button
                onClick={startRun}
                disabled={starting}
                className="rounded bg-brand-500 px-3 py-1.5 text-xs text-white hover:bg-sky-400 disabled:opacity-50"
              >
                {starting ? "Queuing…" : "Start"}
              </button>
              <button
                onClick={() => setShowRunForm(false)}
                className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-400 hover:text-white"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {runs && runs.length === 0 && (
          <p className="text-sm text-gray-500">No runs yet.</p>
        )}

        {runs && runs.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-gray-800 text-xs text-gray-400 uppercase">
                <th className="py-2 pr-4">Run</th>
                <th className="py-2 pr-4">Mode</th>
                <th className="py-2 pr-4">Status</th>
                <th className="py-2 pr-4">Progress</th>
                <th className="py-2 pr-4">Message</th>
                <th className="py-2 pr-4">Started</th>
                <th className="py-2" />
              </tr>
            </thead>
            <tbody>
              {runs.map((run: any) => (
                <tr
                  key={run.id}
                  className={`border-b border-gray-800/50 hover:bg-gray-900 cursor-pointer ${
                    selectedRun === run.id ? "bg-gray-900" : ""
                  }`}
                  onClick={() => setSelectedRun(selectedRun === run.id ? null : run.id)}
                >
                  <td className="py-2 pr-4 text-gray-300">#{run.id}</td>
                  <td className="py-2 pr-4 text-gray-400">{run.mode}</td>
                  <td className="py-2 pr-4">
                    <StatusBadge status={run.status} />
                  </td>
                  <td className="py-2 pr-4 text-gray-400">
                    {run.status === "running" ? `${run.progress_pct.toFixed(0)}%` : run.status === "completed" ? "100%" : "—"}
                  </td>
                  <td className="py-2 pr-4 text-gray-400 text-xs max-w-[200px] truncate">
                    {run.message ?? "—"}
                  </td>
                  <td className="py-2 pr-4 text-gray-400">
                    {run.started_at ? new Date(run.started_at).toLocaleString() : "—"}
                  </td>
                  <td className="py-2 text-right text-xs">
                    <a
                      href={`/strategy/${id}/runs/${run.id}`}
                      className="text-brand-400 hover:text-sky-300 mr-3"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Detail →
                    </a>
                    <span className="text-gray-500">{selectedRun === run.id ? "▲ Hide" : "▼ Show"}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* Run detail panel */}
      {selectedRun && (
        <section className="space-y-4">
          <h2 className="text-sm font-medium uppercase tracking-wide text-gray-400">
            Run #{selectedRun} — Details
          </h2>

          {runMetrics && Object.keys(runMetrics).length > 0 && (
            <div>
              <h3 className="mb-2 text-xs text-gray-500">Metrics</h3>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {Object.entries(runMetrics).map(([key, val]) => (
                  <div key={key} className="rounded bg-gray-900 px-3 py-2">
                    <p className="text-xs text-gray-500">{key.replace(/_/g, " ")}</p>
                    <p className="text-sm font-mono text-white">
                      {typeof val === "number"
                        ? key === "win_rate" ? `${(val * 100).toFixed(1)}%`
                        : key === "total_trades" ? String(val)
                        : val.toFixed(4)
                        : String(val)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {runTrades && runTrades.length > 0 && (
            <div>
              <h3 className="mb-2 text-xs text-gray-500">Trades ({runTrades.length})</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-gray-800 text-gray-500 uppercase">
                      <th className="py-1 pr-3">Symbol</th>
                      <th className="py-1 pr-3">Dir</th>
                      <th className="py-1 pr-3">Entry</th>
                      <th className="py-1 pr-3">Exit</th>
                      <th className="py-1 pr-3">SL</th>
                      <th className="py-1 pr-3">TP</th>
                      <th className="py-1 pr-3">PnL</th>
                      <th className="py-1 pr-3">Opened</th>
                      <th className="py-1">Closed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runTrades.map((t: any, i: number) => (
                      <tr
                        key={i}
                        className={`border-b border-gray-800/40 ${
                          t.profit >= 0 ? "text-green-400" : "text-red-400"
                        }`}
                      >
                        <td className="py-1 pr-3">{t.symbol}</td>
                        <td className="py-1 pr-3">{t.direction}</td>
                        <td className="py-1 pr-3 font-mono">{t.entry_price?.toFixed(4)}</td>
                        <td className="py-1 pr-3 font-mono">{t.exit_price?.toFixed(4)}</td>
                        <td className="py-1 pr-3 font-mono text-gray-500">{t.sl_price?.toFixed(4) ?? "—"}</td>
                        <td className="py-1 pr-3 font-mono text-gray-500">{t.tp_price?.toFixed(4) ?? "—"}</td>
                        <td className="py-1 pr-3 font-mono font-medium">
                          {t.profit >= 0 ? "+" : ""}{(t.profit * 100).toFixed(2)}%
                        </td>
                        <td className="py-1 pr-3 text-gray-400">
                          {new Date(t.opened_at).toLocaleDateString()}
                        </td>
                        <td className="py-1 text-gray-400">
                          {new Date(t.closed_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Chat panel */}
          <ChatPanel strategyId={id} runId={selectedRun} />
        </section>
      )}
    </div>
  );
}
