"use client";

import { useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { EquityChart } from "@/components/equity-chart";
import { MetricsGrid } from "@/components/metrics-grid";
import { useSSE } from "@/hooks/use-sse";

interface Trade {
  id: number;
  symbol: string;
  direction: string;
  entry_price: number;
  exit_price: number | null;
  volume: number;
  profit: number | null;
  opened_at: string;
  closed_at: string | null;
  exit_reason?: string;
}

interface RunEvent {
  type: string;
  [key: string]: unknown;
}

export default function RunDetailPage() {
  const params = useParams<{ id: string; run_id: string }>();
  const router = useRouter();
  const strategyId = params.id;
  const runId = params.run_id;

  const [activeTab, setActiveTab] = useState<"trades" | "events" | "chat">("trades");
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [sendingChat, setSendingChat] = useState(false);
  const [stopping, setStopping] = useState(false);

  const strategyUrl = `/api/v1/strategies/${strategyId}`;
  const runUrl = `/api/v1/strategies/${strategyId}/runs/${runId}`;
  const metricsUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/metrics`;
  const equityUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/equity`;
  const tradesUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/trades`;
  const chatUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/chat`;

  const { data: strategy } = useSWR(strategyUrl, fetcher);
  const { data: run, mutate: mutateRun } = useSWR(runUrl, fetcher, {
    refreshInterval: (data) => (data?.status === "running" || data?.status === "pending") ? 3000 : 0,
  });
  const { data: metrics } = useSWR(metricsUrl, fetcher, {
    refreshInterval: run?.status === "completed" ? 0 : 5000,
  });
  const { data: equityData } = useSWR(equityUrl, fetcher);
  const { data: trades } = useSWR(tradesUrl, fetcher);
  const { data: chatHistory, mutate: mutateChat } = useSWR(chatUrl, fetcher);

  const onSSEEvent = useCallback((data: unknown) => {
    setEvents((prev) => [data as RunEvent, ...prev].slice(0, 200));
  }, []);

  useSSE(
    run?.status === "running" ? `/api/v1/strategies/${strategyId}/runs/${runId}/events` : null,
    onSSEEvent
  );

  async function handleStop() {
    setStopping(true);
    try {
      await fetch(`/api/v1/strategies/${strategyId}/runs/${runId}/stop`, { method: "POST" });
      mutateRun();
    } finally {
      setStopping(false);
    }
  }

  async function handleSendChat(e: React.FormEvent) {
    e.preventDefault();
    if (!chatInput.trim()) return;
    setSendingChat(true);
    try {
      await fetch(chatUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: chatInput }),
      });
      setChatInput("");
      mutateChat();
    } finally {
      setSendingChat(false);
    }
  }

  const isActive = run?.status === "running" || run?.status === "pending";

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <button onClick={() => router.push("/strategy")} className="hover:text-white">Strategies</button>
          <span>/</span>
          <button onClick={() => router.push(`/strategy/${strategyId}`)} className="hover:text-white">
            {strategy?.name ?? `Strategy ${strategyId}`}
          </button>
          <span>/</span>
          <span className="text-white">Run #{runId}</span>
        </div>
        <div className="flex items-center gap-3">
          {run && (
            <span className={`rounded px-2 py-1 text-xs font-medium ${
              run.status === "completed" ? "bg-green-900 text-green-300" :
              run.status === "running" ? "bg-blue-900 text-blue-300" :
              run.status === "error" ? "bg-red-900 text-red-300" :
              "bg-gray-700 text-gray-300"
            }`}>
              {run.status}
            </span>
          )}
          {run && (
            <span className="rounded px-2 py-1 text-xs bg-gray-700 text-gray-300">
              {run.mode}
            </span>
          )}
          {isActive && (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="rounded bg-red-600 px-3 py-1.5 text-sm text-white hover:bg-red-500 disabled:opacity-50"
            >
              {stopping ? "Stopping…" : "Stop"}
            </button>
          )}
        </div>
      </div>

      {/* Progress bar for active runs */}
      {isActive && run && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-400">
            <span>Progress</span>
            <span>{run.progress_pct?.toFixed(1)}%</span>
          </div>
          <div className="h-2 rounded bg-gray-700">
            <div
              className="h-2 rounded bg-brand-500 transition-all"
              style={{ width: `${run.progress_pct ?? 0}%` }}
            />
          </div>
          {run.message && <p className="text-xs text-gray-400">{run.message}</p>}
        </div>
      )}

      {/* Metrics */}
      {metrics && <MetricsGrid metrics={metrics} />}

      {/* Equity Chart */}
      {equityData && equityData.length > 0 && (
        <div className="rounded border border-gray-700 bg-gray-900 p-4">
          <h3 className="mb-3 text-sm font-medium text-gray-300">Equity Curve</h3>
          <EquityChart data={equityData} />
        </div>
      )}

      {/* Tabs */}
      <div>
        <div className="flex border-b border-gray-700">
          {(["trades", "events", "chat"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm capitalize ${
                activeTab === tab ? "border-b-2 border-brand-500 text-white" : "text-gray-400 hover:text-white"
              }`}
            >
              {tab}
              {tab === "trades" && trades ? ` (${trades.length})` : ""}
              {tab === "events" && events.length > 0 ? ` (${events.length})` : ""}
            </button>
          ))}
        </div>

        {/* Trades Tab */}
        {activeTab === "trades" && (
          <div className="mt-4">
            {!trades || trades.length === 0 ? (
              <p className="text-gray-500 text-sm">No trades yet</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-400">
                      <th className="pb-2 pr-4">Symbol</th>
                      <th className="pb-2 pr-4">Direction</th>
                      <th className="pb-2 pr-4">Entry</th>
                      <th className="pb-2 pr-4">Exit</th>
                      <th className="pb-2 pr-4">Profit</th>
                      <th className="pb-2 pr-4">Opened</th>
                      <th className="pb-2">Exit Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(trades as Trade[]).map((t) => (
                      <tr key={t.id} className="border-t border-gray-800">
                        <td className="py-2 pr-4 text-white">{t.symbol}</td>
                        <td className={`py-2 pr-4 ${t.direction === "buy" ? "text-green-400" : "text-red-400"}`}>{t.direction}</td>
                        <td className="py-2 pr-4 text-gray-300">{t.entry_price?.toFixed(4)}</td>
                        <td className="py-2 pr-4 text-gray-300">{t.exit_price?.toFixed(4) ?? "—"}</td>
                        <td className={`py-2 pr-4 ${(t.profit ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                          {t.profit != null ? `${(t.profit * 100).toFixed(2)}%` : "—"}
                        </td>
                        <td className="py-2 pr-4 text-gray-400 text-xs">{t.opened_at ? new Date(t.opened_at).toLocaleString() : "—"}</td>
                        <td className="py-2 text-gray-400 text-xs">{t.exit_reason ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Events Tab */}
        {activeTab === "events" && (
          <div className="mt-4 space-y-2 max-h-96 overflow-y-auto">
            {events.length === 0 ? (
              <p className="text-gray-500 text-sm">No events yet{isActive ? " — waiting for run events…" : ""}</p>
            ) : (
              events.map((evt, i) => (
                <div key={i} className="rounded bg-gray-800 px-3 py-2 text-xs font-mono text-gray-300">
                  <span className="text-brand-400 mr-2">[{evt.type}]</span>
                  {JSON.stringify(evt, null, 0)}
                </div>
              ))
            )}
          </div>
        )}

        {/* Chat Tab */}
        {activeTab === "chat" && (
          <div className="mt-4 space-y-4">
            <div className="max-h-96 overflow-y-auto space-y-2">
              {!chatHistory || chatHistory.length === 0 ? (
                <p className="text-gray-500 text-sm">No messages yet. Ask about this run's performance.</p>
              ) : (
                (chatHistory as { id: number; role: string; message: string; created_at: string }[]).map((msg) => (
                  <div key={msg.id} className={`rounded p-3 text-sm ${msg.role === "user" ? "bg-brand-900 text-white" : "bg-gray-800 text-gray-300"}`}>
                    <div className="text-xs text-gray-400 mb-1 capitalize">{msg.role}</div>
                    {msg.message}
                  </div>
                ))
              )}
            </div>
            <form onSubmit={handleSendChat} className="flex gap-2">
              <input
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask about this run…"
                className="flex-1 rounded border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              />
              <button
                type="submit"
                disabled={sendingChat || !chatInput.trim()}
                className="rounded bg-brand-500 px-4 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-50"
              >
                Send
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}
