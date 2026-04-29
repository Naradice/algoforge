"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { EquityChart } from "@/components/equity-chart";
import { MetricsGrid } from "@/components/metrics-grid";
import { StrategyChart } from "@/components/strategy-chart";
import { useSSE } from "@/hooks/use-sse";

interface Trade {
  id: number;
  symbol: string;
  direction: string;
  entry_price: number;
  exit_price: number | null;
  volume: number;
  sl_price: number | null;
  tp_price: number | null;
  profit: number | null;
  opened_at: string;
  closed_at: string | null;
  exit_reason: string | null;
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
  const chatBottomRef = useRef<HTMLDivElement>(null);

  const strategyUrl = `/api/v1/strategies/${strategyId}`;
  const runUrl = `/api/v1/strategies/${strategyId}/runs/${runId}`;
  const metricsUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/metrics`;
  const equityUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/equity`;
  const tradesUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/trades`;
  const chatUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/chat`;

  const [chartFrom, setChartFrom] = useState("");
  const [chartTo, setChartTo] = useState("");
  const [appliedRange, setAppliedRange] = useState({ from: "", to: "" });
  const [chartMode, setChartMode] = useState<"indicators" | "conditions">("indicators");

  function chartDataUrl() {
    const base = `/api/v1/strategies/${strategyId}/runs/${runId}/chart-data`;
    const p = new URLSearchParams();
    if (appliedRange.from) p.set("from_ts", String(Math.floor(new Date(appliedRange.from).getTime() / 1000)));
    if (appliedRange.to)   p.set("to_ts",   String(Math.floor(new Date(appliedRange.to).getTime()   / 1000)));
    const qs = p.toString();
    return qs ? `${base}?${qs}` : base;
  }

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
  const { data: chartData, isLoading: chartLoading } = useSWR(
    run?.status === "completed" ? chartDataUrl() : null,
    fetcher,
    { revalidateOnFocus: false },
  );

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

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, sendingChat]);

  async function handleSendChat(e: React.FormEvent) {
    e.preventDefault();
    if (!chatInput.trim()) return;
    setSendingChat(true);
    const msg = chatInput;
    setChatInput("");
    try {
      await fetch(chatUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      await mutateChat();
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

      {/* Price Chart */}
      {run?.status === "completed" && (
        <div className="rounded border border-gray-700 bg-gray-900 p-4 space-y-3">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-medium text-gray-300">Price Chart</h3>
            <span className="text-gray-600 text-xs">|</span>
            <div className="flex rounded border border-gray-700 overflow-hidden text-xs">
              {(["indicators", "conditions"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setChartMode(mode)}
                  className={`px-2 py-0.5 capitalize ${chartMode === mode ? "bg-brand-500 text-white" : "bg-gray-800 text-gray-400 hover:text-gray-200"}`}
                >
                  {mode}
                </button>
              ))}
            </div>
            <span className="text-gray-600 text-xs">|</span>
            <input
              type="datetime-local"
              value={chartFrom}
              onChange={(e) => setChartFrom(e.target.value)}
              className="rounded border border-gray-700 bg-gray-800 px-2 py-0.5 text-xs text-gray-300 focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
            <span className="text-gray-600 text-xs">→</span>
            <input
              type="datetime-local"
              value={chartTo}
              onChange={(e) => setChartTo(e.target.value)}
              className="rounded border border-gray-700 bg-gray-800 px-2 py-0.5 text-xs text-gray-300 focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
            <button
              onClick={() => setAppliedRange({ from: chartFrom, to: chartTo })}
              className="rounded bg-brand-500 px-2 py-0.5 text-xs text-white hover:bg-sky-400"
            >
              Apply
            </button>
            {(appliedRange.from || appliedRange.to) && (
              <button
                onClick={() => { setChartFrom(""); setChartTo(""); setAppliedRange({ from: "", to: "" }); }}
                className="text-xs text-gray-500 hover:text-gray-300"
              >
                Reset
              </button>
            )}
            {chartData && (
              <span className="text-xs text-gray-600 ml-auto">{chartData.candles?.length ?? 0} bars</span>
            )}
          </div>
          {chartLoading && (
            <div className="flex items-center justify-center h-16 text-gray-500 text-sm">Loading chart…</div>
          )}
          {chartData && (
            <StrategyChart
              candles={chartData.candles ?? []}
              indicators={Object.fromEntries(
                Object.entries(chartData.indicators ?? {}).filter(([, s]: [string, any]) =>
                  chartMode === "conditions"
                    ? true
                    : !(s.group?.startsWith("cond:") || s.line_style === "dashed" || s.line_style === "step")
                )
              )}
              markers={chartData.markers ?? []}
            />
          )}
        </div>
      )}

      {/* Equity Curve */}
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
                    <tr className="text-left text-gray-400 border-b border-gray-700">
                      <th className="pb-2 pr-3">#</th>
                      <th className="pb-2 pr-3">Dir</th>
                      <th className="pb-2 pr-3">Entry</th>
                      <th className="pb-2 pr-3">Exit</th>
                      <th className="pb-2 pr-3">SL</th>
                      <th className="pb-2 pr-3">TP</th>
                      <th className="pb-2 pr-3">Vol</th>
                      <th className="pb-2 pr-3">PnL</th>
                      <th className="pb-2 pr-3">Opened</th>
                      <th className="pb-2 pr-3">Closed</th>
                      <th className="pb-2">Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(trades as Trade[]).map((t) => (
                      <tr key={t.id} className="border-t border-gray-800 hover:bg-gray-800/40">
                        <td className="py-2 pr-3 text-gray-500 text-xs">{t.id}</td>
                        <td className={`py-2 pr-3 font-medium ${t.direction === "buy" ? "text-green-400" : "text-red-400"}`}>
                          {t.direction === "buy" ? "▲" : "▼"} {t.direction}
                        </td>
                        <td className="py-2 pr-3 text-gray-300 font-mono text-xs">{t.entry_price?.toFixed(4)}</td>
                        <td className="py-2 pr-3 text-gray-300 font-mono text-xs">{t.exit_price?.toFixed(4) ?? "—"}</td>
                        <td className="py-2 pr-3 text-gray-500 font-mono text-xs">{t.sl_price?.toFixed(4) ?? "—"}</td>
                        <td className="py-2 pr-3 text-gray-500 font-mono text-xs">{t.tp_price?.toFixed(4) ?? "—"}</td>
                        <td className="py-2 pr-3 text-gray-400 text-xs">{t.volume}</td>
                        <td className={`py-2 pr-3 font-mono text-xs font-medium ${(t.profit ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                          {t.profit != null ? `${t.profit >= 0 ? "+" : ""}${t.profit.toFixed(4)}` : "—"}
                        </td>
                        <td className="py-2 pr-3 text-gray-400 text-xs whitespace-nowrap">{t.opened_at ? new Date(t.opened_at).toLocaleString() : "—"}</td>
                        <td className="py-2 pr-3 text-gray-400 text-xs whitespace-nowrap">{t.closed_at ? new Date(t.closed_at).toLocaleString() : "—"}</td>
                        <td className="py-2 text-gray-500 text-xs">{t.exit_reason ?? "—"}</td>
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
          <div className="mt-4 flex flex-col gap-3">
            {/* Message history */}
            <div className="flex flex-col gap-2 max-h-[480px] overflow-y-auto pr-1">
              {!chatHistory || chatHistory.length === 0 ? (
                <div className="rounded border border-gray-700 bg-gray-900 p-6 text-center">
                  <p className="text-gray-400 text-sm font-medium mb-1">Ask about this run</p>
                  <p className="text-gray-500 text-xs">
                    The AI has access to the strategy definition, metrics, and recent trades.
                    Try: &ldquo;Why is the win rate low?&rdquo; or &ldquo;Suggest parameter improvements.&rdquo;
                  </p>
                </div>
              ) : (
                (chatHistory as { id: number; role: string; message: string; created_at: string }[]).map((msg) => {
                  const isUser = msg.role === "user";
                  return (
                    <div key={msg.id} className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
                      <div className={`max-w-[80%] rounded-lg px-4 py-3 text-sm ${
                        isUser
                          ? "bg-brand-600 text-white rounded-br-sm"
                          : "bg-gray-800 text-gray-200 rounded-bl-sm"
                      }`}>
                        {!isUser && (
                          <div className="text-xs text-gray-400 mb-1 font-medium">AlgoForge AI</div>
                        )}
                        <p className="whitespace-pre-wrap leading-relaxed">{msg.message}</p>
                        <div className={`text-xs mt-1 ${isUser ? "text-sky-200/60" : "text-gray-500"}`}>
                          {new Date(msg.created_at).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>
                  );
                })
              )}

              {/* Thinking indicator */}
              {sendingChat && (
                <div className="flex justify-start">
                  <div className="bg-gray-800 rounded-lg rounded-bl-sm px-4 py-3">
                    <div className="flex gap-1 items-center h-4">
                      <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                      <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                      <span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
                    </div>
                  </div>
                </div>
              )}

              <div ref={chatBottomRef} />
            </div>

            {/* Input */}
            <form onSubmit={handleSendChat} className="flex gap-2">
              <input
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask about this run's performance…"
                disabled={sendingChat}
                className="flex-1 rounded border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500 disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={sendingChat || !chatInput.trim()}
                className="rounded bg-brand-500 px-4 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-50 whitespace-nowrap"
              >
                {sendingChat ? "Thinking…" : "Send"}
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
}
