"use client";

import { useState, useCallback, useRef, useEffect, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
import useSWR from "swr";
import { apiFetch, fetcher, fetcherWithMeta } from "@/lib/fetcher";
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

// Declared outside the component so the type is available before any hooks run
type ChartAccum = {
  candles: any[];
  indicators: Record<string, any>;
  markers: any[];
  events?: any[];
  has_more: boolean;
  bar_count: number;
};

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

  const [tradesPage, setTradesPage] = useState(1);
  const [tradesPageSize, setTradesPageSize] = useState(50);
  const [tradesDirection, setTradesDirection] = useState("");
  const [tradesPhase, setTradesPhase] = useState("");
  const [tradesExitReason, setTradesExitReason] = useState("");
  const [tradesProfitable, setTradesProfitable] = useState("");

  const strategyUrl = `/api/v1/strategies/${strategyId}`;
  const runUrl = `/api/v1/strategies/${strategyId}/runs/${runId}`;
  const metricsUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/metrics`;
  const equityUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/equity`;
  const chatUrl = `/api/v1/strategies/${strategyId}/runs/${runId}/chat`;

  const tradesUrl = useMemo(() => {
    const p = new URLSearchParams({ page: String(tradesPage), page_size: String(tradesPageSize) });
    if (tradesDirection) p.set("direction", tradesDirection);
    if (tradesPhase) p.set("phase", tradesPhase);
    if (tradesExitReason) p.set("exit_reason", tradesExitReason);
    if (tradesProfitable) p.set("profitable", tradesProfitable);
    return `/api/v1/strategies/${strategyId}/runs/${runId}/trades?${p}`;
  }, [strategyId, runId, tradesPage, tradesPageSize, tradesDirection, tradesPhase, tradesExitReason, tradesProfitable]);

  const [chartFrom, setChartFrom] = useState("");
  const [chartTo, setChartTo] = useState("");
  const [appliedRange, setAppliedRange] = useState({ from: "", to: "" });
  const [chartMode, setChartMode] = useState<"indicators" | "conditions">("indicators");
  const [accumulated, setAccumulated] = useState<ChartAccum | null>(null);
  const [loadingOlder, setLoadingOlder] = useState(false);

  // SWR hooks — run must be declared before chartUrl because it appears in the deps array
  const { data: strategy } = useSWR(strategyUrl, fetcher);
  const { data: run, mutate: mutateRun } = useSWR(runUrl, fetcher, {
    refreshInterval: (data) => (data?.status === "running" || data?.status === "pending") ? 3000 : 0,
  });
  const { data: metrics } = useSWR(metricsUrl, fetcher, {
    refreshInterval: run?.status === "completed" ? 0 : 5000,
  });
  const { data: equityData } = useSWR(equityUrl, fetcher);
  const { data: tradesResp } = useSWR(tradesUrl, fetcherWithMeta, {
    refreshInterval: (data) => ((data as any)?.meta?.total === 0 ? 5000 : 0),
  });
  const trades = tradesResp?.data as Trade[] | undefined;
  const tradesTotalCount = (tradesResp?.meta?.total ?? 0) as number;
  const { data: chatHistory, mutate: mutateChat } = useSWR(chatUrl, fetcher);

  // chartUrl depends on run.status — must come after the run useSWR
  const chartUrl = useMemo(() => {
    if (run?.status !== "completed") return null;
    const base = `/api/v1/strategies/${strategyId}/runs/${runId}/chart-data`;
    const p = new URLSearchParams();
    if (appliedRange.from) p.set("from_ts", String(Math.floor(new Date(appliedRange.from).getTime() / 1000)));
    if (appliedRange.to)   p.set("to_ts",   String(Math.floor(new Date(appliedRange.to).getTime()   / 1000)));
    const qs = p.toString();
    return qs ? `${base}?${qs}` : base;
  }, [strategyId, runId, run?.status, appliedRange]);

  const { data: freshChartData, isLoading: chartLoading } = useSWR(
    chartUrl,
    fetcher,
    { revalidateOnFocus: false },
  );

  // Reset accumulated state when the chart URL changes (different range / run)
  useEffect(() => { setAccumulated(null); }, [chartUrl]);
  // Seed accumulator from SWR's initial fetch
  useEffect(() => { if (freshChartData) setAccumulated(freshChartData); }, [freshChartData]);

  async function loadOlderBars() {
    if (!accumulated?.candles.length || loadingOlder) return;
    const earliestTs = Math.min(...accumulated.candles.map((c: any) => c.time));
    const base = `/api/v1/strategies/${strategyId}/runs/${runId}/chart-data`;
    setLoadingOlder(true);
    try {
      const res = await apiFetch(`${base}?to_ts=${earliestTs - 1}&limit=2000`);
      if (!res.ok) return;
      const body = await res.json();
      const older: ChartAccum = body.data ?? body;
      setAccumulated((prev) => {
        if (!prev) return older;
        const seenTimes = new Set(prev.candles.map((c: any) => c.time));
        const newCandles = [
          ...(older.candles ?? []).filter((c: any) => !seenTimes.has(c.time)),
          ...prev.candles,
        ];
        const mergedIndicators: Record<string, any> = { ...prev.indicators };
        for (const [key, series] of Object.entries(older.indicators ?? {})) {
          if (mergedIndicators[key]) {
            const existTimes = new Set(mergedIndicators[key].data?.map((p: any) => p.time));
            mergedIndicators[key] = {
              ...mergedIndicators[key],
              data: [
                ...((series as any).data ?? []).filter((p: any) => !existTimes.has(p.time)),
                ...(mergedIndicators[key].data ?? []),
              ],
            };
          } else {
            mergedIndicators[key] = series;
          }
        }
        const seenMarkers = new Set(prev.markers?.map((m: any) => `${m.time}:${m.color}:${m.shape}`));
        const mergedMarkers = [
          ...(older.markers ?? []).filter((m: any) => !seenMarkers.has(`${m.time}:${m.color}:${m.shape}`)),
          ...(prev.markers ?? []),
        ].sort((a: any, b: any) => a.time - b.time);
        return {
          ...prev,
          candles: newCandles,
          indicators: mergedIndicators,
          markers: mergedMarkers,
          has_more: older.has_more ?? false,
          bar_count: newCandles.length,
        };
      });
    } finally {
      setLoadingOlder(false);
    }
  }

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
      await apiFetch(`/api/v1/strategies/${strategyId}/runs/${runId}/stop`, { method: "POST" });
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
      await apiFetch(chatUrl, {
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
            <div className="ml-auto flex items-center gap-2">
              {accumulated?.has_more && (
                <button
                  onClick={loadOlderBars}
                  disabled={loadingOlder}
                  className="text-xs text-sky-400 hover:text-sky-300 disabled:opacity-50"
                >
                  {loadingOlder ? "Loading…" : "← Load older"}
                </button>
              )}
              {accumulated && (
                <span className="text-xs text-gray-600">
                  {accumulated.bar_count ?? accumulated.candles?.length ?? 0} bars
                  {accumulated.has_more && <span className="text-gray-700"> · more available</span>}
                </span>
              )}
            </div>
          </div>
          {chartLoading && (
            <div className="flex items-center justify-center h-16 text-gray-500 text-sm">Loading chart…</div>
          )}
          {accumulated && (
            <StrategyChart
              candles={accumulated.candles ?? []}
              indicators={Object.fromEntries(
                Object.entries(accumulated.indicators ?? {}).filter(([, s]: [string, any]) =>
                  chartMode === "conditions"
                    ? true
                    : !(s.group?.startsWith("cond:") || s.line_style === "dashed" || s.line_style === "step")
                )
              )}
              markers={accumulated.markers ?? []}
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
              {tab === "trades" && tradesTotalCount > 0 ? ` (${tradesTotalCount.toLocaleString()})` : ""}
              {tab === "events" && events.length > 0 ? ` (${events.length})` : ""}
            </button>
          ))}
        </div>

        {/* Trades Tab */}
        {activeTab === "trades" && (
          <div className="mt-4 space-y-3">
            {/* Filter + pagination bar */}
            <div className="flex flex-wrap items-center gap-2">
              <select
                value={tradesDirection}
                onChange={(e) => { setTradesDirection(e.target.value); setTradesPage(1); }}
                className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-white focus:outline-none"
              >
                <option value="">All directions</option>
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </select>
              <select
                value={tradesPhase}
                onChange={(e) => { setTradesPhase(e.target.value); setTradesPage(1); }}
                className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-white focus:outline-none"
              >
                <option value="">All phases</option>
                <option value="is">IS</option>
                <option value="oos">OOS</option>
              </select>
              <select
                value={tradesExitReason}
                onChange={(e) => { setTradesExitReason(e.target.value); setTradesPage(1); }}
                className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-white focus:outline-none"
              >
                <option value="">All exit reasons</option>
                <option value="signal">Signal</option>
                <option value="sl">Stop-loss</option>
                <option value="tp">Take-profit</option>
                <option value="end_of_data">End of data</option>
                <option value="trailing_stop">Trailing stop</option>
              </select>
              <select
                value={tradesProfitable}
                onChange={(e) => { setTradesProfitable(e.target.value); setTradesPage(1); }}
                className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-white focus:outline-none"
              >
                <option value="">All results</option>
                <option value="true">Winners only</option>
                <option value="false">Losers only</option>
              </select>
              <div className="ml-auto flex items-center gap-2">
                <select
                  value={tradesPageSize}
                  onChange={(e) => { setTradesPageSize(Number(e.target.value)); setTradesPage(1); }}
                  className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-white focus:outline-none"
                >
                  <option value={20}>20 / page</option>
                  <option value={50}>50 / page</option>
                  <option value={100}>100 / page</option>
                  <option value={500}>500 / page</option>
                </select>
                <button
                  onClick={() => setTradesPage((p) => Math.max(1, p - 1))}
                  disabled={tradesPage === 1}
                  className="rounded border border-gray-700 px-2 py-1 text-xs text-gray-400 hover:text-white disabled:opacity-40"
                >
                  ←
                </button>
                <span className="text-xs text-gray-500">
                  {tradesPage} / {Math.max(1, Math.ceil(tradesTotalCount / tradesPageSize))}
                </span>
                <button
                  onClick={() => setTradesPage((p) => p + 1)}
                  disabled={tradesPage >= Math.ceil(tradesTotalCount / tradesPageSize)}
                  className="rounded border border-gray-700 px-2 py-1 text-xs text-gray-400 hover:text-white disabled:opacity-40"
                >
                  →
                </button>
              </div>
            </div>

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
                    {trades.map((t, i) => (
                      <tr key={t.id} className="border-t border-gray-800 hover:bg-gray-800/40">
                        <td className="py-2 pr-3 text-gray-500 text-xs">{(tradesPage - 1) * tradesPageSize + i + 1}</td>
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
