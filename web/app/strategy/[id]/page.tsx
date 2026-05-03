"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import { useParams } from "next/navigation";
import useSWR, { mutate } from "swr";
import { apiFetch, fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";
import { useToast } from "@/lib/toast";
import { StrategyEditor, StrategyDefinitionView } from "@/components/strategy-editor";
import { StrategyChart } from "@/components/strategy-chart";
import { EquityChart } from "@/components/equity-chart";
import { InvestigationPanel } from "@/components/investigation-panel";

// ---------------------------------------------------------------------------
// Metrics panel — handles combined / IS / OOS split view
// ---------------------------------------------------------------------------

const METRIC_FORMAT: Record<string, (v: number) => string> = {
  win_rate:              (v) => `${(v * 100).toFixed(1)}%`,
  total_trades:          (v) => String(Math.round(v)),
  max_consecutive_losses:(v) => String(Math.round(v)),
  max_positions:         (v) => String(Math.round(v)),
};

function fmtMetric(key: string, val: number): string {
  const base = key.replace(/^(is_|oos_)/, "");
  const fmt = METRIC_FORMAT[base];
  if (fmt) return fmt(val);
  const pct = ["sl_pct", "tp_pct", "slippage_pct", "commission_pct",
               "daily_loss_limit_pct", "avg_mae", "avg_mfe"].includes(base);
  return pct ? `${(val * 100).toFixed(2)}%` : val.toFixed(4);
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded bg-gray-900 px-3 py-2">
      <p className="text-xs text-gray-500 leading-tight">{label.replace(/_/g, " ")}</p>
      <p className="text-sm font-mono text-white">{value}</p>
    </div>
  );
}

function MetricsPanel({ metrics }: { metrics: Record<string, number> }) {
  const hasWF = Object.keys(metrics).some((k) => k.startsWith("is_") || k.startsWith("oos_"));

  const combinedKeys = Object.keys(metrics).filter((k) => !k.startsWith("is_") && !k.startsWith("oos_"));
  const isKeys       = Object.keys(metrics).filter((k) => k.startsWith("is_"));
  const oosKeys      = Object.keys(metrics).filter((k) => k.startsWith("oos_"));

  return (
    <div className="space-y-3">
      {hasWF && (
        <div className="flex gap-2 text-xs">
          <span className="rounded bg-sky-900/60 px-2 py-0.5 text-sky-300">IS = in-sample</span>
          <span className="rounded bg-amber-900/60 px-2 py-0.5 text-amber-300">OOS = out-of-sample</span>
        </div>
      )}

      {/* Combined */}
      <div>
        {hasWF && <p className="mb-1 text-xs text-gray-500">Combined</p>}
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {combinedKeys.map((k) => (
            <MetricCard key={k} label={k} value={fmtMetric(k, metrics[k])} />
          ))}
        </div>
      </div>

      {/* IS / OOS side by side */}
      {hasWF && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <p className="mb-1 text-xs text-sky-400">In-sample</p>
            <div className="grid grid-cols-2 gap-2">
              {isKeys.map((k) => (
                <MetricCard key={k} label={k.replace("is_", "")} value={fmtMetric(k, metrics[k])} />
              ))}
            </div>
          </div>
          <div>
            <p className="mb-1 text-xs text-amber-400">Out-of-sample</p>
            <div className="grid grid-cols-2 gap-2">
              {oosKeys.map((k) => (
                <MetricCard key={k} label={k.replace("oos_", "")} value={fmtMetric(k, metrics[k])} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

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

type ChartAccum = {
  candles: any[];
  indicators: Record<string, any>;
  markers: any[];
  events?: any[];
  has_more: boolean;
  bar_count: number;
};

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
  const [walkForwardRatio, setWalkForwardRatio] = useState(0);
  const [windowSize, setWindowSize] = useState(0);
  const [startingCapital, setStartingCapital] = useState(1.0);
  const [starting, setStarting] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  type RiskOverride = {
    risk_type: "fixed" | "percent_equity" | "atr";
    position_size: number;
    risk_pct: number;
    atr_multiplier: number;
    sl_pct: number;
    tp_pct: number;
    slippage_pct: number;
    commission_pct: number;
    max_positions: number;
    daily_loss_limit_pct: number;
    cooldown_bars: number;
    trailing_stop: boolean;
    trailing_atr_multiplier: number;
    trailing_clip_with_price: boolean;
    trailing_only_in_profit: boolean;
  };
  const DEFAULT_RISK: RiskOverride = {
    risk_type: "fixed", position_size: 1.0, risk_pct: 0.01, atr_multiplier: 2.0,
    sl_pct: 0.02, tp_pct: 0.04, slippage_pct: 0.0005, commission_pct: 0.001,
    max_positions: 1, daily_loss_limit_pct: 0.0, cooldown_bars: 0,
    trailing_stop: false, trailing_atr_multiplier: 3.0, trailing_clip_with_price: false,
    trailing_only_in_profit: true,
  };
  const [risk, setRisk] = useState<RiskOverride>(DEFAULT_RISK);
  const patchRisk = (p: Partial<RiskOverride>) => setRisk((r) => ({ ...r, ...p }));

  const [selectedRun, setSelectedRun] = useState<number | null>(null);
  const [overlaySlip, setOverlaySlip] = useState(0.0005);
  const [overlayComm, setOverlayComm] = useState(0.001);
  const [overlayResult, setOverlayResult] = useState<{ metrics: Record<string, number>; orig_metrics: Record<string, number>; orig_slip: number; orig_comm: number } | null>(null);
  const overlayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
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

  const selectedRunObj = runs?.find((r: any) => r.id === selectedRun);

  const { data: equityCurve } = useSWR(
    selectedRun && selectedRunObj?.status === "completed"
      ? `/api/v1/strategies/${id}/runs/${selectedRun}/equity`
      : null,
    fetcher
  );

  // Reset overlay when switching runs; seed sliders from original costs when result loads
  useEffect(() => {
    setOverlayResult(null);
  }, [selectedRun]);

  useEffect(() => {
    if (overlayResult && overlayResult.orig_slip !== undefined) {
      // Seed sliders to original costs on first load (only if user hasn't moved them yet)
      setOverlaySlip((s) => (s === 0.0005 ? overlayResult.orig_slip : s));
      setOverlayComm((c) => (c === 0.001  ? overlayResult.orig_comm : c));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overlayResult?.orig_slip]);

  function fetchOverlay(runId: number, slip: number, comm: number) {
    if (overlayTimerRef.current) clearTimeout(overlayTimerRef.current);
    overlayTimerRef.current = setTimeout(async () => {
      try {
        const res = await apiFetch(`/api/v1/strategies/${id}/runs/${runId}/cost-overlay`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ slippage_pct: slip, commission_pct: comm }),
        });
        if (res.ok) {
          const body = await res.json();
          setOverlayResult(body.data);
        }
      } catch { /* non-fatal */ }
    }, 300);
  }

  // Chart time range — ISO date strings (datetime-local value), empty = full range
  const [chartFrom, setChartFrom] = useState("");
  const [chartTo, setChartTo] = useState("");
  const [appliedRange, setAppliedRange] = useState<{ from: string; to: string }>({ from: "", to: "" });
  const [chartMode, setChartMode] = useState<"indicators" | "conditions">("indicators");

  const [chartAccumulated, setChartAccumulated] = useState<ChartAccum | null>(null);
  const [chartLoadingOlder, setChartLoadingOlder] = useState(false);

  const chartUrl = useMemo(() => {
    if (!selectedRun || selectedRunObj?.status !== "completed") return null;
    const base = `/api/v1/strategies/${id}/runs/${selectedRun}/chart-data`;
    const params = new URLSearchParams();
    if (appliedRange.from) params.set("from_ts", String(Math.floor(new Date(appliedRange.from).getTime() / 1000)));
    if (appliedRange.to)   params.set("to_ts",   String(Math.floor(new Date(appliedRange.to).getTime()   / 1000)));
    const qs = params.toString();
    return qs ? `${base}?${qs}` : base;
  }, [id, selectedRun, selectedRunObj?.status, appliedRange]);

  const { data: freshChartData, isLoading: chartLoading } = useSWR(
    chartUrl,
    fetcher,
    { revalidateOnFocus: false },
  );

  useEffect(() => { setChartAccumulated(null); }, [chartUrl]);
  useEffect(() => { if (freshChartData) setChartAccumulated(freshChartData); }, [freshChartData]);

  async function loadOlderBarsInline() {
    if (!chartAccumulated?.candles.length || chartLoadingOlder || !selectedRun) return;
    const earliestTs = Math.min(...chartAccumulated.candles.map((c: any) => c.time));
    const base = `/api/v1/strategies/${id}/runs/${selectedRun}/chart-data`;
    setChartLoadingOlder(true);
    try {
      const res = await apiFetch(`${base}?to_ts=${earliestTs - 1}&limit=2000`);
      if (!res.ok) return;
      const body = await res.json();
      const older: ChartAccum = body.data ?? body;
      setChartAccumulated((prev) => {
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
      setChartLoadingOlder(false);
    }
  }

  const [stopping, setStopping] = useState<number | null>(null);
  const [deletingRun, setDeletingRun] = useState<number | null>(null);
  const [deletingStrategy, setDeletingStrategy] = useState(false);
  const [copying, setCopying] = useState(false);

  const [editingDef, setEditingDef] = useState(false);
  const [savingDef, setSavingDef] = useState(false);
  const [defError, setDefError] = useState<string | null>(null);
  const editedDefRef = useRef<object>({});

  const [showValidate, setShowValidate] = useState(false);
  const [validateDatasetId, setValidateDatasetId] = useState("");
  const [validateLimitBars, setValidateLimitBars] = useState("3000");
  const [disabledConds, setDisabledConds] = useState<Record<string, number[]>>({});
  const validateAbortRef = useRef<AbortController | null>(null);
  const [validating, setValidating] = useState(false);
  const [validateCandles, setValidateCandles] = useState<any[]>([]);
  const [validateIndicators, setValidateIndicators] = useState<any>({});
  const [validateMarkers, setValidateMarkers] = useState<any[]>([]);
  const [validateCounts, setValidateCounts] = useState<Record<string, number>>({});
  const [validateError, setValidateError] = useState<string | null>(null);

  function signalsToMarkers(signals: Record<string, number[]>) {
    const defs = [
      { key: "long_entry",  color: "#22c55e", position: "belowBar" as const, label: "Long entry" },
      { key: "long_exit",   color: "#38bdf8", position: "aboveBar" as const, label: "Long exit" },
      { key: "short_entry", color: "#ef4444", position: "aboveBar" as const, label: "Short entry" },
      { key: "short_exit",  color: "#f97316", position: "belowBar" as const, label: "Short exit" },
    ];
    const out: any[] = [];
    for (const { key, color, position, label } of defs)
      for (const ts of (signals[key] ?? []))
        out.push({ time: ts, position, color, shape: "circle", text: label });
    return out.sort((a, b) => a.time - b.time);
  }

  function toggleCond(blockKey: string, idx: number) {
    setDisabledConds((prev) => {
      const arr = prev[blockKey] ?? [];
      const next = arr.includes(idx) ? arr.filter((i) => i !== idx) : [...arr, idx];
      return { ...prev, [blockKey]: next };
    });
  }

  function buildFilteredDef() {
    const def = JSON.parse(JSON.stringify(strategy?.definition ?? {}));
    const blockMap: Record<string, [string, string]> = {
      long_entry: ["long", "entry"], long_exit: ["long", "exit"],
      short_entry: ["short", "entry"], short_exit: ["short", "exit"],
    };
    for (const [key, [dir, phase]] of Object.entries(blockMap)) {
      const disabled = disabledConds[key] ?? [];
      if (disabled.length > 0 && def[dir]?.[phase]?.conditions) {
        def[dir][phase].conditions = def[dir][phase].conditions.filter((_: any, i: number) => !disabled.includes(i));
      }
    }
    return def;
  }

  const validateCandlesRef = useRef<any[]>([]);
  validateCandlesRef.current = validateCandles;

  useEffect(() => {
    if (!validateDatasetId || validateCandlesRef.current.length === 0) return;
    const t = setTimeout(() => runValidate(true), 350);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disabledConds]);

  async function runValidate(soft = false) {
    if (!validateDatasetId) return;
    validateAbortRef.current?.abort();
    setValidating(true);
    setValidateError(null);
    if (!soft) { setValidateCandles([]); setValidateIndicators({}); }
    setValidateMarkers([]); setValidateCounts({});
    try {
      const limitBars = validateLimitBars === "all" ? undefined : parseInt(validateLimitBars);
      const res = await apiFetch(`/api/v1/strategies/${id}/validate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: parseInt(validateDatasetId), definition: buildFilteredDef(), ...(limitBars ? { limit_bars: limitBars } : {}) }),
      });
      if (!res.ok || !res.body) { setValidateError(`Error ${res.status}`); return; }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n"); buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let evt: any; try { evt = JSON.parse(line.slice(6)); } catch { continue; }
          if (evt.type === "init") { setValidateCandles(evt.candles ?? []); setValidateIndicators(evt.indicators ?? {}); }
          else if (evt.type === "signals") {
            setValidateMarkers(signalsToMarkers(evt));
            setValidateCounts({ long_entry: (evt.long_entry ?? []).length, long_exit: (evt.long_exit ?? []).length, short_entry: (evt.short_entry ?? []).length, short_exit: (evt.short_exit ?? []).length });
          } else if (evt.type === "error") { setValidateError(evt.message ?? "Error"); break; }
        }
      }
    } catch (e: any) { setValidateError(e.message ?? "Unknown error"); }
    finally { setValidating(false); }
  }

  function startEditDef() {
    setDefError(null);
    setEditingDef(true);
  }

  async function saveDef() {
    setDefError(null);
    setSavingDef(true);
    try {
      const res = await apiFetch(`/api/v1/strategies/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ definition: editedDefRef.current }),
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
      const res = await apiFetch(`/api/v1/strategies/${id}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode,
          dataset_id: datasetId ? parseInt(datasetId) : null,
          walk_forward_ratio: walkForwardRatio > 0 ? walkForwardRatio / 100 : null,
          window_size: windowSize > 0 ? windowSize : null,
          starting_capital: startingCapital !== 1.0 ? startingCapital : null,
          risk_override: risk,
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

  async function copyStrategy() {
    setCopying(true);
    try {
      const res = await apiFetch(`/api/v1/strategies/${id}/copy`, { method: "POST" });
      if (!res.ok) { toast("Failed to copy strategy", "error"); return; }
      const body = await res.json();
      toast("Strategy copied", "success");
      window.location.href = `/strategy/${body.data.id}`;
    } finally {
      setCopying(false);
    }
  }

  async function deleteStrategy() {
    if (!confirm(`Delete "${strategy.name}" and all its runs? This cannot be undone.`)) return;
    setDeletingStrategy(true);
    try {
      const res = await apiFetch(`/api/v1/strategies/${id}`, { method: "DELETE" });
      if (!res.ok) {
        toast("Failed to delete strategy", "error");
        return;
      }
      toast("Strategy deleted", "success");
      window.location.href = "/strategy";
    } finally {
      setDeletingStrategy(false);
    }
  }

  async function deleteRun(runId: number) {
    if (!confirm("Delete this run and all its results? This cannot be undone.")) return;
    setDeletingRun(runId);
    try {
      const res = await apiFetch(`/api/v1/strategies/${id}/runs/${runId}`, { method: "DELETE" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        toast(body.detail ?? "Failed to delete run", "error");
        return;
      }
      if (selectedRun === runId) setSelectedRun(null);
      toast("Run deleted", "success");
      mutate(`/api/v1/strategies/${id}/runs`);
    } finally {
      setDeletingRun(null);
    }
  }

  async function stopRun(runId: number) {
    setStopping(runId);
    try {
      await apiFetch(`/api/v1/strategies/${id}/runs/${runId}/stop`, { method: "POST" });
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
        <div className="flex items-center gap-3">
          <button
            onClick={copyStrategy}
            disabled={copying}
            className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800 disabled:opacity-50"
          >
            {copying ? "Copying…" : "Copy Strategy"}
          </button>
          <button
            onClick={deleteStrategy}
            disabled={deletingStrategy}
            className="rounded border border-red-800 px-3 py-1.5 text-xs text-red-400 hover:bg-red-900/30 disabled:opacity-50"
          >
            {deletingStrategy ? "Deleting…" : "Delete Strategy"}
          </button>
          <a href="/strategy" className="text-sm text-gray-400 hover:text-white">
            ← Back
          </a>
        </div>
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
            <StrategyDefinitionView definition={strategy.definition} />
            <div className="mt-2 flex gap-3">
              <button onClick={startEditDef} className="text-xs text-gray-400 hover:text-white">
                Edit definition
              </button>
              <button
                onClick={() => { setShowValidate(!showValidate); setValidateCandles([]); setValidateMarkers([]); setValidateError(null); }}
                className="text-xs text-sky-400 hover:text-sky-300"
              >
                {showValidate ? "Hide validate" : "Validate →"}
              </button>
            </div>

            {showValidate && (
              <div className="mt-3 space-y-3">
                {/* Condition toggles — derived from saved definition */}
                {(() => {
                  const def = strategy?.definition ?? {};
                  const blocks = [
                    { key: "long_entry",  label: "Long entry",  color: "#22c55e", conds: def.long?.entry?.conditions ?? [] },
                    { key: "long_exit",   label: "Long exit",   color: "#38bdf8", conds: def.long?.exit?.conditions  ?? [] },
                    { key: "short_entry", label: "Short entry", color: "#ef4444", conds: def.short?.entry?.conditions ?? [] },
                    { key: "short_exit",  label: "Short exit",  color: "#f97316", conds: def.short?.exit?.conditions  ?? [] },
                  ].filter((b) => b.conds.length > 0);
                  if (blocks.length === 0) return null;
                  return (
                    <div className="rounded border border-gray-700 bg-gray-900 p-3 space-y-2">
                      {blocks.map(({ key, label, color, conds }) => (
                        <div key={key}>
                          <p className="text-xs font-medium mb-1" style={{ color }}>{label}</p>
                          <div className="space-y-0.5 pl-2">
                            {conds.map((c: any, i: number) => {
                              const isDisabled = (disabledConds[key] ?? []).includes(i);
                              const label = c.type === "streak" ? `streak(${c.left} ${c.op} ${c.right}, ${c.min_streak})`
                                : c.type === "group_ref" ? `group: ${c.group_id}`
                                : c.type === "ml_signal" ? "ML signal"
                                : c.type === "llm_signal" ? "LLM signal"
                                : `${c.left} ${c.op} ${c.right}`;
                              return (
                                <label key={i} className="flex items-center gap-2 text-xs cursor-pointer select-none">
                                  <input type="checkbox" checked={!isDisabled} onChange={() => toggleCond(key, i)} className="accent-sky-500 shrink-0" />
                                  <span className={isDisabled ? "text-gray-600 line-through" : "text-gray-300"}>{label}</span>
                                </label>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  );
                })()}
                <div className="flex items-center gap-2 flex-wrap">
                  <select
                    value={validateDatasetId}
                    onChange={(e) => { setValidateDatasetId(e.target.value); setValidateCandles([]); setValidateMarkers([]); }}
                    className="rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                  >
                    <option value="">Select dataset…</option>
                    {datasets?.map((d: any) => (
                      <option key={d.id} value={d.id}>{d.name} ({d.row_count?.toLocaleString()} rows)</option>
                    ))}
                  </select>
                  <select
                    value={validateLimitBars}
                    onChange={(e) => setValidateLimitBars(e.target.value)}
                    className="rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                  >
                    <option value="1000">Last 1 000 bars</option>
                    <option value="3000">Last 3 000 bars</option>
                    <option value="10000">Last 10 000 bars</option>
                    <option value="all">All bars</option>
                  </select>
                  <button
                    onClick={() => runValidate()}
                    disabled={!validateDatasetId || validating}
                    className="rounded bg-sky-700 px-3 py-1.5 text-xs text-white hover:bg-sky-600 disabled:opacity-50"
                  >
                    {validating ? "Loading…" : "Run"}
                  </button>
                </div>
                {validating && <div className="h-0.5 w-full rounded bg-gray-700 overflow-hidden"><div className="h-full bg-sky-500 animate-pulse w-full" /></div>}
                {validateError && <p className="text-xs text-red-400">{validateError}</p>}
                {Object.keys(validateCounts).some((k) => (validateCounts[k] ?? 0) > 0) && (
                  <p className="text-xs text-gray-400 flex gap-3 flex-wrap">
                    {validateCounts.long_entry > 0 && <span><span className="inline-block w-2 h-2 rounded-full bg-green-500 mr-1" />Long entry: {validateCounts.long_entry}</span>}
                    {validateCounts.long_exit > 0 && <span><span className="inline-block w-2 h-2 rounded-full bg-sky-400 mr-1" />Long exit: {validateCounts.long_exit}</span>}
                    {validateCounts.short_entry > 0 && <span><span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-1" />Short entry: {validateCounts.short_entry}</span>}
                    {validateCounts.short_exit > 0 && <span><span className="inline-block w-2 h-2 rounded-full bg-orange-500 mr-1" />Short exit: {validateCounts.short_exit}</span>}
                  </p>
                )}
                {validateCandles.length > 0 && (
                  <StrategyChart
                    candles={validateCandles}
                    indicators={validateIndicators}
                    markers={validateMarkers}
                    defaultZoom={1}
                  />
                )}
              </div>
            )}
          </>
        )}
        {editingDef && (
          <div className="space-y-3">
            <div className="rounded border border-gray-700 bg-gray-950 p-4">
              <StrategyEditor
                initialDefinition={strategy.definition}
                onChange={(def) => { editedDefRef.current = def; }}
                strategyId={id}
                datasets={datasets}
              />
            </div>
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
              <>
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
                <div>
                  <label className="mb-1 block text-xs text-gray-400">
                    Walk-forward split — in-sample:{" "}
                    <span className="text-white font-mono">
                      {walkForwardRatio === 0 ? "disabled" : `${walkForwardRatio}% IS / ${100 - walkForwardRatio}% OOS`}
                    </span>
                  </label>
                  <input
                    type="range" min={0} max={90} step={5} value={walkForwardRatio}
                    onChange={(e) => setWalkForwardRatio(Number(e.target.value))}
                    className="w-full accent-sky-500"
                  />
                  <p className="mt-0.5 text-xs text-gray-600">
                    0 = off. Splits data chronologically; metrics reported separately for each half.
                  </p>
                </div>
                <div>
                  <label className="mb-1 block text-xs text-gray-400">
                    Rolling indicator window —{" "}
                    <span className="text-white font-mono">
                      {windowSize === 0 ? "disabled (full history)" : `${windowSize} bars`}
                    </span>
                  </label>
                  <input
                    type="number" min={0} step={10} value={windowSize}
                    onChange={(e) => setWindowSize(Math.max(0, parseInt(e.target.value) || 0))}
                    className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none"
                  />
                  <p className="mt-0.5 text-xs text-gray-600">
                    0 = off. Set to 100+ for Renko strategies — recomputes indicators on a rolling window to match live MT5 behaviour (slower).
                  </p>
                </div>
                <div>
                  <label className="mb-1 block text-xs text-gray-400">
                    Starting capital —{" "}
                    <span className="text-white font-mono">
                      {startingCapital === 1.0 ? "1.0 (normalised)" : startingCapital.toLocaleString()}
                    </span>
                  </label>
                  <input
                    type="number" min={1} step={1000} value={startingCapital}
                    onChange={(e) => setStartingCapital(Math.max(1, parseFloat(e.target.value) || 1.0))}
                    className="w-28 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none"
                  />
                  <p className="mt-0.5 text-xs text-gray-600">
                    1.0 = normalised %. Set to actual capital (e.g. 1000000 JPY) to compare PnL with trade_strategy.
                  </p>
                </div>
              </>
            )}
            {/* Risk */}
            <div className="border-t border-gray-800 pt-3 space-y-2">
              <p className="text-xs font-medium text-gray-400">Risk</p>
              <div className="flex items-center gap-2">
                <span className="w-32 shrink-0 text-xs text-gray-500">Sizing method</span>
                <select value={risk.risk_type} onChange={(e) => patchRisk({ risk_type: e.target.value as RiskOverride["risk_type"] })}
                  className="rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none">
                  <option value="fixed">Fixed lot</option>
                  <option value="percent_equity">% of equity per trade</option>
                  <option value="atr">ATR-based</option>
                </select>
              </div>
              {risk.risk_type === "fixed" && (
                <div className="flex items-center gap-2">
                  <span className="w-32 shrink-0 text-xs text-gray-500">Position size</span>
                  <input type="number" value={risk.position_size} step={0.1} min={0} onChange={(e) => patchRisk({ position_size: parseFloat(e.target.value) || 0 })}
                    className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none" />
                  <span className="text-xs text-gray-600">fraction of equity</span>
                </div>
              )}
              {(risk.risk_type === "percent_equity" || risk.risk_type === "atr") && (
                <div className="flex items-center gap-2">
                  <span className="w-32 shrink-0 text-xs text-gray-500">Risk per trade</span>
                  <input type="number" value={risk.risk_pct} step={0.005} min={0} onChange={(e) => patchRisk({ risk_pct: parseFloat(e.target.value) || 0 })}
                    className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none" />
                  <span className="text-xs text-gray-600">{(risk.risk_pct * 100).toFixed(1)}% of equity</span>
                </div>
              )}
              {risk.risk_type === "atr" && (
                <div className="flex items-center gap-2">
                  <span className="w-32 shrink-0 text-xs text-gray-500">ATR multiplier</span>
                  <input type="number" value={risk.atr_multiplier} step={0.5} min={0.5} onChange={(e) => patchRisk({ atr_multiplier: parseFloat(e.target.value) || 0 })}
                    className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none" />
                </div>
              )}
              {[
                { label: "Stop-loss",   key: "sl_pct" as const,   step: 0.005 },
                { label: "Take-profit", key: "tp_pct" as const,   step: 0.005 },
                { label: "Slippage",    key: "slippage_pct" as const,  step: 0.0001 },
                { label: "Commission",  key: "commission_pct" as const, step: 0.0001 },
              ].map(({ label, key, step }) => (
                <div key={key} className="flex items-center gap-2">
                  <span className="w-32 shrink-0 text-xs text-gray-500">{label}</span>
                  <input type="number" value={risk[key]} step={step} min={0} onChange={(e) => patchRisk({ [key]: parseFloat(e.target.value) || 0 })}
                    className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none" />
                  <span className="text-xs text-gray-600">{(risk[key] * 100).toFixed(3)}%</span>
                </div>
              ))}
              <div className="flex items-center gap-2">
                <span className="w-32 shrink-0 text-xs text-gray-500">Max positions</span>
                <input type="number" value={risk.max_positions} step={1} min={1} onChange={(e) => patchRisk({ max_positions: parseInt(e.target.value) || 1 })}
                  className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none" />
              </div>
              <div className="flex items-center gap-2">
                <span className="w-32 shrink-0 text-xs text-gray-500">Daily loss limit</span>
                <input type="number" value={risk.daily_loss_limit_pct} step={0.005} min={0} onChange={(e) => patchRisk({ daily_loss_limit_pct: parseFloat(e.target.value) || 0 })}
                  className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none" />
                <span className="text-xs text-gray-600">{risk.daily_loss_limit_pct === 0 ? "disabled" : `${(risk.daily_loss_limit_pct * 100).toFixed(1)}%`}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-32 shrink-0 text-xs text-gray-500">Cooldown after loss</span>
                <input type="number" value={risk.cooldown_bars} step={1} min={0} onChange={(e) => patchRisk({ cooldown_bars: parseInt(e.target.value) || 0 })}
                  className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none" />
                <span className="text-xs text-gray-600">bars</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-32 shrink-0 text-xs text-gray-500">Trailing stop</span>
                <input type="checkbox" checked={risk.trailing_stop} onChange={(e) => patchRisk({ trailing_stop: e.target.checked })}
                  className="rounded border border-gray-700 bg-gray-800" />
                <span className="text-xs text-gray-600">ATR-based (requires ATR indicator)</span>
              </div>
              {risk.trailing_stop && (
                <>
                  <div className="flex items-center gap-2">
                    <span className="w-32 shrink-0 text-xs text-gray-500">Trail ATR mult.</span>
                    <input type="number" value={risk.trailing_atr_multiplier} step={0.5} min={0.5} onChange={(e) => patchRisk({ trailing_atr_multiplier: parseFloat(e.target.value) || 3.0 })}
                      className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none" />
                    <span className="text-xs text-gray-600">× ATR distance from close</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-32 shrink-0 text-xs text-gray-500">Clip with price</span>
                    <input type="checkbox" checked={risk.trailing_clip_with_price} onChange={(e) => patchRisk({ trailing_clip_with_price: e.target.checked })}
                      className="rounded border border-gray-700 bg-gray-800" />
                    <span className="text-xs text-gray-600">Floor stop at entry price (breakeven)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-32 shrink-0 text-xs text-gray-500">Only in profit</span>
                    <input type="checkbox" checked={risk.trailing_only_in_profit} onChange={(e) => patchRisk({ trailing_only_in_profit: e.target.checked })}
                      className="rounded border border-gray-700 bg-gray-800" />
                    <span className="text-xs text-gray-600">Only move stop when position is profitable (matches trade_strategy)</span>
                  </div>
                </>
              )}
            </div>

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
          <div className="rounded-lg border border-dashed border-gray-700 px-6 py-10 text-center">
            <p className="text-gray-300 font-medium text-sm mb-1">No runs yet</p>
            <p className="text-gray-500 text-xs mb-4">Start a backtest to see results here.</p>
            {!showRunForm && (
              <button
                onClick={() => setShowRunForm(true)}
                className="rounded bg-brand-500 px-3 py-1.5 text-xs text-white hover:bg-sky-400"
              >
                Start first run
              </button>
            )}
          </div>
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
                    <span className="text-gray-500 mr-3">{selectedRun === run.id ? "▲ Hide" : "▼ Show"}</span>
                    {!["pending", "running"].includes(run.status) && (
                      <button
                        onClick={(e) => { e.stopPropagation(); deleteRun(run.id); }}
                        disabled={deletingRun === run.id}
                        className="text-red-500 hover:text-red-400 disabled:opacity-50"
                      >
                        {deletingRun === run.id ? "…" : "Delete"}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {/* AI Investigation */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-medium uppercase tracking-wide text-gray-400">AI Investigation</h2>
          <p className="text-xs text-gray-600">Automatically sweeps risk parameters and reports findings</p>
        </div>
        <InvestigationPanel strategyId={id} datasets={datasets ?? []} />
      </section>

      {/* Run detail panel */}
      {selectedRun && (
        <section className="space-y-4">
          <h2 className="text-sm font-medium uppercase tracking-wide text-gray-400">
            Run #{selectedRun} — Details
          </h2>

          {/* Price chart */}
          {selectedRunObj?.status === "completed" && (
            <div className="space-y-2">
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="text-xs font-medium text-gray-400">Price Chart</h3>
                <span className="text-gray-600 text-xs">|</span>
                <div className="flex rounded border border-gray-700 overflow-hidden text-xs">
                  {(["indicators", "conditions"] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setChartMode(mode)}
                      className={`px-2 py-0.5 capitalize ${chartMode === mode ? "bg-brand-500 text-white" : "bg-gray-900 text-gray-400 hover:text-gray-200"}`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
                <span className="text-gray-600 text-xs">|</span>
                <span className="text-xs text-gray-500">Range:</span>
                <input
                  type="datetime-local"
                  value={chartFrom}
                  onChange={(e) => setChartFrom(e.target.value)}
                  className="rounded border border-gray-700 bg-gray-900 px-2 py-0.5 text-xs text-gray-300 focus:outline-none focus:ring-1 focus:ring-brand-500"
                />
                <span className="text-gray-600 text-xs">→</span>
                <input
                  type="datetime-local"
                  value={chartTo}
                  onChange={(e) => setChartTo(e.target.value)}
                  className="rounded border border-gray-700 bg-gray-900 px-2 py-0.5 text-xs text-gray-300 focus:outline-none focus:ring-1 focus:ring-brand-500"
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
                  {chartAccumulated?.has_more && (
                    <button
                      onClick={loadOlderBarsInline}
                      disabled={chartLoadingOlder}
                      className="text-xs text-sky-400 hover:text-sky-300 disabled:opacity-50"
                    >
                      {chartLoadingOlder ? "Loading…" : "← Load older"}
                    </button>
                  )}
                  {chartAccumulated && (
                    <span className="text-xs text-gray-600">
                      {chartAccumulated.bar_count ?? chartAccumulated.candles?.length ?? 0} bars
                      {chartAccumulated.has_more && <span className="text-gray-700"> · more available</span>}
                    </span>
                  )}
                </div>
              </div>
              {chartLoading && (
                <div className="flex items-center justify-center h-16 text-gray-500 text-sm">Loading chart…</div>
              )}
              {chartAccumulated && (
                <StrategyChart
                  candles={chartAccumulated.candles ?? []}
                  indicators={Object.fromEntries(
                    Object.entries(chartAccumulated.indicators ?? {}).filter(([, s]: [string, any]) =>
                      chartMode === "conditions"
                        ? true
                        : !(s.group?.startsWith("cond:") || s.line_style === "dashed" || s.line_style === "step")
                    )
                  )}
                  markers={chartAccumulated.markers ?? []}
                />
              )}
            </div>
          )}

          {runMetrics && Object.keys(runMetrics).length > 0 && (
            <MetricsPanel metrics={runMetrics} />
          )}

          {equityCurve && equityCurve.length > 0 && (
            <div className="rounded-lg border border-gray-800 p-3">
              <p className="text-xs font-medium text-gray-400 mb-2">Equity Curve</p>
              <EquityChart data={equityCurve} />
            </div>
          )}

          {/* Cost Sensitivity Analysis */}
          {selectedRun && selectedRunObj?.status === "completed" && (
            <div className="rounded-lg border border-gray-800 p-3 space-y-3">
              <p className="text-xs font-medium text-gray-400">Cost Sensitivity</p>
              <div className="flex flex-wrap gap-4">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 w-24">Slippage /side</span>
                  <input
                    type="number" step={0.0001} min={0} value={overlaySlip}
                    onChange={(e) => { const v = parseFloat(e.target.value) || 0; setOverlaySlip(v); fetchOverlay(selectedRun, v, overlayComm); }}
                    className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none"
                  />
                  <span className="text-xs text-gray-600">{(overlaySlip * 100).toFixed(3)}%</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 w-24">Commission</span>
                  <input
                    type="number" step={0.0001} min={0} value={overlayComm}
                    onChange={(e) => { const v = parseFloat(e.target.value) || 0; setOverlayComm(v); fetchOverlay(selectedRun, overlaySlip, v); }}
                    className="w-20 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none"
                  />
                  <span className="text-xs text-gray-600">{(overlayComm * 100).toFixed(3)}%</span>
                </div>
                {!overlayResult && (
                  <button
                    onClick={() => fetchOverlay(selectedRun, overlaySlip, overlayComm)}
                    className="rounded border border-gray-700 px-2 py-1 text-xs text-gray-400 hover:text-white"
                  >
                    Calculate
                  </button>
                )}
              </div>
              {overlayResult && (
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {(["total_pnl", "win_rate", "profit_factor", "total_trades"] as const).map((k) => {
                    const orig = overlayResult.orig_metrics[k] ?? 0;
                    const adj  = overlayResult.metrics[k] ?? 0;
                    const diff = adj - orig;
                    const diffColor = diff > 0 ? "text-green-400" : diff < 0 ? "text-red-400" : "text-gray-500";
                    return (
                      <div key={k} className="rounded bg-gray-900 px-3 py-2">
                        <p className="text-xs text-gray-500">{k.replace(/_/g, " ")}</p>
                        <p className="text-sm font-mono text-white">{fmtMetric(k, adj)}</p>
                        {diff !== 0 && (
                          <p className={`text-xs font-mono ${diffColor}`}>
                            {diff > 0 ? "+" : ""}{fmtMetric(k, diff)}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {runTrades && runTrades.length > 0 && (
            <div>
              <h3 className="mb-2 text-xs text-gray-500">Trades ({runTrades.length})</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-gray-800 text-gray-500 uppercase">
                      <th className="py-1 pr-3">Phase</th>
                      <th className="py-1 pr-3">Symbol</th>
                      <th className="py-1 pr-3">Dir</th>
                      <th className="py-1 pr-3">Entry</th>
                      <th className="py-1 pr-3">Exit</th>
                      <th className="py-1 pr-3">SL</th>
                      <th className="py-1 pr-3">TP</th>
                      <th className="py-1 pr-3">PnL</th>
                      <th className="py-1 pr-3">MAE</th>
                      <th className="py-1 pr-3">MFE</th>
                      <th className="py-1 pr-3">Reason</th>
                      <th className="py-1">Opened</th>
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
                        <td className="py-1 pr-3">
                          {t.phase === "oos"
                            ? <span className="rounded px-1 bg-amber-900/50 text-amber-300 text-[10px]">OOS</span>
                            : t.phase === "is"
                            ? <span className="rounded px-1 bg-sky-900/50 text-sky-300 text-[10px]">IS</span>
                            : <span className="text-gray-600">—</span>}
                        </td>
                        <td className="py-1 pr-3">{t.symbol}</td>
                        <td className="py-1 pr-3">{t.direction}</td>
                        <td className="py-1 pr-3 font-mono">{t.entry_price?.toFixed(4)}</td>
                        <td className="py-1 pr-3 font-mono">{t.exit_price?.toFixed(4)}</td>
                        <td className="py-1 pr-3 font-mono text-gray-500">{t.sl_price?.toFixed(4) ?? "—"}</td>
                        <td className="py-1 pr-3 font-mono text-gray-500">{t.tp_price?.toFixed(4) ?? "—"}</td>
                        <td className="py-1 pr-3 font-mono font-medium">
                          {t.profit >= 0 ? "+" : ""}{(t.profit * 100).toFixed(2)}%
                        </td>
                        <td className="py-1 pr-3 font-mono text-gray-500">{t.mae != null ? `${(t.mae * 100).toFixed(2)}%` : "—"}</td>
                        <td className="py-1 pr-3 font-mono text-gray-500">{t.mfe != null ? `${(t.mfe * 100).toFixed(2)}%` : "—"}</td>
                        <td className="py-1 pr-3 text-gray-500 text-[10px]">{t.exit_reason ?? "—"}</td>
                        <td className="py-1 text-gray-400">
                          {new Date(t.opened_at).toLocaleDateString()}
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
