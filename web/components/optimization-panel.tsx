"use client";

import { useState } from "react";
import { apiFetch } from "@/lib/fetcher";
import { useToast } from "@/lib/toast";

interface TopResult {
  params: Record<string, any>;
  trades: number;
  win_rate: number;
  total_pnl: number;
  profit_factor: number;
  max_drawdown: number;
  sharpe_ratio: number;
}

interface CurvePoint {
  param_value: any;
  param_label: string;
  total_pnl: number;
  win_rate: number;
  profit_factor: number;
  trades: number;
  max_drawdown: number;
}

interface OptimizationData {
  primary_param: string;
  target_metric: string;
  top_results: TopResult[];
  sensitivity_curve: CurvePoint[];
  all_results: TopResult[];
}

interface Props {
  strategyId: string;
  datasets: { id: number; name: string; row_count: number }[];
  currentDefinition?: any;
  onStrategyUpdated?: () => void;
}

export function OptimizationPanel({
  strategyId,
  datasets,
  currentDefinition,
  onStrategyUpdated,
}: Props) {
  const { toast } = useToast();
  // Default to dataset with USDJPY or first available
  const defaultDs = datasets.find((d) => d.name.includes("USDJPY") && d.name.includes("Latest")) || datasets[0];
  const [datasetId, setDatasetId] = useState<string>(defaultDs ? String(defaultDs.id) : "");
  const [targetMetric, setTargetMetric] = useState("total_pnl");
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState<number | null>(null);
  const [data, setData] = useState<OptimizationData | null>(null);

  // Custom param inputs
  const isAnalyst = currentDefinition?.strategy_type?.startsWith("analyst_");
  const [slPipsInput, setSlPipsInput] = useState("15, 20, 25, 30, 40, 50, 60, 75, 100, 150, None");
  const [tpPipsInput, setTpPipsInput] = useState("50, 80, 100, 150, opposite");

  async function handleRunOptimization() {
    if (!datasetId) {
      toast("Please select a dataset first", "error");
      return;
    }

    setLoading(true);
    try {
      const param_ranges: Record<string, any[]> = {};
      if (isAnalyst) {
        param_ranges.sl_pips = slPipsInput
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean)
          .map((s) => (s.toLowerCase() === "none" ? null : parseFloat(s)));
        param_ranges.tp_pips = tpPipsInput
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean)
          .map((s) => (s.toLowerCase() === "none" ? null : s.toLowerCase() === "opposite" ? "opposite" : parseFloat(s)));
      } else {
        param_ranges.sl_pct = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05];
        param_ranges.tp_pct = [0.01, 0.02, 0.03, 0.04, 0.06];
      }

      const res = await apiFetch(`/api/v1/strategies/${strategyId}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: parseInt(datasetId),
          param_ranges,
          target_metric: targetMetric,
          primary_param: isAnalyst ? "sl_pips" : "sl_pct",
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Optimization failed" }));
        toast(err.detail || "Optimization failed", "error");
        return;
      }

      const json = await res.json();
      setData(json.data);
      toast("Parameter optimization completed!", "success");
    } catch (e: any) {
      toast(e.message || "Network error", "error");
    } finally {
      setLoading(false);
    }
  }

  async function handleApplyParams(params: Record<string, any>, rank: number) {
    setApplying(rank);
    try {
      const res = await apiFetch(`/api/v1/strategies/${strategyId}/apply-params`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params }),
      });
      if (res.ok) {
        toast(`Rank ${rank} parameters applied to strategy definition!`, "success");
        onStrategyUpdated?.();
      } else {
        toast("Failed to apply parameters", "error");
      }
    } catch (e: any) {
      toast(e.message || "Failed to apply parameters", "error");
    } finally {
      setApplying(null);
    }
  }

  // Render SVG sensitivity curve
  function renderSensitivityChart(curve: CurvePoint[]) {
    if (!curve || curve.length === 0) return null;

    const width = 720;
    const height = 260;
    const padding = { top: 25, right: 30, bottom: 45, left: 65 };

    const pnls = curve.map((c) => c.total_pnl);
    const minPnl = Math.min(0, ...pnls);
    const maxPnl = Math.max(0, ...pnls);
    const pnlRange = (maxPnl - minPnl) || 1;

    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const getX = (idx: number) => padding.left + (idx / Math.max(1, curve.length - 1)) * chartW;
    const getY = (pnl: number) => padding.top + chartH - ((pnl - minPnl) / pnlRange) * chartH;
    const zeroY = getY(0);

    const points = curve.map((c, i) => `${getX(i)},${getY(c.total_pnl)}`).join(" ");

    return (
      <div className="rounded-xl border border-gray-800 bg-gray-950 p-5 shadow-lg space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <span>📈</span>
              <span>Parameter Sensitivity Curve: Stop Loss (SL) vs Total PnL</span>
            </h4>
            <p className="text-xs text-gray-400">
              Shows how strategy profitability responds to varying Stop Loss levels.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <span className="flex items-center gap-1.5 text-emerald-400">
              <span className="inline-block w-3 h-0.5 bg-emerald-400 rounded"></span> Total PnL (JPY)
            </span>
            <span className="flex items-center gap-1.5 text-gray-400">
              <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-400/80"></span> Data Point
            </span>
          </div>
        </div>

        <div className="overflow-x-auto">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto min-w-[500px]">
            {/* Grid Lines */}
            <line
              x1={padding.left}
              y1={zeroY}
              x2={width - padding.right}
              y2={zeroY}
              stroke="#4b5563"
              strokeDasharray="4 4"
              strokeWidth="1.2"
            />
            <text x={padding.left - 8} y={zeroY + 4} textAnchor="end" fill="#9ca3af" fontSize="10" fontFamily="monospace">
              0
            </text>

            <line
              x1={padding.left}
              y1={padding.top}
              x2={width - padding.right}
              y2={padding.top}
              stroke="#1f2937"
              strokeWidth="1"
            />
            <text x={padding.left - 8} y={padding.top + 4} textAnchor="end" fill="#9ca3af" fontSize="10" fontFamily="monospace">
              {maxPnl >= 100000 ? `+${(maxPnl / 10000).toFixed(0)}万` : `+${maxPnl.toLocaleString()}`}
            </text>

            {minPnl < 0 && (
              <>
                <line
                  x1={padding.left}
                  y1={height - padding.bottom}
                  x2={width - padding.right}
                  y2={height - padding.bottom}
                  stroke="#1f2937"
                  strokeWidth="1"
                />
                <text x={padding.left - 8} y={height - padding.bottom + 4} textAnchor="end" fill="#9ca3af" fontSize="10" fontFamily="monospace">
                  {minPnl <= -100000 ? `${(minPnl / 10000).toFixed(0)}万` : minPnl.toLocaleString()}
                </text>
              </>
            )}

            {/* Gradient Fill under curve */}
            <defs>
              <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
              </linearGradient>
            </defs>
            <polygon
              points={`${getX(0)},${zeroY} ${points} ${getX(curve.length - 1)},${zeroY}`}
              fill="url(#pnlGrad)"
            />

            {/* Curve Line */}
            <polyline
              fill="none"
              stroke="#10b981"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={points}
            />

            {/* Dots and Labels */}
            {curve.map((c, i) => {
              const x = getX(i);
              const y = getY(c.total_pnl);
              const isProfit = c.total_pnl >= 0;
              return (
                <g key={i} className="cursor-pointer group">
                  <circle
                    cx={x}
                    cy={y}
                    r="4"
                    fill={isProfit ? "#10b981" : "#ef4444"}
                    stroke="#030712"
                    strokeWidth="2"
                    className="transition-all hover:r-6"
                  />
                  {/* X Axis Label */}
                  <text
                    x={x}
                    y={height - padding.bottom + 18}
                    textAnchor="middle"
                    fill="#9ca3af"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    {c.param_label}
                  </text>
                  {/* Tooltip on Hover */}
                  <title>{`${c.param_label}: PnL ${c.total_pnl >= 0 ? "+" : ""}${c.total_pnl.toLocaleString()} JPY (Win: ${(c.win_rate * 100).toFixed(1)}%, PF: ${c.profit_factor})`}</title>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* ── Control Header ────────────────────────────────────────── */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/50 p-5 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <span>⚡</span>
              <span>Automated Parameter Optimizer & Grid Search</span>
            </h3>
            <p className="text-xs text-gray-400">
              Sweep parameter combinations, extract Top 3 optimal setups, and analyze profitability curves.
            </p>
          </div>
          <button
            onClick={handleRunOptimization}
            disabled={loading || !datasetId}
            className="flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-5 py-2 text-sm font-medium text-white transition hover:bg-emerald-500 disabled:opacity-50 shadow-md shadow-emerald-900/20"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                <span>Searching Optimal Parameters...</span>
              </>
            ) : (
              <>
                <span>▶ Run Optimization</span>
              </>
            )}
          </button>
        </div>

        {/* Configurations Form */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 border-t border-gray-800">
          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1">Target Dataset</label>
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.row_count?.toLocaleString()} bars)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-400 mb-1">Target Optimization Metric</label>
            <select
              value={targetMetric}
              onChange={(e) => setTargetMetric(e.target.value)}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
            >
              <option value="total_pnl">Total PnL (Highest Profit)</option>
              <option value="profit_factor">Profit Factor (Reward / Risk)</option>
              <option value="sharpe_ratio">Sharpe Ratio (Risk-Adjusted)</option>
              <option value="win_rate">Win Rate (%)</option>
            </select>
          </div>

          {isAnalyst && (
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">Stop Loss Range (pips)</label>
              <input
                type="text"
                value={slPipsInput}
                onChange={(e) => setSlPipsInput(e.target.value)}
                placeholder="15, 20, 25, 30, 40, 50, 75, 100, None"
                className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none font-mono"
              />
            </div>
          )}
        </div>
      </div>

      {/* ── Results Section ────────────────────────────────────────── */}
      {data && (
        <div className="space-y-6 animate-fadeIn">
          {/* Top 3 Best Parameters Cards */}
          <div>
            <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <span>🏆</span>
              <span>Top 3 Optimal Parameter Configurations</span>
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {data.top_results.map((top, idx) => {
                const rankColors = [
                  { border: "border-amber-500/50", bg: "bg-amber-950/20", badge: "bg-amber-500 text-black", title: "🥇 1st Place (Best Overall)" },
                  { border: "border-slate-400/50", bg: "bg-slate-900/40", badge: "bg-slate-300 text-black", title: "🥈 2nd Place" },
                  { border: "border-amber-700/50", bg: "bg-amber-950/10", badge: "bg-amber-700 text-white", title: "🥉 3rd Place" },
                ][idx] || { border: "border-gray-800", bg: "bg-gray-900", badge: "bg-gray-700 text-white", title: `Rank ${idx + 1}` };

                const isProfit = top.total_pnl >= 0;

                return (
                  <div
                    key={idx}
                    className={`rounded-xl border ${rankColors.border} ${rankColors.bg} p-4 shadow-lg flex flex-col justify-between space-y-4`}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${rankColors.badge}`}>
                          Rank {idx + 1}
                        </span>
                        <span className="text-xs text-gray-400">{rankColors.title}</span>
                      </div>

                      {/* Parameter summary */}
                      <div className="rounded-lg bg-gray-950/70 p-2.5 my-2 border border-gray-800 font-mono text-xs space-y-1">
                        {Object.entries(top.params).map(([k, v]) => (
                          <div key={k} className="flex justify-between">
                            <span className="text-gray-400">{k}:</span>
                            <span className="text-white font-semibold">{v === null ? "None" : String(v)}</span>
                          </div>
                        ))}
                      </div>

                      {/* Performance metrics grid */}
                      <div className="grid grid-cols-2 gap-2 text-xs pt-1">
                        <div className="rounded bg-gray-900/60 p-2">
                          <p className="text-gray-500">Total PnL</p>
                          <p className={`font-mono font-bold text-sm ${isProfit ? "text-emerald-400" : "text-red-400"}`}>
                            {isProfit ? "+" : ""}{top.total_pnl.toLocaleString()} JPY
                          </p>
                        </div>
                        <div className="rounded bg-gray-900/60 p-2">
                          <p className="text-gray-500">Win Rate</p>
                          <p className="font-mono font-bold text-sm text-white">
                            {(top.win_rate * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="rounded bg-gray-900/60 p-2">
                          <p className="text-gray-500">Profit Factor</p>
                          <p className="font-mono font-bold text-sm text-white">
                            {top.profit_factor.toFixed(2)}
                          </p>
                        </div>
                        <div className="rounded bg-gray-900/60 p-2">
                          <p className="text-gray-500">Max Drawdown</p>
                          <p className="font-mono font-bold text-sm text-gray-300">
                            {(top.max_drawdown * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={() => handleApplyParams(top.params, idx + 1)}
                      disabled={applying === idx + 1}
                      className="w-full rounded-lg bg-gray-800 hover:bg-gray-700 text-xs font-medium text-white py-2 transition border border-gray-700 hover:border-emerald-500 flex items-center justify-center gap-1.5"
                    >
                      {applying === idx + 1 ? (
                        <span>Applying...</span>
                      ) : (
                        <>
                          <span>✓</span>
                          <span>Apply to Strategy</span>
                        </>
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Sensitivity Curve Chart */}
          {renderSensitivityChart(data.sensitivity_curve)}

          {/* All Results Table */}
          <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-4 space-y-3">
            <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Complete Grid Search Results ({data.all_results.length} Combinations)
            </h4>
            <div className="overflow-x-auto max-h-64 overflow-y-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead className="sticky top-0 bg-gray-950 text-gray-400 border-b border-gray-800">
                  <tr>
                    <th className="py-2 px-3">Parameters</th>
                    <th className="py-2 px-3">Trades</th>
                    <th className="py-2 px-3">Win Rate</th>
                    <th className="py-2 px-3">Total PnL</th>
                    <th className="py-2 px-3">Profit Factor</th>
                    <th className="py-2 px-3">Max DD</th>
                    <th className="py-2 px-3">Sharpe</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800 text-gray-300">
                  {data.all_results.map((r, i) => (
                    <tr key={i} className="hover:bg-gray-800/40">
                      <td className="py-1.5 px-3 text-white">
                        {Object.entries(r.params)
                          .map(([k, v]) => `${k}:${v === null ? "None" : v}`)
                          .join(", ")}
                      </td>
                      <td className="py-1.5 px-3">{r.trades}</td>
                      <td className="py-1.5 px-3">{(r.win_rate * 100).toFixed(1)}%</td>
                      <td className={`py-1.5 px-3 font-semibold ${r.total_pnl >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                        {r.total_pnl >= 0 ? "+" : ""}{r.total_pnl.toLocaleString()}
                      </td>
                      <td className="py-1.5 px-3">{r.profit_factor.toFixed(2)}</td>
                      <td className="py-1.5 px-3">{(r.max_drawdown * 100).toFixed(1)}%</td>
                      <td className="py-1.5 px-3">{r.sharpe_ratio.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
