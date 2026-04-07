"use client";

import { useState, useEffect, useRef } from "react";
import useSWR, { mutate } from "swr";
import { useParams, useRouter } from "next/navigation";
import { fetcher } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";
import { useToast } from "@/lib/toast";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

// ── Types ─────────────────────────────────────────────────────────────────────

type Tab = "overview" | "preview" | "characteristics";
type CharsTab = "endogenous" | "exogenous";
type SeasonalityView = "hour_day" | "hour_week" | "weekday_month" | "day_month" | "month_year";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type FullMetrics = Record<string, any>;

// Keys that the backend writes one-by-one into metrics; each maps to one chart section.
// Order matches the backend registry (fastest → slowest) so charts appear progressively.
const ENDOGENOUS_KEYS = ["return_dist", "ccdf", "diffusion", "acf", "vol_clustering", "qq"] as const;
const EXOGENOUS_KEYS = ["exogenous_jump_tail", "exogenous_cdf", "exogenous_rolling_mean", "exogenous_long_lag_acf", "exogenous_seasonality"] as const;
const ALL_CHAR_KEYS = [...ENDOGENOUS_KEYS, ...EXOGENOUS_KEYS] as const;

// ── Chart helpers ─────────────────────────────────────────────────────────────

const COLOR = "#60a5fa";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const NoShape = (_: any): React.ReactElement => <g />;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const SmallDot = ({ cx, cy, fill }: any): React.ReactElement => (
  <circle cx={cx} cy={cy} r={2.5} fill={fill} opacity={0.85} />
);

const CHART_STYLE = {
  contentStyle: { background: "#111827", border: "1px solid #374151", fontSize: 10 },
  labelStyle: { color: "#9ca3af" },
};

const DAY_COLORS = ["#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa", "#fb923c", "#22d3ee"];
const SEASON_COLORS = ["#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa", "#fb923c", "#22d3ee", "#4ade80"];

const SEASON_VIEWS: { id: SeasonalityView; label: string; desc: string }[] = [
  { id: "hour_day",      label: "Hour / Day",      desc: "Average by hour of day (0–23 h)" },
  { id: "hour_week",     label: "Hour / Week",      desc: "Intraday pattern across the full week" },
  { id: "weekday_month", label: "Weekday / Month",  desc: "Day-of-week pattern by week of month" },
  { id: "day_month",     label: "Day / Month",      desc: "Day-of-month (1–31) pattern by year" },
  { id: "month_year",    label: "Month / Year",     desc: "Monthly pattern by year" },
];

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{title}</h3>
      {children}
    </div>
  );
}

// ── Endogenous charts ─────────────────────────────────────────────────────────

function ReturnDistSection({ m }: { m: FullMetrics }) {
  const stats = m.stats;
  const lg = (v: number | null) => (v !== null && v > 0) ? Math.log10(v) : null;
  const histData = m.centers
    .map((x, j) => { const y = lg(m.hist[j]); return y !== null ? { x, y } : null; })
    .filter((d): d is { x: number; y: number } => d !== null);
  const normalData = m.centers
    .map((x, j) => { const y = lg(m.normal_pdf[j]); return y !== null ? { x, y } : null; })
    .filter((d): d is { x: number; y: number } => d !== null);
  const laplaceData = m.centers
    .map((x, j) => { const y = lg(m.laplace_pdf[j]); return y !== null ? { x, y } : null; })
    .filter((d): d is { x: number; y: number } => d !== null);

  return (
    <Section title="Statistics — Return Distribution">
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-2">
        {[
          { label: "N", value: stats.n.toLocaleString() },
          { label: "Mean", value: stats.mean.toExponential(2) },
          { label: "Std", value: stats.std.toExponential(2) },
          { label: "Skewness", value: stats.skewness.toFixed(3) },
          { label: "Kurtosis", value: stats.kurtosis.toFixed(2) },
          { label: "Hurst", value: isNaN(stats.hurst) ? "—" : stats.hurst.toFixed(3) },
        ].map(({ label, value }) => (
          <div key={label} className="rounded border border-gray-800 bg-gray-950 p-2">
            <p className="text-xs text-gray-500 uppercase">{label}</p>
            <p className="mt-0.5 text-sm font-bold text-white font-mono">{value}</p>
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-600 italic">Normal: skew=0, kurtosis=3, Hurst≈0.5</p>
      <p className="text-xs text-gray-600">Dots: empirical · dashed: Normal · dotted: Laplace (log scale)</p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="x" name="log return" tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "log return", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="Density" tick={{ fontSize: 10, fill: "#6b7280" }}
            tickFormatter={(v: number) => `10^${v.toFixed(0)}`} width={56}
            label={{ value: "log Density", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => Math.pow(10, v).toExponential(3)} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          <Scatter name="Empirical" data={histData} fill={COLOR} shape={SmallDot} />
          <Scatter name="Normal" data={normalData} fill={COLOR}
            line={{ stroke: COLOR, strokeWidth: 1.2, strokeDasharray: "5 3" }} shape={NoShape} legendType="none" />
          <Scatter name="Laplace" data={laplaceData} fill={COLOR}
            line={{ stroke: COLOR, strokeWidth: 1.2, strokeDasharray: "2 2" }} shape={NoShape} legendType="none" />
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

function CcdfSection({ m }: { m: FullMetrics }) {
  const data = m.x
    .map((x, j) => {
      const y = m.y[j];
      return (y !== null && y > 0 && x > 0) ? { x: Math.log10(x), y: Math.log10(y) } : null;
    })
    .filter((d): d is { x: number; y: number } => d !== null);
  return (
    <Section title="Fat Tail — CCDF (log-log)">
      <p className="text-xs text-gray-600">P(|r| &gt; x) vs |r|. Heavy tails deviate above the Normal reference.</p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="x" name="|log return|" tick={{ fontSize: 10, fill: "#6b7280" }}
            tickFormatter={(v: number) => `10^${v.toFixed(1)}`}
            label={{ value: "|log return|", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="P(|r|>x)" tick={{ fontSize: 10, fill: "#6b7280" }}
            tickFormatter={(v: number) => `10^${v.toFixed(0)}`} width={56}
            label={{ value: "P(|r|>x)", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => Math.pow(10, v).toExponential(3)} />
          <Scatter name="CCDF" data={data} fill={COLOR} line={{ stroke: COLOR, strokeWidth: 1.5 }} shape={SmallDot} />
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

function AcfSection({ m }: { m: FullMetrics }) {
  const rData = m.returns.map((v, lag) => ({ lag, acf: v }));
  const absData = m.abs_returns.map((v, lag) => ({ lag, acf: v }));
  return (
    <Section title="Autocorrelation — returns r (dashed) and |r| (solid)">
      <p className="text-xs text-gray-600">Significant positive ACF of |r| at many lags signals volatility clustering.</p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="lag" name="Lag" tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "Lag", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="acf" name="ACF" tick={{ fontSize: 10, fill: "#6b7280" }} width={44} />
          <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(4)} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          <Scatter name="ACF(r)" data={rData} fill={COLOR}
            line={{ stroke: COLOR, strokeWidth: 1.5, strokeDasharray: "5 3" }} shape={NoShape} />
          <Scatter name="ACF(|r|)" data={absData} fill={COLOR}
            line={{ stroke: COLOR, strokeWidth: 1.5 }} shape={NoShape} />
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

function DiffusionSection({ m }: { m: FullMetrics }) {
  const data = m.lags
    .map((lag, j) => {
      const v = m.vars[j];
      return (lag > 0 && v > 0) ? { lag: Math.log10(lag), v: Math.log10(v) } : null;
    })
    .filter((d): d is { lag: number; v: number } => d !== null);
  return (
    <Section title="Diffusion Scaling — Var(lag) / Var(1) (log-log)">
      <p className="text-xs text-gray-600">Slope ≈ 1 indicates random-walk diffusion; slope &gt; 1 suggests super-diffusion.</p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="lag" name="Lag" tick={{ fontSize: 10, fill: "#6b7280" }}
            tickFormatter={(v: number) => `10^${v.toFixed(1)}`}
            label={{ value: "Lag", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="v" name="Var ratio" tick={{ fontSize: 10, fill: "#6b7280" }} width={52}
            tickFormatter={(v: number) => `10^${v.toFixed(1)}`} />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => Math.pow(10, v).toFixed(4)} />
          <Scatter name="Diffusion" data={data} fill={COLOR}
            line={{ stroke: COLOR, strokeWidth: 1.5 }} shape={NoShape} />
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

function VolClusteringSection({ m }: { m: FullMetrics }) {
  const data = m.values.map((v, lag) => ({ lag, acf: v }));
  return (
    <Section title="Volatility Clustering — ACF of |r| (lags 0–100)">
      <p className="text-xs text-gray-600">Real markets show slowly-decaying positive autocorrelation in absolute returns.</p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="lag" name="Lag" tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "Lag", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="acf" name="ACF" tick={{ fontSize: 10, fill: "#6b7280" }} width={44} />
          <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(4)} />
          <Scatter name="ACF(|r|)" data={data} fill={COLOR}
            line={{ stroke: COLOR, strokeWidth: 1.5 }} shape={NoShape} />
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

function QqSection({ m }: { m: FullMetrics }) {
  const tVals = m.points.map((p) => p.t);
  const mn = Math.min(...tVals);
  const mx = Math.max(...tVals);
  const { slope, intercept } = m.line;
  const fitLine = [
    { t: mn, s: slope * mn + intercept },
    { t: mx, s: slope * mx + intercept },
  ];
  return (
    <Section title="QQ Plot vs Normal Distribution">
      <p className="text-xs text-gray-600">Points along the diagonal indicate normality; heavy tails bow outward.</p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="t" name="Theoretical" tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "Theoretical quantile", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="s" name="Sample" tick={{ fontSize: 10, fill: "#6b7280" }} width={56}
            tickFormatter={(v: number) => v.toExponential(1)}
            label={{ value: "Sample quantile", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toExponential(4)} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          <Scatter name="Normal fit" data={fitLine} fill="#4b5563"
            line={{ stroke: "#4b5563", strokeDasharray: "4 2" }} shape={NoShape} legendType="none" />
          <Scatter name="Sample" data={m.points} fill={COLOR}
            line={{ stroke: COLOR, strokeWidth: 1.5 }} shape={NoShape} />
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ── Exogenous charts ──────────────────────────────────────────────────────────

// ── Seasonality shared helpers ────────────────────────────────────────────────

function SeasonSubChart({
  title,
  data,
  xKey,
  yKey,
  xLabel,
  yLabel,
  xTickFormatter,
  yTickFormatter,
  referenceXLines,
  xLabelFormatter,
  series,
  height = 220,
  yZeroRef = false,
}: {
  title: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data?: any;
  xKey?: string;
  yKey?: string;
  xLabel?: string;
  yLabel?: string;
  xTickFormatter?: (v: number) => string;
  yTickFormatter?: (v: number) => string;
  referenceXLines?: number[];
  xLabelFormatter?: (v: number) => string;
  series?: { name: string; color: string; pts: { x: number; y: number }[] }[];
  height?: number;
  yZeroRef?: boolean;
}) {
  return (
    <div>
      <p className="text-xs text-gray-500 mb-1">{title}</p>
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 4, right: 8, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey={xKey ?? "x"} tick={{ fontSize: 10, fill: "#6b7280" }}
            tickFormatter={xTickFormatter}
            label={xLabel ? { value: xLabel, position: "insideBottom", offset: -14, fill: "#6b7280", fontSize: 11 } : undefined} />
          <YAxis type="number" dataKey={yKey ?? "y"} tick={{ fontSize: 10, fill: "#6b7280" }} width={58}
            tickFormatter={yTickFormatter ?? ((v: number) => v.toExponential(1))}
            label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 } : undefined} />
          {yZeroRef && <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />}
          {referenceXLines?.map((x) => <ReferenceLine key={x} x={x} stroke="#374151" strokeDasharray="2 2" />)}
          <Tooltip {...CHART_STYLE}
            formatter={(v: number) => v.toExponential(4)}
            labelFormatter={xLabelFormatter ? (v: number) => xLabelFormatter(v) : undefined} />
          <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
          {(series ?? []).map((s) => (
            <Scatter key={s.name} name={s.name} data={s.pts}
              fill={s.color} line={{ stroke: s.color, strokeWidth: 1.5 }} shape={NoShape} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function hasVol(arr: (number | null)[]): boolean {
  return arr.some((v) => v !== null);
}

// ── Unified SeasonalityPanel ──────────────────────────────────────────────────

function SeasonalityPanel({ m }: { m: FullMetrics }) {
  const [view, setView] = useState<SeasonalityView>("hour_week");
  const intrad = m.intraday;
  const seas = m.seasonality;

  return (
    <Section title="Seasonality">
      {/* View tabs */}
      <div className="flex gap-1 flex-wrap">
        {SEASON_VIEWS.map(({ id, label }) => (
          <button key={id} onClick={() => setView(id)}
            className={`text-xs px-3 py-1.5 rounded transition-colors ${
              view === id ? "bg-brand-500 text-white" : "bg-gray-800 text-gray-400 hover:text-white"
            }`}>
            {label}
          </button>
        ))}
      </div>
      <p className="text-xs text-gray-600">{SEASON_VIEWS.find((v) => v.id === view)?.desc}</p>

      {/* ── Hour / Day ────────────────────────────────────────────────────── */}
      {view === "hour_day" && (
        intrad ? (() => {
          // handle both old field names (mean/std) and new (return_mean/return_std)
          const rmean = intrad.return_mean ?? intrad.mean ?? [];
          const rstd  = intrad.return_std  ?? intrad.std  ?? [];
          return (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-1">
            <SeasonSubChart
              title="Return mean by hour"
              series={[{ name: "return mean", color: COLOR,
                pts: intrad.hours.map((h, j) => ({ x: h, y: rmean[j] })).filter((p) => p.y !== undefined) }]}
              xKey="x" yKey="y" xLabel="Hour (UTC)"
              xTickFormatter={(v) => String(v)} yZeroRef
            />
            <SeasonSubChart
              title="Return std by hour"
              series={[{ name: "return std", color: "#f59e0b",
                pts: intrad.hours.map((h, j) => ({ x: h, y: rstd[j] })).filter((p) => p.y !== undefined) }]}
              xKey="x" yKey="y" xLabel="Hour (UTC)"
              xTickFormatter={(v) => String(v)}
            />
            {intrad.volume_mean && hasVol(intrad.volume_mean) && (
              <SeasonSubChart
                title="Volume mean by hour"
                series={[{ name: "volume mean", color: "#34d399",
                  pts: intrad.hours.map((h, j) => ({ x: h, y: intrad.volume_mean![j] ?? NaN }))
                    .filter((p) => !isNaN(p.y)) }]}
                xKey="x" yKey="y" xLabel="Hour (UTC)"
                xTickFormatter={(v) => String(v)}
                yTickFormatter={(v) => v.toFixed(1)}
              />
            )}
          </div>
        );})() : <p className="text-xs text-gray-500">No intraday data.</p>
      )}

      {/* ── Hour / Week ───────────────────────────────────────────────────── */}
      {view === "hour_week" && (
        seas ? (
          <div className="space-y-4 pt-1">
            {/* Return chart */}
            <SeasonSubChart
              title="Return mean — intraday across the week"
              series={seas.weekly.days.map((day, d) => ({
                name: day.label,
                color: DAY_COLORS[d % DAY_COLORS.length],
                pts: day.slots
                  .map((x, j) => { const y = day.return_mean[j]; return y !== null ? { x, y } : null; })
                  .filter((p): p is { x: number; y: number } => p !== null),
              }))}
              xKey="x" yKey="y" xLabel="Day of week (time →)"
              xTickFormatter={(v) => {
                const i = seas.weekly.day_boundaries.indexOf(v);
                return i >= 0 ? seas.weekly.days[i]?.label ?? "" : "";
              }}
              xLabelFormatter={(slot) => {
                const di = Math.floor(slot / seas.weekly.counts_per_day);
                return `${seas.weekly.days[di]?.label ?? ""} ${seas.weekly.time_labels[slot % seas.weekly.counts_per_day] ?? ""}`;
              }}
              referenceXLines={seas.weekly.day_boundaries.slice(1, 7)}
              yZeroRef height={250}
            />
            {/* Volume chart */}
            {seas.weekly.days.some((d) => hasVol(d.volume)) && (
              <SeasonSubChart
                title="Volume mean — intraday across the week"
                series={seas.weekly.days.map((day, d) => ({
                  name: day.label,
                  color: DAY_COLORS[d % DAY_COLORS.length],
                  pts: day.slots
                    .map((x, j) => { const y = day.volume[j]; return y !== null ? { x, y } : null; })
                    .filter((p): p is { x: number; y: number } => p !== null),
                }))}
                xKey="x" yKey="y" xLabel="Day of week (time →)"
                xTickFormatter={(v) => {
                  const i = seas.weekly.day_boundaries.indexOf(v);
                  return i >= 0 ? seas.weekly.days[i]?.label ?? "" : "";
                }}
                xLabelFormatter={(slot) => {
                  const di = Math.floor(slot / seas.weekly.counts_per_day);
                  return `${seas.weekly.days[di]?.label ?? ""} ${seas.weekly.time_labels[slot % seas.weekly.counts_per_day] ?? ""}`;
                }}
                referenceXLines={seas.weekly.day_boundaries.slice(1, 7)}
                yTickFormatter={(v) => v.toFixed(1)} height={250}
              />
            )}
          </div>
        ) : <p className="text-xs text-gray-500">No seasonality data.</p>
      )}

      {/* ── Weekday / Month ───────────────────────────────────────────────── */}
      {view === "weekday_month" && (
        seas ? (
          <div className="space-y-4 pt-1">
            <SeasonSubChart
              title="Return mean by day-of-week, per week of month"
              series={seas.monthly.weeks.map((wk, i) => ({
                name: wk.label,
                color: SEASON_COLORS[i % SEASON_COLORS.length],
                pts: wk.days
                  .map((x, j) => { const y = wk.return_mean[j]; return y !== null ? { x, y } : null; })
                  .filter((p): p is { x: number; y: number } => p !== null),
              }))}
              xKey="x" yKey="y" xLabel="Day of week"
              xTickFormatter={(v) => seas.monthly.day_labels[v] ?? ""}
              xLabelFormatter={(v) => seas.monthly.day_labels[v] ?? ""}
              yZeroRef height={240}
            />
            {seas.monthly.weeks.some((wk) => hasVol(wk.volume)) && (
              <SeasonSubChart
                title="Volume mean by day-of-week, per week of month"
                series={seas.monthly.weeks.map((wk, i) => ({
                  name: wk.label,
                  color: SEASON_COLORS[i % SEASON_COLORS.length],
                  pts: wk.days
                    .map((x, j) => { const y = wk.volume[j]; return y !== null ? { x, y } : null; })
                    .filter((p): p is { x: number; y: number } => p !== null),
                }))}
                xKey="x" yKey="y" xLabel="Day of week"
                xTickFormatter={(v) => seas.monthly.day_labels[v] ?? ""}
                xLabelFormatter={(v) => seas.monthly.day_labels[v] ?? ""}
                yTickFormatter={(v) => v.toFixed(1)} height={240}
              />
            )}
          </div>
        ) : <p className="text-xs text-gray-500">No seasonality data.</p>
      )}

      {/* ── Day / Month ───────────────────────────────────────────────────── */}
      {view === "day_month" && (
        seas ? (
          <div className="space-y-4 pt-1">
            <SeasonSubChart
              title="Return mean by day of month, per year"
              series={seas.day_of_month.series.map((s, i) => ({
                name: s.label,
                color: SEASON_COLORS[i % SEASON_COLORS.length],
                pts: s.days
                  .map((x, j) => { const y = s.return_mean[j]; return y !== null ? { x, y } : null; })
                  .filter((p): p is { x: number; y: number } => p !== null),
              }))}
              xKey="x" yKey="y" xLabel="Day of month"
              xTickFormatter={(v) => String(v)} yZeroRef height={240}
            />
            {seas.day_of_month.series.some((s) => hasVol(s.volume)) && (
              <SeasonSubChart
                title="Volume mean by day of month, per year"
                series={seas.day_of_month.series.map((s, i) => ({
                  name: s.label,
                  color: SEASON_COLORS[i % SEASON_COLORS.length],
                  pts: s.days
                    .map((x, j) => { const y = s.volume[j]; return y !== null ? { x, y } : null; })
                    .filter((p): p is { x: number; y: number } => p !== null),
                }))}
                xKey="x" yKey="y" xLabel="Day of month"
                xTickFormatter={(v) => String(v)}
                yTickFormatter={(v) => v.toFixed(1)} height={240}
              />
            )}
          </div>
        ) : <p className="text-xs text-gray-500">No seasonality data.</p>
      )}

      {/* ── Month / Year ──────────────────────────────────────────────────── */}
      {view === "month_year" && (
        seas ? (
          <div className="space-y-4 pt-1">
            <SeasonSubChart
              title="Return mean by month, per year"
              series={seas.yearly.series.map((s, i) => ({
                name: s.label,
                color: SEASON_COLORS[i % SEASON_COLORS.length],
                pts: s.months
                  .map((x, j) => { const y = s.return_mean[j]; return y !== null ? { x, y } : null; })
                  .filter((p): p is { x: number; y: number } => p !== null),
              }))}
              xKey="x" yKey="y" xLabel="Month"
              xTickFormatter={(v) => seas.yearly.month_labels[v - 1] ?? ""}
              xLabelFormatter={(v) => seas.yearly.month_labels[v - 1] ?? ""}
              yZeroRef height={240}
            />
            {seas.yearly.series.some((s) => hasVol(s.volume)) && (
              <SeasonSubChart
                title="Volume mean by month, per year"
                series={seas.yearly.series.map((s, i) => ({
                  name: s.label,
                  color: SEASON_COLORS[i % SEASON_COLORS.length],
                  pts: s.months
                    .map((x, j) => { const y = s.volume[j]; return y !== null ? { x, y } : null; })
                    .filter((p): p is { x: number; y: number } => p !== null),
                }))}
                xKey="x" yKey="y" xLabel="Month"
                xTickFormatter={(v) => seas.yearly.month_labels[v - 1] ?? ""}
                xLabelFormatter={(v) => seas.yearly.month_labels[v - 1] ?? ""}
                yTickFormatter={(v) => v.toFixed(1)} height={240}
              />
            )}
          </div>
        ) : <p className="text-xs text-gray-500">No seasonality data.</p>
      )}
    </Section>
  );
}

function JumpTailSection({ m }: { m: FullMetrics }) {
  if (!m) return null;
  const jt = m;
  return (
    <Section title="Jump / Tail Statistics">
      <p className="text-xs text-gray-600">Jump rate = fraction of returns exceeding ±3σ. Quantiles show tail heaviness.</p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-gray-800">
              {["Jump rate (>|3σ|)", "3σ threshold", "q0.1%", "q1%", "q99%", "q99.9%"].map((h) => (
                <th key={h} className="py-2 px-3 text-right text-gray-500 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="text-right py-2 px-3 text-gray-300 font-mono">{(jt.jump_rate * 100).toFixed(3)}%</td>
              <td className="text-right py-2 px-3 text-gray-300 font-mono">{jt.threshold_3sigma.toExponential(3)}</td>
              <td className="text-right py-2 px-3 text-gray-300 font-mono">{jt.q001.toExponential(3)}</td>
              <td className="text-right py-2 px-3 text-gray-300 font-mono">{jt.q01.toExponential(3)}</td>
              <td className="text-right py-2 px-3 text-gray-300 font-mono">{jt.q99.toExponential(3)}</td>
              <td className="text-right py-2 px-3 text-gray-300 font-mono">{jt.q999.toExponential(3)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function CdfSection({ m }: { m: FullMetrics }) {
  if (!m) return null;
  const data = m.x.map((x, j) => ({ x, y: m.y[j] }));
  return (
    <Section title="CDF — Cumulative Distribution of Returns">
      <p className="text-xs text-gray-600">Empirical CDF of log-returns. S-curve shape; fat tails bow wider than Normal.</p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="x" name="log return" tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "log return", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="CDF" domain={[0, 1]} tick={{ fontSize: 10, fill: "#6b7280" }} width={44}
            label={{ value: "CDF", angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 }} />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(4)} />
          <Scatter name="CDF" data={data} fill={COLOR}
            line={{ stroke: COLOR, strokeWidth: 1.5 }} shape={NoShape} />
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

function RollingMeanSection({ m }: { m: FullMetrics }) {
  if (!m) return null;
  const rm = m;
  const data = rm.index.map((x, j) => ({ x, y: rm.values[j] }));
  return (
    <Section title="Drift / Rolling Mean of Returns">
      <p className="text-xs text-gray-600">
        Rolling mean of log-returns (window ≈ 5% of series). Should hover near zero for a drift-free market.
      </p>
      <ResponsiveContainer width="100%" height={270}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="x" name="Position" tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "Position (candle)", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="y" name="Rolling mean" tick={{ fontSize: 10, fill: "#6b7280" }} width={56}
            tickFormatter={(v: number) => v.toExponential(1)} />
          <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toExponential(4)} />
          <Scatter name={`Rolling mean (w=${rm.window})`} data={data} fill={COLOR}
            line={{ stroke: COLOR, strokeWidth: 1.2 }} shape={NoShape} />
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

function LongLagAcfSection({ m }: { m: FullMetrics }) {
  if (!m) return null;
  const la = m;
  const data = la.lags.map((lag, j) => ({ lag, acf: la.values[j] }));
  return (
    <Section title="Long-lag ACF of Returns">
      <p className="text-xs text-gray-600">ACF of log-returns up to lag 200. Values near zero indicate no long-range linear autocorrelation.</p>
      <div className="overflow-x-auto mb-3">
        <table className="text-xs border-collapse">
          <thead>
            <tr className="border-b border-gray-800">
              {[10, 20, 30, 40, 50].map((lg) => (
                <th key={lg} className="py-1 px-3 text-right text-gray-500">lag {lg}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              {[10, 20, 30, 40, 50].map((lg) => (
                <td key={lg} className="py-1 px-3 text-right text-gray-300 font-mono">
                  {la.highlights[String(lg)] != null ? la.highlights[String(lg)].toFixed(4) : "—"}
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis type="number" dataKey="lag" name="Lag" tick={{ fontSize: 10, fill: "#6b7280" }}
            label={{ value: "Lag", position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 }} />
          <YAxis type="number" dataKey="acf" name="ACF" tick={{ fontSize: 10, fill: "#6b7280" }} width={44} />
          <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />
          <Tooltip {...CHART_STYLE} formatter={(v: number) => v.toFixed(4)} />
          <Scatter name="Long-lag ACF" data={data} fill={COLOR}
            line={{ stroke: COLOR, strokeWidth: 1.2 }} shape={NoShape} />
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ── CharacteristicsPanel ──────────────────────────────────────────────────────

function SkeletonSection({ title }: { title: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-3 animate-pulse">
      <h3 className="text-xs font-semibold text-gray-600 uppercase tracking-wider">{title}</h3>
      <div className="h-[270px] bg-gray-800/50 rounded" />
    </div>
  );
}

function CharacteristicsPanel({ metrics }: { metrics: Record<string, any> }) {
  const [charsTab, setCharsTab] = useState<CharsTab>("endogenous");
  const m = metrics;

  const isComputing = ALL_CHAR_KEYS.some((k) => !(k in m));
  const doneCount = ALL_CHAR_KEYS.filter((k) => k in m).length;

  const exoAllDone = EXOGENOUS_KEYS.every((k) => k in m);
  const exoHasData = EXOGENOUS_KEYS.some((k) => k in m && !m[k]?.error);
  const hasExogenous = exoHasData || !exoAllDone;

  return (
    <div className="space-y-4">
      {/* Progress indicator while computing */}
      {isComputing && (
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <svg className="animate-spin h-3 w-3 text-brand-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Computing… {doneCount}/{ALL_CHAR_KEYS.length} analyses done
        </div>
      )}

      {/* Sub-tabs */}
      <div className="flex gap-1 border-b border-gray-800">
        {(["endogenous", "exogenous"] as CharsTab[]).map((t) => (
          <button key={t} onClick={() => setCharsTab(t)}
            className={`px-4 py-2 text-sm border-b-2 -mb-px transition-colors capitalize ${
              charsTab === t ? "border-brand-500 text-brand-500" : "border-transparent text-gray-400 hover:text-white"
            }`}>
            {t}
            {t === "exogenous" && !hasExogenous && <span className="ml-1 text-xs text-gray-600">(n/a)</span>}
          </button>
        ))}
      </div>

      {charsTab === "endogenous" && (
        <div className="space-y-4">
          {m.return_dist && !m.return_dist.error
            ? <ReturnDistSection m={m.return_dist} />
            : <SkeletonSection title="Statistics — Return Distribution" />}
          {m.ccdf && !m.ccdf.error
            ? <CcdfSection m={m.ccdf} />
            : <SkeletonSection title="Fat Tail — CCDF (log-log)" />}
          {m.acf && !m.acf.error
            ? <AcfSection m={m.acf} />
            : <SkeletonSection title="Autocorrelation — returns r and |r|" />}
          {m.diffusion && !m.diffusion.error
            ? <DiffusionSection m={m.diffusion} />
            : <SkeletonSection title="Diffusion Scaling" />}
          {m.vol_clustering && !m.vol_clustering.error
            ? <VolClusteringSection m={m.vol_clustering} />
            : <SkeletonSection title="Volatility Clustering" />}
          {m.qq && !m.qq.error
            ? <QqSection m={m.qq} />
            : <SkeletonSection title="QQ Plot vs Normal Distribution" />}
        </div>
      )}

      {charsTab === "exogenous" && (
        <div className="space-y-4">
          {exoAllDone && !exoHasData ? (
            <p className="text-sm text-gray-500">No exogenous data available for this dataset.</p>
          ) : (
            <>
              {m.exogenous_seasonality && !m.exogenous_seasonality.error
                ? <SeasonalityPanel m={m.exogenous_seasonality} />
                : (!exoAllDone && <SkeletonSection title="Seasonality" />)}
              {m.exogenous_jump_tail && !m.exogenous_jump_tail.error
                ? <JumpTailSection m={m.exogenous_jump_tail} />
                : (!exoAllDone && <SkeletonSection title="Jump / Tail Statistics" />)}
              {m.exogenous_cdf && !m.exogenous_cdf.error
                ? <CdfSection m={m.exogenous_cdf} />
                : (!exoAllDone && <SkeletonSection title="CDF — Cumulative Distribution" />)}
              {m.exogenous_rolling_mean && !m.exogenous_rolling_mean.error
                ? <RollingMeanSection m={m.exogenous_rolling_mean} />
                : (!exoAllDone && <SkeletonSection title="Drift / Rolling Mean" />)}
              {m.exogenous_long_lag_acf && !m.exogenous_long_lag_acf.error
                ? <LongLagAcfSection m={m.exogenous_long_lag_acf} />
                : (!exoAllDone && <SkeletonSection title="Long-lag ACF" />)}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { toast } = useToast();
  const [tab, setTab] = useState<Tab>("overview");
  const [deleting, setDeleting] = useState(false);
  const [renamingName, setRenamingName] = useState<string | null>(null);
  const [showComparePicker, setShowComparePicker] = useState(false);
  const [compareSearch, setCompareSearch] = useState("");
  const { data: allDatasets } = useSWR(
    showComparePicker ? "/api/v1/datasets?page_size=200" : null,
    fetcher,
  );
  const [computingChars, setComputingChars] = useState(false);
  const [computingElapsed, setComputingElapsed] = useState(0);
  const computingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Timestamp of the row that existed when recompute was triggered; used to
  // distinguish the new row from the stale one still in the DB during the delay
  // between enqueue and worker creating the empty row.
  const computeTriggeredAtRef = useRef<string | null>(null);
  const isRunning = (d: any) => d?.status === "running";
  const { data: dataset } = useSWR(`/api/v1/datasets/${id}`, fetcher, {
    refreshInterval: (data) => isRunning(data) ? 3000 : 0,
  });
  const { data: datasource } = useSWR(
    dataset?.datasource_id ? `/api/v1/datasources/${dataset.datasource_id}` : null,
    fetcher,
  );

  const isWebReport = datasource?.type === "web_report";

  const { data: chars } = useSWR(
    !isWebReport && tab === "characteristics" ? `/api/v1/datasets/${id}/characteristics` : null,
    fetcher,
    { refreshInterval: (data) => {
      if (computingChars) return 2000;
      if (data && ALL_CHAR_KEYS.some((k) => !(k in (data.metrics ?? {})))) return 2000;
      return 0;
    }},
  );
  const [previewTimeframe, setPreviewTimeframe] = useState<string>("");
  const tfParam = previewTimeframe ? `&timeframe=${previewTimeframe}` : "";
  const canPreview = dataset?.status === "ready" || dataset?.status === "running";
  const { data: preview } = useSWR(
    !isWebReport && tab === "preview" && canPreview ? `/api/v1/datasets/${id}/preview?rows=200${tfParam}` : null,
    fetcher,
    { refreshInterval: tab === "preview" && isRunning(dataset) ? 5000 : 0 },
  );
  const { data: liveProgress } = useSWR(
    isRunning(dataset) ? `/api/v1/datasets/${id}/live-progress` : null,
    fetcher,
    { refreshInterval: 3000 },
  );
  const [fileShowAll, setFileShowAll] = useState(false);
  const { data: webFiles, mutate: mutateFiles } = useSWR(
    isWebReport && dataset?.datasource_id
      ? `/api/v1/datasources/${dataset.datasource_id}/web-report/files`
      : null,
    fetcher,
  );

  // Stop polling and timer only when a row *newer* than the one that existed
  // when recompute was triggered has all its keys. This prevents the stale row
  // (which already has all keys) from immediately cancelling the poll.
  const isNewerRow = !computeTriggeredAtRef.current ||
    (chars?.computed_at && new Date(chars.computed_at) > new Date(computeTriggeredAtRef.current));
  const charsComplete = chars && isNewerRow && ALL_CHAR_KEYS.every((k) => k in (chars.metrics ?? {}));
  if (computingChars && charsComplete) {
    setComputingChars(false);
    computeTriggeredAtRef.current = null;
    if (computingTimerRef.current) {
      clearInterval(computingTimerRef.current);
      computingTimerRef.current = null;
    }
  }

  // Cleanup timer on unmount
  useEffect(() => () => { if (computingTimerRef.current) clearInterval(computingTimerRef.current); }, []);

  async function computeChars() {
    // Record the current row's timestamp so we can ignore it when checking completion.
    computeTriggeredAtRef.current = chars?.computed_at ?? new Date().toISOString();
    setComputingChars(true);
    setComputingElapsed(0);
    if (computingTimerRef.current) clearInterval(computingTimerRef.current);
    computingTimerRef.current = setInterval(() => setComputingElapsed((s) => s + 1), 1000);
    await fetch(`/api/v1/datasets/${id}/characteristics/compute`, { method: "POST" });
  }

  async function renameDataset() {
    if (renamingName === null || !renamingName.trim()) return;
    const res = await fetch(`/api/v1/datasets/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: renamingName.trim() }),
    });
    if (res.ok) {
      toast("Dataset renamed", "success");
      mutate(`/api/v1/datasets/${id}`);
      mutate("/api/v1/datasets");
      setRenamingName(null);
    } else {
      toast("Failed to rename", "error");
    }
  }

  async function deleteDataset() {
    if (!confirm(`Delete dataset "${dataset?.name}"? This cannot be undone.`)) return;
    setDeleting(true);
    const res = await fetch(`/api/v1/datasets/${id}`, { method: "DELETE" });
    if (res.ok) {
      window.location.href = "/data";
    } else {
      const body = await res.json().catch(() => ({}));
      alert(body.error?.message ?? "Failed to delete");
      setDeleting(false);
    }
  }

  return (
    <div className="space-y-4 max-w-5xl">
      <div className="flex items-start justify-between">
        <div>
          <a href="/data" className="text-xs text-gray-500 hover:text-white">← Data</a>
          {renamingName !== null ? (
            <div className="mt-1 flex items-center gap-2">
              <input
                value={renamingName}
                onChange={(e) => setRenamingName(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") renameDataset(); if (e.key === "Escape") setRenamingName(null); }}
                autoFocus
                className="rounded border border-gray-700 bg-gray-900 px-3 py-1 text-xl font-semibold text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              />
              <button onClick={renameDataset} className="text-xs text-brand-400 hover:underline">Save</button>
              <button onClick={() => setRenamingName(null)} className="text-xs text-gray-500 hover:text-white">Cancel</button>
            </div>
          ) : (
            <div className="mt-1 flex items-center gap-2">
              <h1 className="text-2xl font-semibold text-white">{dataset?.name ?? "…"}</h1>
              {dataset && (
                <button onClick={() => setRenamingName(dataset.name)} className="text-xs text-gray-500 hover:text-gray-300">
                  rename
                </button>
              )}
            </div>
          )}
          <div className="flex gap-3 mt-1 text-xs text-gray-400">
            {dataset?.symbol && <span>{dataset.symbol}</span>}
            {dataset?.timeframe && <span>{dataset.timeframe}</span>}
            {dataset && <StatusBadge status={dataset.status} />}
            {dataset?.row_count && <span>{dataset.row_count.toLocaleString()} rows</span>}
          </div>
        </div>
        {dataset && (
          <div className="flex gap-2">
            {!isWebReport && dataset.status === "ready" && (
              <a
                href={`/api/v1/datasets/${id}/download${previewTimeframe ? `?timeframe=${previewTimeframe}` : ""}`}
                className="rounded border border-gray-600 px-3 py-1.5 text-xs text-gray-300 hover:border-gray-400 hover:text-white"
              >
                Download CSV
              </a>
            )}
            {isWebReport && dataset.datasource_id && (
              <a
                href={`/data/datasources/${dataset.datasource_id}`}
                className="rounded border border-gray-600 px-3 py-1.5 text-xs text-gray-300 hover:border-gray-400 hover:text-white"
              >
                ← Datasource
              </a>
            )}
            <button
              onClick={deleteDataset}
              disabled={deleting}
              className="rounded border border-red-800 px-3 py-1.5 text-xs text-red-400 hover:bg-red-900/20 disabled:opacity-50"
            >
              {deleting ? "Deleting…" : "Delete"}
            </button>
          </div>
        )}
      </div>

      {/* ── Web report dataset: file browser ── */}
      {isWebReport && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-3">
            <InfoCard label="Collected" value={dataset?.from_ts ? new Date(dataset.from_ts).toLocaleString() : "—"} />
            <InfoCard label="Files this run" value={dataset?.row_count?.toLocaleString() ?? "—"} />
            <InfoCard label="Subfolder" value={dataset?.artifact_path ?? "—"} mono />
          </div>

          <div className="rounded border border-gray-800 bg-gray-900 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-medium text-gray-400 uppercase tracking-wide">
                Downloaded Files
                {webFiles && <span className="ml-2 text-gray-600 normal-case font-normal">{webFiles.length} total</span>}
              </h2>
              <button onClick={() => mutateFiles()} className="text-xs text-gray-500 hover:text-white">Refresh</button>
            </div>

            {!webFiles && <p className="text-sm text-gray-500">Loading…</p>}
            {webFiles && webFiles.length === 0 && <p className="text-sm text-gray-500">No files downloaded yet.</p>}
            {webFiles && webFiles.length > 0 && (
              <>
                <div className="rounded border border-gray-800 overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-800 text-left">
                        <th className="px-4 py-2 text-xs text-gray-400 font-medium">Filename</th>
                        <th className="px-4 py-2 text-xs text-gray-400 font-medium">Size</th>
                        <th className="px-4 py-2 text-xs text-gray-400 font-medium">Downloaded</th>
                        <th className="px-4 py-2 text-xs text-gray-400 font-medium"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {(fileShowAll ? webFiles : webFiles.slice(0, 20)).map((f: any) => (
                        <tr key={f.path} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                          <td className="px-4 py-2 font-mono text-xs text-white">{f.name}</td>
                          <td className="px-4 py-2 text-gray-400 text-xs tabular-nums">
                            {f.size_bytes >= 1048576 ? `${(f.size_bytes / 1048576).toFixed(1)} MB`
                              : f.size_bytes >= 1024 ? `${(f.size_bytes / 1024).toFixed(1)} KB`
                              : `${f.size_bytes} B`}
                          </td>
                          <td className="px-4 py-2 text-gray-400 text-xs">{new Date(f.modified_at).toLocaleString()}</td>
                          <td className="px-4 py-2 text-right">
                            <a href={`/api/v1/datasources/${dataset?.datasource_id}/web-report/files/${f.path}`}
                              target="_blank" rel="noopener noreferrer"
                              className="text-xs text-brand-400 hover:text-brand-300 hover:underline">
                              Open
                            </a>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {webFiles.length > 20 && (
                    <div className="px-4 py-2 border-t border-gray-800">
                      <button onClick={() => setFileShowAll((v) => !v)} className="text-xs text-gray-500 hover:text-white">
                        {fileShowAll ? "Show less" : `Show all ${webFiles.length} files`}
                      </button>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── Time-series dataset: tabs ── */}
      {!isWebReport && <div className="flex gap-1 border-b border-gray-800">
        {(["overview", "preview", "characteristics"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm border-b-2 -mb-px transition-colors ${
              tab === t ? "border-brand-500 text-brand-500" : "border-transparent text-gray-400 hover:text-white"
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>}

      {/* Overview */}
      {!isWebReport && tab === "overview" && dataset && (
        <div className="grid grid-cols-2 gap-4">
          <InfoCard label="From" value={dataset.from_ts ? new Date(dataset.from_ts).toLocaleDateString() : "—"} />
          <InfoCard label="To" value={dataset.to_ts ? new Date(dataset.to_ts).toLocaleDateString() : "—"} />
          <InfoCard label="Rows" value={dataset.row_count?.toLocaleString() ?? "—"} />
          <InfoCard label="Status" value={dataset.status} />
          <InfoCard label="Artifact" value={dataset.artifact_path ?? "—"} mono />
        </div>
      )}

      {/* Preview */}
      {!isWebReport && tab === "preview" && (
        <div className="space-y-3">
          {isRunning(dataset) && (
            <div className="flex items-center gap-2 rounded border border-brand-500/30 bg-brand-500/10 px-3 py-2 text-xs text-brand-300">
              <span className="inline-block w-2 h-2 rounded-full bg-brand-400 animate-pulse" />
              Simulation running —{" "}
              {liveProgress
                ? <>{liveProgress.total_trades?.toLocaleString() ?? 0} ticks generated (batch {(liveProgress.batch_num ?? 0) + 1})</>
                : "waiting for first batch…"
              }
            </div>
          )}
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-400 uppercase">Timeframe</label>
            <select
              value={previewTimeframe || dataset?.timeframe || ""}
              onChange={(e) => setPreviewTimeframe(e.target.value)}
              className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-white focus:outline-none"
            >
              {["M1", "M5", "M15", "M30", "H1", "H4", "D1"].map((tf) => (
                <option key={tf} value={tf}>{tf}</option>
              ))}
            </select>
            <span className="text-xs text-gray-500">(applies to tick datasets)</span>
          </div>
          <div className="overflow-x-auto">
            {!preview && <p className="text-gray-400 text-sm">Loading preview…</p>}
            {preview && preview.length > 0 && (
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-400 uppercase">
                    {Object.keys(preview[0]).map((col) => (
                      <th key={col} className="py-2 pr-4 whitespace-nowrap">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.slice(0, 50).map((row: any, i: number) => (
                    <tr key={i} className="border-b border-gray-800/40">
                      {Object.values(row).map((v: any, j: number) => (
                        <td key={j} className="py-1.5 pr-4 text-gray-300 whitespace-nowrap font-mono">
                          {typeof v === "number" ? v.toFixed(5) : String(v)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}

      {/* Characteristics */}
      {!isWebReport && tab === "characteristics" && (
        <div className="space-y-4">
          {!chars && (
            <div className="flex items-center gap-3">
              {computingChars ? (
                <div className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4 text-brand-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  <p className="text-gray-400 text-sm">
                    Computing characteristics… {computingElapsed}s elapsed
                    {computingElapsed > 90 && <span className="text-yellow-500 ml-1">(large dataset — may take a few minutes)</span>}
                  </p>
                </div>
              ) : (
                <>
                  <p className="text-gray-400 text-sm">No characteristics computed yet.</p>
                  <button onClick={computeChars} className="rounded bg-brand-500 px-3 py-1.5 text-xs text-white hover:bg-sky-400">
                    Compute now
                  </button>
                </>
              )}
            </div>
          )}
          {chars && (
            <>
              <div className="flex justify-end gap-2">
                <button onClick={() => { setCompareSearch(""); setShowComparePicker(true); }}
                  className="rounded border border-gray-700 px-3 py-1.5 text-xs text-gray-300 hover:border-gray-500 hover:text-white">
                  Compare with…
                </button>
                <button onClick={computeChars} className="rounded bg-gray-700 px-3 py-1.5 text-xs text-white hover:bg-gray-600">
                  Recompute
                </button>
              </div>
              <CharacteristicsPanel metrics={chars.metrics} />
            </>
          )}

          {/* Compare picker modal */}
          {showComparePicker && (
            <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
              onClick={() => setShowComparePicker(false)}>
              <div className="bg-gray-900 border border-gray-700 rounded-lg w-96 max-h-[480px] flex flex-col"
                onClick={(e) => e.stopPropagation()}>
                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
                  <h3 className="text-sm font-medium text-white">Compare with another dataset</h3>
                  <button onClick={() => setShowComparePicker(false)}
                    className="text-gray-500 hover:text-white text-lg leading-none">×</button>
                </div>
                <div className="px-4 py-2 border-b border-gray-800">
                  <input
                    autoFocus
                    value={compareSearch}
                    onChange={(e) => setCompareSearch(e.target.value)}
                    placeholder="Search by name or symbol…"
                    className="w-full rounded border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
                  />
                </div>
                <div className="overflow-y-auto flex-1">
                  {!allDatasets && <p className="text-xs text-gray-500 px-4 py-3">Loading…</p>}
                  {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                  {(allDatasets ?? []).filter((d: any) =>
                    d.id !== Number(id) &&
                    (d.name.toLowerCase().includes(compareSearch.toLowerCase()) ||
                      (d.symbol ?? "").toLowerCase().includes(compareSearch.toLowerCase()))
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  ).map((d: any) => (
                    <button key={d.id}
                      onClick={() => { setShowComparePicker(false); router.push(`/data/compare?ids=${id},${d.id}`); }}
                      className="w-full text-left px-4 py-2.5 hover:bg-gray-800 border-b border-gray-800/40 transition-colors">
                      <p className="text-sm text-white">{d.name}</p>
                      <p className="text-xs text-gray-500">
                        {[d.symbol, d.timeframe, d.row_count ? d.row_count.toLocaleString() + " rows" : null]
                          .filter(Boolean).join(" · ")}
                      </p>
                    </button>
                  ))}
                  {allDatasets && (allDatasets as any[]).filter((d: any) => d.id !== Number(id)).length === 0 && (
                    <p className="text-xs text-gray-500 px-4 py-3">No other datasets found.</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function InfoCard({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded border border-gray-800 bg-gray-900 p-3">
      <p className="text-xs text-gray-400 uppercase">{label}</p>
      <p className={`mt-1 text-white ${mono ? "font-mono text-xs" : "text-sm font-medium"}`}>{value}</p>
    </div>
  );
}

function StatCard({ label, value, hint }: { label: string; value: string | undefined; hint?: string }) {
  return (
    <div className="rounded border border-gray-800 bg-gray-900 p-3">
      <p className="text-xs text-gray-400 uppercase">{label}</p>
      <p className="mt-1 text-lg font-bold text-white font-mono">{value ?? "—"}</p>
      {hint && <p className="text-xs text-gray-500 mt-0.5">{hint}</p>}
    </div>
  );
}
