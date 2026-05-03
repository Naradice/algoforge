"use client";

import { Suspense, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import useSWR from "swr";
import { apiFetch, fetcher } from "@/lib/fetcher";
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

// ── Constants ─────────────────────────────────────────────────────────────────

const COMPARE_COLORS = ["#60a5fa", "#34d399", "#f59e0b", "#f87171", "#a78bfa", "#fb923c"];
const CHART_STYLE = {
  contentStyle: { background: "#111827", border: "1px solid #374151", fontSize: 10 },
  labelStyle: { color: "#9ca3af" },
};
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const NoShape = (_: any): React.ReactElement => <g />;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const SmallDot = ({ cx, cy, fill }: any): React.ReactElement => (
  <circle cx={cx} cy={cy} r={2.5} fill={fill} opacity={0.7} />
);

// ── Types ─────────────────────────────────────────────────────────────────────

type CompareEntry = {
  id: number;
  name: string;
  color: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metrics: Record<string, any> | null;
};

// ── Layout helpers ────────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-5 space-y-3">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">{title}</h3>
      {children}
    </div>
  );
}

// ── Multi-series line chart ───────────────────────────────────────────────────

function MultiLine({
  series,
  height = 270,
  xLabel,
  yLabel,
  xTickFormatter,
  yTickFormatter,
  yZeroRef = false,
}: {
  series: { name: string; color: string; data: { x: number; y: number }[] }[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xTickFormatter?: (v: number) => string;
  yTickFormatter?: (v: number) => string;
  yZeroRef?: boolean;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 4, right: 16, bottom: 24, left: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis type="number" dataKey="x" tick={{ fontSize: 10, fill: "#6b7280" }}
          tickFormatter={xTickFormatter}
          label={xLabel ? { value: xLabel, position: "insideBottom", offset: -12, fill: "#6b7280", fontSize: 11 } : undefined} />
        <YAxis type="number" dataKey="y" tick={{ fontSize: 10, fill: "#6b7280" }} width={58}
          tickFormatter={yTickFormatter ?? ((v: number) => v.toExponential(1))}
          label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft", fill: "#6b7280", fontSize: 11 } : undefined} />
        {yZeroRef && <ReferenceLine y={0} stroke="#374151" strokeDasharray="2 2" />}
        <Tooltip {...CHART_STYLE} />
        <Legend verticalAlign="top" wrapperStyle={{ fontSize: 11 }} />
        {series.map((s) => (
          <Scatter key={s.name} name={s.name} data={s.data}
            fill={s.color} line={{ stroke: s.color, strokeWidth: 1.5 }} shape={NoShape} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ── Comparison chart sections ─────────────────────────────────────────────────

function StatsTable({ entries }: { entries: CompareEntry[] }) {
  const ready = entries.filter((e) => e.metrics?.return_dist && !e.metrics.return_dist.error);
  if (ready.length === 0) return null;

  const rows = [
    { label: "N", key: "n", fmt: (v: number) => v.toLocaleString() },
    { label: "Mean", key: "mean", fmt: (v: number) => v.toExponential(3) },
    { label: "Std", key: "std", fmt: (v: number) => v.toExponential(3) },
    { label: "Skewness", key: "skewness", fmt: (v: number) => v.toFixed(4) },
    { label: "Kurtosis", key: "kurtosis", fmt: (v: number) => v.toFixed(3) },
    { label: "Hurst", key: "hurst", fmt: (v: number) => isNaN(v) ? "—" : v.toFixed(4) },
  ];

  return (
    <Section title="Summary Statistics">
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="py-2 px-3 text-left text-gray-500 font-medium">Metric</th>
              {ready.map((e) => (
                <th key={e.id} className="py-2 px-3 text-right font-semibold" style={{ color: e.color }}>{e.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ label, key, fmt }) => (
              <tr key={key} className="border-b border-gray-800/50">
                <td className="py-1.5 px-3 text-gray-400">{label}</td>
                {ready.map((e) => {
                  const v = e.metrics!.return_dist.stats[key];
                  return (
                    <td key={e.id} className="py-1.5 px-3 text-right text-gray-200 font-mono">
                      {v != null ? fmt(v) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function CompareReturnDist({ entries }: { entries: CompareEntry[] }) {
  const ready = entries.filter((e) => e.metrics?.return_dist && !e.metrics.return_dist.error);
  if (ready.length === 0) return null;

  const lg = (v: number | null) => (v !== null && v > 0) ? Math.log10(v) : null;
  const series = ready.map((e) => {
    const m = e.metrics!.return_dist;
    const data = m.centers
      .map((x: number, j: number) => { const y = lg(m.hist[j]); return y !== null ? { x, y } : null; })
      .filter((d: unknown): d is { x: number; y: number } => d !== null);
    return { name: e.name, color: e.color, data };
  });

  return (
    <Section title="Return Distribution (log scale)">
      <p className="text-xs text-gray-600">Empirical log-histogram. Similar curves = similar return distribution.</p>
      <MultiLine series={series} xLabel="log return" yLabel="log Density"
        yTickFormatter={(v) => `10^${v.toFixed(0)}`} />
    </Section>
  );
}

function CompareCcdf({ entries }: { entries: CompareEntry[] }) {
  const ready = entries.filter((e) => e.metrics?.ccdf && !e.metrics.ccdf.error);
  if (ready.length === 0) return null;

  const series = ready.map((e) => {
    const m = e.metrics!.ccdf;
    const data = m.x
      .map((x: number, j: number) => {
        const y = m.y[j];
        return (y !== null && y > 0 && x > 0) ? { x: Math.log10(x), y: Math.log10(y) } : null;
      })
      .filter((d: unknown): d is { x: number; y: number } => d !== null);
    return { name: e.name, color: e.color, data };
  });

  return (
    <Section title="Fat Tail — CCDF (log-log)">
      <p className="text-xs text-gray-600">P(|r| &gt; x) vs |r|. Steeper slope = lighter tail. Similar slopes = similar tail regime.</p>
      <MultiLine series={series} xLabel="|log return|" yLabel="P(|r|>x)"
        xTickFormatter={(v) => `10^${v.toFixed(1)}`}
        yTickFormatter={(v) => `10^${v.toFixed(0)}`} />
    </Section>
  );
}

function CompareAcf({ entries }: { entries: CompareEntry[] }) {
  const ready = entries.filter((e) => e.metrics?.acf && !e.metrics.acf.error);
  if (ready.length === 0) return null;

  const series = ready.map((e) => {
    const m = e.metrics!.acf;
    return { name: e.name, color: e.color, data: m.abs_returns.map((v: number, lag: number) => ({ x: lag, y: v })) };
  });

  return (
    <Section title="Autocorrelation of |returns|">
      <p className="text-xs text-gray-600">Persistent positive values = volatility clustering. Compare decay rate and magnitude.</p>
      <MultiLine series={series} xLabel="Lag" yLabel="ACF"
        yTickFormatter={(v) => v.toFixed(3)} yZeroRef />
    </Section>
  );
}

function CompareDiffusion({ entries }: { entries: CompareEntry[] }) {
  const ready = entries.filter((e) => e.metrics?.diffusion && !e.metrics.diffusion.error);
  if (ready.length === 0) return null;

  const series = ready.map((e) => {
    const m = e.metrics!.diffusion;
    const data = m.lags
      .map((lag: number, j: number) => {
        const v = m.vars[j];
        return (lag > 0 && v > 0) ? { x: Math.log10(lag), y: Math.log10(v) } : null;
      })
      .filter((d: unknown): d is { x: number; y: number } => d !== null);
    return { name: e.name, color: e.color, data };
  });

  return (
    <Section title="Diffusion Scaling (log-log)">
      <p className="text-xs text-gray-600">Slope ≈ 1 = random walk. Matching slopes = similar diffusion regime.</p>
      <MultiLine series={series} xLabel="Lag"
        xTickFormatter={(v) => `10^${v.toFixed(1)}`}
        yTickFormatter={(v) => `10^${v.toFixed(1)}`} />
    </Section>
  );
}

function CompareVolClustering({ entries }: { entries: CompareEntry[] }) {
  const ready = entries.filter((e) => e.metrics?.vol_clustering && !e.metrics.vol_clustering.error);
  if (ready.length === 0) return null;

  const series = ready.map((e) => {
    const m = e.metrics!.vol_clustering;
    return { name: e.name, color: e.color, data: m.values.map((v: number, lag: number) => ({ x: lag, y: v })) };
  });

  return (
    <Section title="Volatility Clustering — ACF of |returns| (0–100 lags)">
      <p className="text-xs text-gray-600">Real markets show slowly-decaying positive ACF. Compare how fast the simulation's clustering decays.</p>
      <MultiLine series={series} xLabel="Lag" yLabel="ACF"
        yTickFormatter={(v) => v.toFixed(3)} yZeroRef />
    </Section>
  );
}

function CompareQq({ entries }: { entries: CompareEntry[] }) {
  const ready = entries.filter((e) => e.metrics?.qq && !e.metrics.qq.error);
  if (ready.length === 0) return null;

  return (
    <Section title="QQ Plot vs Normal">
      <p className="text-xs text-gray-600">Points bowing away from the diagonal = fat tails. Similar shapes = similar tail behaviour.</p>
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
          {ready.map((e) => (
            <Scatter key={e.id} name={e.name} data={e.metrics!.qq.points}
              fill={e.color} shape={(props: unknown) => <SmallDot {...(props as object)} fill={e.color} />} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </Section>
  );
}

function CompareJumpTail({ entries }: { entries: CompareEntry[] }) {
  const ready = entries.filter((e) => e.metrics?.exogenous_jump_tail && !e.metrics.exogenous_jump_tail.error);
  if (ready.length === 0) return null;

  const rows = [
    { label: "Jump rate (>|3σ|)", key: "jump_rate", fmt: (v: number) => (v * 100).toFixed(3) + "%" },
    { label: "3σ threshold", key: "threshold_3sigma", fmt: (v: number) => v.toExponential(3) },
    { label: "q0.1%", key: "q001", fmt: (v: number) => v.toExponential(3) },
    { label: "q1%", key: "q01", fmt: (v: number) => v.toExponential(3) },
    { label: "q99%", key: "q99", fmt: (v: number) => v.toExponential(3) },
    { label: "q99.9%", key: "q999", fmt: (v: number) => v.toExponential(3) },
  ];

  return (
    <Section title="Jump / Tail Statistics">
      <div className="overflow-x-auto">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="py-2 px-3 text-left text-gray-500 font-medium">Metric</th>
              {ready.map((e) => (
                <th key={e.id} className="py-2 px-3 text-right font-semibold" style={{ color: e.color }}>{e.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ label, key, fmt }) => (
              <tr key={key} className="border-b border-gray-800/50">
                <td className="py-1.5 px-3 text-gray-400">{label}</td>
                {ready.map((e) => {
                  const v = e.metrics!.exogenous_jump_tail[key];
                  return (
                    <td key={e.id} className="py-1.5 px-3 text-right text-gray-200 font-mono">
                      {v != null ? fmt(v) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function CompareCdf({ entries }: { entries: CompareEntry[] }) {
  const ready = entries.filter((e) => e.metrics?.exogenous_cdf && !e.metrics.exogenous_cdf.error);
  if (ready.length === 0) return null;

  const series = ready.map((e) => {
    const m = e.metrics!.exogenous_cdf;
    return { name: e.name, color: e.color, data: m.x.map((x: number, j: number) => ({ x, y: m.y[j] })) };
  });

  return (
    <Section title="CDF of Returns">
      <p className="text-xs text-gray-600">Empirical CDF overlay. Wider spread = fatter tails.</p>
      <MultiLine series={series} xLabel="log return" yLabel="CDF"
        yTickFormatter={(v) => v.toFixed(2)} />
    </Section>
  );
}

function CompareLongLagAcf({ entries }: { entries: CompareEntry[] }) {
  const ready = entries.filter((e) => e.metrics?.exogenous_long_lag_acf && !e.metrics.exogenous_long_lag_acf.error);
  if (ready.length === 0) return null;

  const series = ready.map((e) => {
    const m = e.metrics!.exogenous_long_lag_acf;
    return { name: e.name, color: e.color, data: m.lags.map((lag: number, j: number) => ({ x: lag, y: m.values[j] })) };
  });

  return (
    <Section title="Long-lag ACF of Returns (0–200 lags)">
      <p className="text-xs text-gray-600">Near-zero = no long-range autocorrelation. Compare structure at higher lags.</p>
      <MultiLine series={series} xLabel="Lag" yLabel="ACF"
        yTickFormatter={(v) => v.toFixed(3)} yZeroRef />
    </Section>
  );
}

// ── Dataset picker modal ──────────────────────────────────────────────────────

function DatasetPicker({
  currentIds,
  onAdd,
  onClose,
}: {
  currentIds: number[];
  onAdd: (id: number) => void;
  onClose: () => void;
}) {
  const [search, setSearch] = useState("");
  const { data: datasets } = useSWR("/api/v1/datasets?page_size=200", fetcher);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const filtered = (datasets ?? []).filter((d: any) =>
    !currentIds.includes(d.id) &&
    (d.name.toLowerCase().includes(search.toLowerCase()) ||
      (d.symbol ?? "").toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
      onClick={onClose}>
      <div className="bg-gray-900 border border-gray-700 rounded-lg w-96 max-h-[480px] flex flex-col"
        onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h3 className="text-sm font-medium text-white">Add dataset to compare</h3>
          <button onClick={onClose} className="text-gray-500 hover:text-white text-lg leading-none">×</button>
        </div>
        <div className="px-4 py-2 border-b border-gray-800">
          <input
            autoFocus
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by name or symbol…"
            className="w-full rounded border border-gray-700 bg-gray-800 px-3 py-1.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-brand-500"
          />
        </div>
        <div className="overflow-y-auto flex-1">
          {!datasets && <p className="text-xs text-gray-500 px-4 py-3">Loading…</p>}
          {datasets && filtered.length === 0 && (
            <p className="text-xs text-gray-500 px-4 py-3">No other datasets found.</p>
          )}
          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
          {filtered.map((d: any) => (
            <button key={d.id} onClick={() => { onAdd(d.id); onClose(); }}
              className="w-full text-left px-4 py-2.5 hover:bg-gray-800 border-b border-gray-800/40 transition-colors">
              <p className="text-sm text-white">{d.name}</p>
              <p className="text-xs text-gray-500">
                {[d.symbol, d.timeframe, d.row_count ? d.row_count.toLocaleString() + " rows" : null]
                  .filter(Boolean).join(" · ")}
              </p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

function CompareContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const ids = (searchParams.get("ids") ?? "").split(",").map(Number).filter((n) => n > 0);

  const [showPicker, setShowPicker] = useState(false);
  const [charsTab, setCharsTab] = useState<"endogenous" | "exogenous">("endogenous");

  // Fetch dataset names and characteristics for all IDs in parallel
  const { data: datasetsData } = useSWR(
    ids.length ? ["compare-datasets", ...ids] : null,
    () => Promise.all(ids.map((id) =>
      apiFetch(`/api/v1/datasets/${id}`).then((r) => r.json()).then((j) => j.data)
    ))
  );
  const { data: charsData } = useSWR(
    ids.length ? ["compare-chars", ...ids] : null,
    () => Promise.all(ids.map((id) =>
      apiFetch(`/api/v1/datasets/${id}/characteristics`).then((r) => r.json()).then((j) => j.data)
    ))
  );

  const entries: CompareEntry[] = ids.map((id, i) => ({
    id,
    name: datasetsData?.[i]?.name ?? `Dataset ${id}`,
    color: COMPARE_COLORS[i % COMPARE_COLORS.length],
    metrics: charsData?.[i]?.metrics ?? null,
  }));

  function addDataset(newId: number) {
    router.push(`/data/compare?ids=${[...ids, newId].join(",")}`);
  }

  function removeDataset(removeId: number) {
    const remaining = ids.filter((id) => id !== removeId);
    if (remaining.length > 0) router.push(`/data/compare?ids=${remaining.join(",")}`);
    else router.push("/data");
  }

  if (ids.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        No datasets selected.{" "}
        <a href="/data" className="text-brand-400 hover:underline">Go back to data</a>
      </div>
    );
  }

  const noChars = charsData && entries.filter((e) => e.metrics === null);

  return (
    <div className="space-y-5 max-w-5xl">
      {/* Header */}
      <div>
        <a href="/data" className="text-xs text-gray-500 hover:text-white">← Data</a>
        <h1 className="mt-1 text-2xl font-semibold text-white">Compare Characteristics</h1>
      </div>

      {/* Dataset chips + add button */}
      <div className="flex flex-wrap items-center gap-2">
        {entries.map((e) => (
          <div key={e.id}
            className="flex items-center gap-1.5 rounded border px-3 py-1.5"
            style={{ borderColor: e.color + "55", background: e.color + "18" }}>
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: e.color }} />
            <a href={`/data/datasets/${e.id}`}
              className="text-sm text-white hover:underline">{e.name}</a>
            {ids.length > 1 && (
              <button onClick={() => removeDataset(e.id)}
                className="ml-1 text-gray-500 hover:text-white text-xs leading-none">×</button>
            )}
          </div>
        ))}
        {ids.length < COMPARE_COLORS.length && (
          <button onClick={() => setShowPicker(true)}
            className="rounded border border-dashed border-gray-700 px-3 py-1.5 text-xs text-gray-500 hover:border-gray-500 hover:text-white transition-colors">
            + Add dataset
          </button>
        )}
      </div>

      {/* Warning for datasets missing characteristics */}
      {noChars && noChars.length > 0 && (
        <div className="rounded border border-yellow-800 bg-yellow-900/20 px-3 py-2 text-xs text-yellow-400">
          {noChars.map((e) => e.name).join(", ")} — characteristics not computed.
          Open the dataset and click <strong>Compute now</strong> on the Characteristics tab first.
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-800">
        {(["endogenous", "exogenous"] as const).map((t) => (
          <button key={t} onClick={() => setCharsTab(t)}
            className={`px-4 py-2 text-sm border-b-2 -mb-px transition-colors capitalize ${
              charsTab === t ? "border-brand-500 text-brand-500" : "border-transparent text-gray-400 hover:text-white"
            }`}>
            {t}
          </button>
        ))}
      </div>

      {charsTab === "endogenous" && (
        <div className="space-y-4">
          <StatsTable entries={entries} />
          <CompareReturnDist entries={entries} />
          <CompareCcdf entries={entries} />
          <CompareAcf entries={entries} />
          <CompareDiffusion entries={entries} />
          <CompareVolClustering entries={entries} />
          <CompareQq entries={entries} />
        </div>
      )}

      {charsTab === "exogenous" && (
        <div className="space-y-4">
          <CompareJumpTail entries={entries} />
          <CompareCdf entries={entries} />
          <CompareLongLagAcf entries={entries} />
        </div>
      )}

      {showPicker && (
        <DatasetPicker currentIds={ids} onAdd={addDataset} onClose={() => setShowPicker(false)} />
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="p-8 text-gray-400 text-sm">Loading…</div>}>
      <CompareContent />
    </Suspense>
  );
}
