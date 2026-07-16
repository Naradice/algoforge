"use client";

import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { type AxisScale, positiveDomain, ScaleToggle } from "@/components/scale-toggle";

export interface RunMetricsPoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
}

export interface RunSeries {
  runId: number;
  label: string; // e.g. "Run #3 lr=0.001 bs=32"
  metrics: RunMetricsPoint[];
}

interface MultiRunLossChartProps {
  runs: RunSeries[];
  height?: number;
}

// Distinct palette — enough for up to 8 runs
const PALETTE = [
  "#0ea5e9", // sky
  "#f97316", // orange
  "#22c55e", // green
  "#a855f7", // purple
  "#ec4899", // pink
  "#eab308", // yellow
  "#14b8a6", // teal
  "#f43f5e", // rose
];

/** Merge per-run metrics into one array keyed by epoch for Recharts */
function mergeByEpoch(runs: RunSeries[]): object[] {
  const map = new Map<number, Record<string, number>>();
  for (const run of runs) {
    for (const pt of run.metrics) {
      if (!map.has(pt.epoch)) map.set(pt.epoch, { epoch: pt.epoch });
      const row = map.get(pt.epoch)!;
      row[`val_loss_${run.runId}`] = pt.val_loss;
      row[`train_loss_${run.runId}`] = pt.train_loss;
    }
  }
  return Array.from(map.values()).sort(
    (a, b) => (a as { epoch: number }).epoch - (b as { epoch: number }).epoch
  );
}

export function MultiRunLossChart({ runs, height = 280 }: MultiRunLossChartProps) {
  const [yScale, setYScale] = useState<AxisScale>("linear");
  const [xScale, setXScale] = useState<AxisScale>("linear");

  if (runs.length === 0 || runs.every((r) => r.metrics.length === 0)) {
    return (
      <div
        className="flex items-center justify-center text-gray-500 text-sm"
        style={{ height }}
      >
        No metrics available
      </div>
    );
  }

  const data = mergeByEpoch(runs);
  const valLossKeys = runs.map((r) => `val_loss_${r.runId}`);
  const yDomain = yScale === "log"
    ? positiveDomain(data.flatMap((row) => valLossKeys.map((k) => (row as Record<string, number>)[k])))
    : undefined;
  const xDomain = xScale === "log"
    ? positiveDomain(data.map((row) => (row as { epoch: number }).epoch))
    : undefined;
  const yTickFormatter = (v: number) => (yScale === "log" ? v.toExponential(1) : v.toFixed(4));

  return (
    <div>
      <div className="flex items-center justify-end gap-4 mb-1">
        <ScaleToggle label="X scale" value={xScale} onChange={setXScale} />
        <ScaleToggle label="Y scale" value={yScale} onChange={setYScale} />
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="epoch"
            stroke="#374151"
            tick={{ fontSize: 11, fill: "#6b7280" }}
            label={{ value: "Epoch", position: "insideBottomRight", offset: -5, fill: "#6b7280", fontSize: 11 }}
            scale={xScale}
            domain={xDomain}
            allowDataOverflow={xScale === "log"}
          />
          <YAxis
            stroke="#374151"
            tick={{ fontSize: 11, fill: "#6b7280" }}
            tickFormatter={yTickFormatter}
            width={60}
            scale={yScale}
            domain={yDomain}
            allowDataOverflow={yScale === "log"}
          />
          <Tooltip
            contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 4, fontSize: 11 }}
            labelFormatter={(v) => `Epoch ${v}`}
            formatter={(value: number, name: string) => {
              const match = name.match(/^val_loss_(\d+)$/);
              return [value.toFixed(6), match ? `Run #${match[1]} val` : name];
            }}
          />
          <Legend
            formatter={(name: string) => {
              const match = name.match(/^val_loss_(\d+)$/);
              if (!match) return name;
              const run = runs.find((r) => r.runId === parseInt(match[1]));
              return run?.label ?? `Run #${match[1]}`;
            }}
            wrapperStyle={{ fontSize: 11, color: "#9ca3af" }}
          />
          {runs.map((run, i) => (
            <Line
              key={run.runId}
              type="monotone"
              dataKey={`val_loss_${run.runId}`}
              stroke={PALETTE[i % PALETTE.length]}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
