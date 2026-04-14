"use client";

import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { IndicatorSeries } from "./ohlc-chart";

interface OscillatorChartProps {
  /** Group label shown in the panel header, e.g. "rsi", "macd" */
  group: string;
  /** All series belonging to this group */
  series: Record<string, IndicatorSeries>;
  height?: number;
}

/** Merge all series in the group into a single array keyed by time */
function mergeSeriesData(series: Record<string, IndicatorSeries>): object[] {
  const map = new Map<number, Record<string, number>>();
  for (const [name, s] of Object.entries(series)) {
    for (const pt of s.data) {
      if (!map.has(pt.time)) map.set(pt.time, { time: pt.time });
      (map.get(pt.time) as Record<string, number>)[name] = pt.value;
    }
  }
  return Array.from(map.values()).sort((a, b) => (a as { time: number }).time - (b as { time: number }).time);
}

function formatTime(unix: number): string {
  return new Date(unix * 1000).toLocaleDateString();
}

// Reference lines for common oscillators
const REFERENCE_LINES: Record<string, number[]> = {
  rsi: [30, 70],
};

export function OscillatorChart({ group, series, height = 160 }: OscillatorChartProps) {
  const data = mergeSeriesData(series);
  const refLines = REFERENCE_LINES[group] ?? [];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 4, right: 30, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="time"
          tickFormatter={formatTime}
          stroke="#374151"
          tick={{ fontSize: 10, fill: "#6b7280" }}
          interval="preserveStartEnd"
        />
        <YAxis stroke="#374151" tick={{ fontSize: 10, fill: "#6b7280" }} width={45} />
        <Tooltip
          contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 4, fontSize: 11 }}
          labelFormatter={(v: number) => new Date(v * 1000).toLocaleString()}
          formatter={(value: number, name: string) => [value.toFixed(4), name]}
        />
        <Legend iconSize={8} wrapperStyle={{ fontSize: 11, color: "#9ca3af" }} />
        {refLines.map((y) => (
          <ReferenceLine key={y} y={y} stroke="#374151" strokeDasharray="4 2" />
        ))}
        {Object.entries(series).map(([name, s]) =>
          s.type === "histogram" ? (
            <Bar
              key={name}
              dataKey={name}
              fill={s.color}
              opacity={0.7}
              isAnimationActive={false}
            />
          ) : (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={s.color}
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          )
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
