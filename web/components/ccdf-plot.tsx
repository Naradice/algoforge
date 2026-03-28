"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface CCDFPoint {
  magnitude: number;
  probability: number;
}

interface CCDFPlotProps {
  data: CCDFPoint[];
  className?: string;
}

export function CCDFPlot({ data, className }: CCDFPlotProps) {
  if (!data || data.length === 0) {
    return <div className={`flex items-center justify-center h-40 text-gray-500 text-sm ${className}`}>No data</div>;
  }

  return (
    <div className={className}>
      <div className="text-xs text-gray-400 mb-2">CCDF (Return Tail Distribution)</div>
      <ResponsiveContainer width="100%" height={200}>
        <ScatterChart margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="magnitude"
            name="Magnitude"
            stroke="#6b7280"
            tick={{ fontSize: 10 }}
            tickFormatter={(v: number) => v.toExponential(1)}
            type="number"
          />
          <YAxis
            dataKey="probability"
            name="P(|ret| > x)"
            stroke="#6b7280"
            tick={{ fontSize: 10 }}
            tickFormatter={(v: number) => v.toFixed(2)}
            type="number"
          />
          <Tooltip
            contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 4 }}
            formatter={(v: number, name: string) => [
              name === "magnitude" ? v.toExponential(3) : v.toFixed(4),
              name === "magnitude" ? "|Return|" : "P(|ret|>x)",
            ]}
          />
          <Scatter data={data} fill="#0ea5e9" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
