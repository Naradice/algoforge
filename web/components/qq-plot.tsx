"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface QQPoint {
  theoretical: number;
  sample: number;
}

interface QQPlotProps {
  data: QQPoint[];
  className?: string;
}

export function QQPlot({ data, className }: QQPlotProps) {
  if (!data || data.length === 0) {
    return <div className={`flex items-center justify-center h-40 text-gray-500 text-sm ${className}`}>No data</div>;
  }

  // Reference line: y = x (perfect normal)
  const minVal = Math.min(...data.map((d) => Math.min(d.theoretical, d.sample)));
  const maxVal = Math.max(...data.map((d) => Math.max(d.theoretical, d.sample)));

  return (
    <div className={className}>
      <div className="text-xs text-gray-400 mb-2">QQ Plot (vs Normal)</div>
      <ResponsiveContainer width="100%" height={200}>
        <ScatterChart margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="theoretical" name="Theoretical" stroke="#6b7280" tick={{ fontSize: 10 }} type="number" />
          <YAxis dataKey="sample" name="Sample" stroke="#6b7280" tick={{ fontSize: 10 }} type="number" />
          <Tooltip
            contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 4 }}
            formatter={(v: number, name: string) => [v.toFixed(4), name === "theoretical" ? "Theoretical" : "Sample"]}
          />
          <Scatter data={data} fill="#0ea5e9" />
          <ReferenceLine
            segment={[{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }]}
            stroke="#f97316"
            strokeDasharray="4 2"
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
