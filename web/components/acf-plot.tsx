"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface ACFPoint {
  lag: number;
  value: number;
  ci_upper?: number;
  ci_lower?: number;
}

interface ACFPlotProps {
  data: ACFPoint[];
  title?: string;
  className?: string;
}

export function ACFPlot({ data, title = "ACF", className }: ACFPlotProps) {
  if (!data || data.length === 0) {
    return <div className={`flex items-center justify-center h-40 text-gray-500 text-sm ${className}`}>No data</div>;
  }

  const ciUpper = data[0]?.ci_upper;
  const ciLower = data[0]?.ci_lower;

  return (
    <div className={className}>
      {title && <div className="text-xs text-gray-400 mb-2">{title}</div>}
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="lag" stroke="#6b7280" tick={{ fontSize: 10 }} />
          <YAxis domain={[-1, 1]} stroke="#6b7280" tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 4 }}
            formatter={(v: number) => [v.toFixed(4), "ACF"]}
          />
          <Bar dataKey="value" fill="#0ea5e9" />
          {ciUpper != null && <ReferenceLine y={ciUpper} stroke="#f97316" strokeDasharray="4 2" />}
          {ciLower != null && <ReferenceLine y={ciLower} stroke="#f97316" strokeDasharray="4 2" />}
          <ReferenceLine y={0} stroke="#6b7280" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
