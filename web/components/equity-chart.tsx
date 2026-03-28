"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface EquityPoint {
  timestamp: string;
  equity: number;
  drawdown: number;
}

interface EquityChartProps {
  data: EquityPoint[];
  className?: string;
}

export function EquityChart({ data, className }: EquityChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className={`flex items-center justify-center h-48 text-gray-500 text-sm ${className}`}>
        No equity data available
      </div>
    );
  }

  const formatted = data.map((d) => ({
    ...d,
    timestamp: d.timestamp ? new Date(d.timestamp).toLocaleDateString() : "",
    equity_pct: +(d.equity * 100).toFixed(3),
    drawdown_pct: +(d.drawdown * 100).toFixed(3),
  }));

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={formatted} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="timestamp" stroke="#6b7280" tick={{ fontSize: 11 }} />
          <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} tickFormatter={(v) => `${v}%`} />
          <Tooltip
            contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 4 }}
            labelStyle={{ color: "#d1d5db" }}
            formatter={(value: number, name: string) => [`${value}%`, name === "equity_pct" ? "Equity" : "Drawdown"]}
          />
          <Legend formatter={(v) => (v === "equity_pct" ? "Equity" : "Drawdown")} />
          <Area type="monotone" dataKey="equity_pct" stroke="#0ea5e9" fill="url(#equityGradient)" strokeWidth={2} />
          <Area type="monotone" dataKey="drawdown_pct" stroke="#ef4444" fill="url(#drawdownGradient)" strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
