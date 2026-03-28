"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface LossPoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
}

interface LossChartProps {
  data: LossPoint[];
  className?: string;
}

export function LossChart({ data, className }: LossChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className={`flex items-center justify-center h-48 text-gray-500 text-sm ${className}`}>
        No training data yet
      </div>
    );
  }

  return (
    <div className={className}>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="epoch" stroke="#6b7280" tick={{ fontSize: 11 }} label={{ value: "Epoch", position: "insideBottom", offset: -5, fill: "#6b7280", fontSize: 11 }} />
          <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151", borderRadius: 4 }}
            labelStyle={{ color: "#d1d5db" }}
            labelFormatter={(v) => `Epoch ${v}`}
          />
          <Legend />
          <Line type="monotone" dataKey="train_loss" stroke="#f97316" strokeWidth={2} dot={false} name="Train Loss" />
          <Line type="monotone" dataKey="val_loss" stroke="#0ea5e9" strokeWidth={2} dot={false} name="Val Loss" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
