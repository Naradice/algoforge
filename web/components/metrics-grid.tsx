interface MetricsGridProps {
  metrics: Record<string, number>;
  className?: string;
}

const METRIC_LABELS: Record<string, string> = {
  win_rate: "Win Rate",
  total_pnl: "Total PnL",
  sharpe_ratio: "Sharpe Ratio",
  max_drawdown: "Max Drawdown",
  profit_factor: "Profit Factor",
  total_trades: "Total Trades",
  avg_trade_pnl: "Avg Trade PnL",
};

function formatValue(key: string, value: number): string {
  if (key === "total_trades") return String(Math.round(value));
  if (key === "win_rate") return `${(value * 100).toFixed(1)}%`;
  if (key === "total_pnl" || key === "avg_trade_pnl") return `${(value * 100).toFixed(2)}%`;
  if (key === "max_drawdown") return `${(value * 100).toFixed(2)}%`;
  if (key === "profit_factor") return value >= 999 ? "∞" : value.toFixed(2);
  return value.toFixed(4);
}

function getColor(key: string, value: number): string {
  if (key === "total_pnl" || key === "avg_trade_pnl") return value >= 0 ? "text-green-400" : "text-red-400";
  if (key === "sharpe_ratio") return value >= 1 ? "text-green-400" : value >= 0 ? "text-yellow-400" : "text-red-400";
  if (key === "max_drawdown") return value > 0.1 ? "text-red-400" : "text-yellow-400";
  if (key === "win_rate") return value >= 0.5 ? "text-green-400" : "text-red-400";
  return "text-white";
}

export function MetricsGrid({ metrics, className }: MetricsGridProps) {
  const entries = Object.entries(metrics).filter(([key]) => METRIC_LABELS[key]);

  if (entries.length === 0) {
    return <div className={`text-gray-500 text-sm ${className}`}>No metrics available</div>;
  }

  return (
    <div className={`grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6 ${className}`}>
      {entries.map(([key, value]) => (
        <div key={key} className="rounded bg-gray-800 p-3">
          <div className="text-xs text-gray-400 mb-1">{METRIC_LABELS[key] ?? key}</div>
          <div className={`text-lg font-semibold ${getColor(key, value)}`}>{formatValue(key, value)}</div>
        </div>
      ))}
    </div>
  );
}
