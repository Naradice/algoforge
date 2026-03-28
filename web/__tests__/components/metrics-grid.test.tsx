import { render, screen } from "@testing-library/react";
import { MetricsGrid } from "@/components/metrics-grid";

describe("MetricsGrid — empty state", () => {
  it("shows placeholder when metrics is empty", () => {
    render(<MetricsGrid metrics={{}} />);
    expect(screen.getByText("No metrics available")).toBeInTheDocument();
  });

  it("shows placeholder when all keys are unknown", () => {
    render(<MetricsGrid metrics={{ foo: 1, bar: 2 }} />);
    expect(screen.getByText("No metrics available")).toBeInTheDocument();
  });
});

describe("MetricsGrid — labels", () => {
  it("renders Win Rate label", () => {
    render(<MetricsGrid metrics={{ win_rate: 0.6 }} />);
    expect(screen.getByText("Win Rate")).toBeInTheDocument();
  });

  it("renders Total Trades label", () => {
    render(<MetricsGrid metrics={{ total_trades: 50 }} />);
    expect(screen.getByText("Total Trades")).toBeInTheDocument();
  });

  it("renders all known labels together", () => {
    render(
      <MetricsGrid
        metrics={{
          win_rate: 0.5,
          total_pnl: 0.1,
          sharpe_ratio: 1.2,
          max_drawdown: 0.05,
          profit_factor: 2,
          total_trades: 100,
          avg_trade_pnl: 0.001,
        }}
      />
    );
    expect(screen.getByText("Win Rate")).toBeInTheDocument();
    expect(screen.getByText("Total PnL")).toBeInTheDocument();
    expect(screen.getByText("Sharpe Ratio")).toBeInTheDocument();
    expect(screen.getByText("Max Drawdown")).toBeInTheDocument();
    expect(screen.getByText("Profit Factor")).toBeInTheDocument();
    expect(screen.getByText("Total Trades")).toBeInTheDocument();
    expect(screen.getByText("Avg Trade PnL")).toBeInTheDocument();
  });
});

describe("MetricsGrid — value formatting", () => {
  it("formats win_rate as percentage (×100, 1 decimal)", () => {
    render(<MetricsGrid metrics={{ win_rate: 0.623 }} />);
    expect(screen.getByText("62.3%")).toBeInTheDocument();
  });

  it("formats total_trades as rounded integer", () => {
    render(<MetricsGrid metrics={{ total_trades: 42.7 }} />);
    expect(screen.getByText("43")).toBeInTheDocument();
  });

  it("formats total_pnl as percentage with 2 decimals", () => {
    render(<MetricsGrid metrics={{ total_pnl: 0.1567 }} />);
    expect(screen.getByText("15.67%")).toBeInTheDocument();
  });

  it("formats avg_trade_pnl as percentage with 2 decimals", () => {
    render(<MetricsGrid metrics={{ avg_trade_pnl: 0.005 }} />);
    expect(screen.getByText("0.50%")).toBeInTheDocument();
  });

  it("formats max_drawdown as percentage with 2 decimals", () => {
    render(<MetricsGrid metrics={{ max_drawdown: 0.0823 }} />);
    expect(screen.getByText("8.23%")).toBeInTheDocument();
  });

  it("formats profit_factor as ∞ when value >= 999", () => {
    render(<MetricsGrid metrics={{ profit_factor: 999 }} />);
    expect(screen.getByText("∞")).toBeInTheDocument();
  });

  it("formats profit_factor as decimal when < 999", () => {
    render(<MetricsGrid metrics={{ profit_factor: 2.5 }} />);
    expect(screen.getByText("2.50")).toBeInTheDocument();
  });

  it("formats sharpe_ratio to 4 decimal places", () => {
    render(<MetricsGrid metrics={{ sharpe_ratio: 1.23456 }} />);
    expect(screen.getByText("1.2346")).toBeInTheDocument();
  });
});

describe("MetricsGrid — color coding", () => {
  it("colors positive total_pnl green", () => {
    render(<MetricsGrid metrics={{ total_pnl: 0.1 }} />);
    expect(screen.getByText("10.00%")).toHaveClass("text-green-400");
  });

  it("colors negative total_pnl red", () => {
    render(<MetricsGrid metrics={{ total_pnl: -0.05 }} />);
    expect(screen.getByText("-5.00%")).toHaveClass("text-red-400");
  });

  it("colors sharpe_ratio green when >= 1", () => {
    render(<MetricsGrid metrics={{ sharpe_ratio: 1.5 }} />);
    expect(screen.getByText("1.5000")).toHaveClass("text-green-400");
  });

  it("colors sharpe_ratio yellow when 0 <= x < 1", () => {
    render(<MetricsGrid metrics={{ sharpe_ratio: 0.7 }} />);
    expect(screen.getByText("0.7000")).toHaveClass("text-yellow-400");
  });

  it("colors sharpe_ratio red when negative", () => {
    render(<MetricsGrid metrics={{ sharpe_ratio: -0.3 }} />);
    expect(screen.getByText("-0.3000")).toHaveClass("text-red-400");
  });

  it("colors win_rate green when >= 0.5", () => {
    render(<MetricsGrid metrics={{ win_rate: 0.5 }} />);
    expect(screen.getByText("50.0%")).toHaveClass("text-green-400");
  });

  it("colors win_rate red when < 0.5", () => {
    render(<MetricsGrid metrics={{ win_rate: 0.4 }} />);
    expect(screen.getByText("40.0%")).toHaveClass("text-red-400");
  });

  it("colors max_drawdown red when > 0.1", () => {
    render(<MetricsGrid metrics={{ max_drawdown: 0.15 }} />);
    expect(screen.getByText("15.00%")).toHaveClass("text-red-400");
  });

  it("colors max_drawdown yellow when <= 0.1", () => {
    render(<MetricsGrid metrics={{ max_drawdown: 0.08 }} />);
    expect(screen.getByText("8.00%")).toHaveClass("text-yellow-400");
  });
});
