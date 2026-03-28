import { render, screen } from "@testing-library/react";
import { EquityChart } from "@/components/equity-chart";

const SAMPLE = [
  { timestamp: "2024-01-01T00:00:00Z", equity: 0.05, drawdown: 0.01 },
  { timestamp: "2024-01-02T00:00:00Z", equity: 0.08, drawdown: 0.02 },
  { timestamp: "2024-01-03T00:00:00Z", equity: 0.03, drawdown: 0.05 },
];

describe("EquityChart — empty state", () => {
  it("renders empty message when data array is empty", () => {
    render(<EquityChart data={[]} />);
    expect(screen.getByText("No equity data available")).toBeInTheDocument();
  });

  it("does not render chart container when empty", () => {
    render(<EquityChart data={[]} />);
    expect(screen.queryByTestId("recharts-AreaChart")).not.toBeInTheDocument();
  });
});

describe("EquityChart — with data", () => {
  it("renders the chart container when data is provided", () => {
    render(<EquityChart data={SAMPLE} />);
    expect(screen.queryByText("No equity data available")).not.toBeInTheDocument();
    expect(screen.getByTestId("recharts-AreaChart")).toBeInTheDocument();
  });

  it("applies className to the outer wrapper", () => {
    const { container } = render(<EquityChart data={SAMPLE} className="custom-cls" />);
    expect(container.firstChild).toHaveClass("custom-cls");
  });
});

describe("EquityChart — edge cases", () => {
  it("handles a single data point without crashing", () => {
    render(<EquityChart data={[{ timestamp: "2024-01-01T00:00:00Z", equity: 0, drawdown: 0 }]} />);
    expect(screen.queryByText("No equity data available")).not.toBeInTheDocument();
  });

  it("handles equity = 0 and drawdown = 0", () => {
    render(<EquityChart data={[{ timestamp: "2024-01-01T00:00:00Z", equity: 0, drawdown: 0 }]} />);
    expect(screen.getByTestId("recharts-AreaChart")).toBeInTheDocument();
  });
});
