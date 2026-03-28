import { render, screen } from "@testing-library/react";
import { LossChart } from "@/components/loss-chart";

const SAMPLE = [
  { epoch: 1, train_loss: 0.80, val_loss: 0.85 },
  { epoch: 2, train_loss: 0.60, val_loss: 0.65 },
  { epoch: 3, train_loss: 0.40, val_loss: 0.45 },
];

describe("LossChart — empty state", () => {
  it("renders placeholder when data array is empty", () => {
    render(<LossChart data={[]} />);
    expect(screen.getByText("No training data yet")).toBeInTheDocument();
  });

  it("does not render chart container when empty", () => {
    render(<LossChart data={[]} />);
    expect(screen.queryByTestId("recharts-LineChart")).not.toBeInTheDocument();
  });
});

describe("LossChart — with data", () => {
  it("renders the chart container", () => {
    render(<LossChart data={SAMPLE} />);
    expect(screen.queryByText("No training data yet")).not.toBeInTheDocument();
    expect(screen.getByTestId("recharts-LineChart")).toBeInTheDocument();
  });

  it("applies className to outer wrapper", () => {
    const { container } = render(<LossChart data={SAMPLE} className="loss-cls" />);
    expect(container.firstChild).toHaveClass("loss-cls");
  });
});

describe("LossChart — edge cases", () => {
  it("renders with a single epoch without crashing", () => {
    render(<LossChart data={[{ epoch: 1, train_loss: 1.0, val_loss: 1.1 }]} />);
    expect(screen.getByTestId("recharts-LineChart")).toBeInTheDocument();
  });

  it("handles zero loss values", () => {
    render(<LossChart data={[{ epoch: 1, train_loss: 0, val_loss: 0 }]} />);
    expect(screen.queryByText("No training data yet")).not.toBeInTheDocument();
  });
});
