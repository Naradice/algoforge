import { render, screen } from "@testing-library/react";
import { StatusBadge } from "@/components/status-badge";

describe("StatusBadge", () => {
  it("renders the status text", () => {
    render(<StatusBadge status="active" />);
    expect(screen.getByText("active")).toBeInTheDocument();
  });

  it("applies green color for active", () => {
    render(<StatusBadge status="active" />);
    expect(screen.getByText("active")).toHaveClass("text-green-400");
  });

  it("applies blue color for running", () => {
    render(<StatusBadge status="running" />);
    expect(screen.getByText("running")).toHaveClass("text-blue-400");
  });

  it("applies red color for error", () => {
    render(<StatusBadge status="error" />);
    expect(screen.getByText("error")).toHaveClass("text-red-400");
  });

  it("applies yellow color for pending", () => {
    render(<StatusBadge status="pending" />);
    expect(screen.getByText("pending")).toHaveClass("text-yellow-400");
  });

  it("applies teal color for trained", () => {
    render(<StatusBadge status="trained" />);
    expect(screen.getByText("trained")).toHaveClass("text-teal-400");
  });

  it.each([
    "inactive",
    "stopped",
    "created",
    "idle",
  ])('applies gray color for status "%s"', (status) => {
    render(<StatusBadge status={status} />);
    expect(screen.getByText(status)).toHaveClass("text-gray-400");
  });

  it("applies darker gray color for status \"archived\"", () => {
    render(<StatusBadge status="archived" />);
    expect(screen.getByText("archived")).toHaveClass("text-gray-500");
  });

  it("falls back to gray for unknown status", () => {
    render(<StatusBadge status="totally_unknown" />);
    expect(screen.getByText("totally_unknown")).toHaveClass("text-gray-400");
  });

  it("applies extra className when provided", () => {
    render(<StatusBadge status="active" className="ml-4 font-bold" />);
    const el = screen.getByText("active");
    expect(el).toHaveClass("ml-4");
    expect(el).toHaveClass("font-bold");
  });

  it("always includes base pill classes", () => {
    render(<StatusBadge status="active" />);
    const el = screen.getByText("active");
    expect(el).toHaveClass("rounded-full");
    expect(el).toHaveClass("text-xs");
  });
});
