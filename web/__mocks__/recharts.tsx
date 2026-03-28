import React from "react";

const make = (name: string) => {
  const C = ({ children }: { children?: React.ReactNode }) => (
    <div data-testid={`recharts-${name}`}>{children}</div>
  );
  C.displayName = name;
  return C;
};

export const ResponsiveContainer = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>;
export const AreaChart = make("AreaChart");
export const LineChart = make("LineChart");
export const BarChart = make("BarChart");
export const ScatterChart = make("ScatterChart");
export const Area = () => null;
export const Line = () => null;
export const Bar = () => null;
export const Scatter = () => null;
export const XAxis = () => null;
export const YAxis = () => null;
export const CartesianGrid = () => null;
export const Tooltip = () => null;
export const Legend = () => null;
export const ReferenceLine = () => null;
export const ReferenceArea = () => null;
