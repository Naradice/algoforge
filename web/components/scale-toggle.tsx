"use client";

export type AxisScale = "linear" | "log";

/** Log scale is undefined at zero/negative — clamp the domain to the smallest positive
 * value actually present so Recharts' log axis stays well-defined. */
export function positiveDomain(values: (number | null | undefined)[]): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (typeof v === "number" && v > 0) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (!isFinite(min) || !isFinite(max)) return [0.0001, 1];
  return [min, max];
}

export function ScaleToggle({
  label,
  value,
  onChange,
}: {
  label: string;
  value: AxisScale;
  onChange: (v: AxisScale) => void;
}) {
  return (
    <div className="flex items-center gap-1.5 text-xs text-gray-400">
      <span>{label}</span>
      <div className="flex rounded border border-gray-700 overflow-hidden">
        {(["linear", "log"] as const).map((opt) => (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            className={`px-2 py-0.5 ${value === opt ? "bg-sky-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}
