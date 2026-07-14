"use client";

// Compact numeric summary of the 5 "Structure" characteristics (long-range dependence,
// spectral periodicity, multiscale wavelet structure, entropy/nonlinearity, regime changes) —
// see backend/data/characteristics.py. Used wherever these need a quick glance without the
// full chart treatment on the dataset detail page's "Structure" tab.

function num(v: unknown, fmt: (n: number) => string): string {
  return typeof v === "number" && !isNaN(v) ? fmt(v) : "—";
}

export function StructureStatGrid({ characteristics }: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  characteristics: Record<string, any> | null;
}) {
  if (!characteristics) return null;

  const lrd = characteristics.long_range_dependence ?? {};
  const sp = characteristics.spectral_periodicity ?? {};
  const mw = characteristics.multiscale_wavelet ?? {};
  const cn = characteristics.complexity_nonlinearity ?? {};
  const rc = characteristics.regime_changes ?? {};

  const stats = [
    { label: "Hurst", value: num(lrd.hurst, (v) => v.toFixed(4)) },
    { label: "Behavior", value: lrd.interpretation ?? "—" },
    { label: "Memory length", value: num(lrd.memory_length, (v) => String(v)) },
    { label: "Periodicity strength", value: num(sp.periodicity_strength, (v) => v.toFixed(2)) },
    { label: "Spectral entropy", value: num(sp.spectral_entropy, (v) => v.toFixed(3)) },
    { label: "Wavelet flatness", value: num(mw.flatness_score, (v) => v.toFixed(3)) },
    { label: "Permutation entropy", value: num(cn.permutation_entropy, (v) => v.toFixed(3)) },
    { label: "Sample entropy", value: num(cn.sample_entropy, (v) => v.toFixed(3)) },
    { label: "Nonlinear (BDS)?", value: cn.nonlinear === true ? "Yes" : cn.nonlinear === false ? "No" : "—" },
    { label: "Changepoints", value: num(rc.n_changepoints, (v) => String(v)) },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
      {stats.map(({ label, value }) => (
        <div key={label} className="rounded border border-gray-800 bg-gray-950 p-2">
          <p className="text-xs text-gray-500 uppercase">{label}</p>
          <p className="mt-0.5 text-sm font-bold text-white font-mono">{value}</p>
        </div>
      ))}
    </div>
  );
}
