"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

// ── Field definitions per datasource type ────────────────────────────────────

interface FieldDef {
  key: string;
  label: string;
  type: "text" | "number" | "select" | "date";
  options?: string[];
  placeholder?: string;
  hint?: string;
}

const TIMEFRAME_OPTIONS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];

const TYPE_FIELD_DEFS: Record<string, FieldDef[]> = {
  ohlc_download: [
    {
      key: "client", label: "Provider", type: "select",
      options: ["yfinance", "vantage"],
      hint: "yfinance is free; vantage requires an API key",
    },
    {
      key: "symbol", label: "Symbol", type: "text",
      placeholder: "USDJPY=X",
      hint: "yfinance: USDJPY=X, EURUSD=X, AAPL. Alpha Vantage: USD/JPY",
    },
    { key: "timeframe", label: "Timeframe", type: "select", options: TIMEFRAME_OPTIONS },
    {
      key: "from_ts", label: "From Date", type: "date",
      hint: "yfinance H1/M data is limited to ~730 days lookback — older dates are auto-adjusted",
    },
    { key: "to_ts", label: "To Date", type: "date", placeholder: "", hint: "Leave blank for today" },
  ],
  ddm_simulation: [
    { key: "timeframe", label: "Timeframe", type: "select", options: TIMEFRAME_OPTIONS },
    {
      key: "length", label: "Output Candles", type: "number",
      placeholder: "1000",
      hint: "Number of OHLC candles to generate",
    },
    { key: "initial_price", label: "Initial Price", type: "number", placeholder: "100.0" },
    {
      key: "tick_time", label: "Seconds per Tick", type: "number",
      placeholder: "1.0",
      hint: "Simulated time per iteration. Trades only happen when bid ≥ ask, so trade density per candle is determined by agent dynamics, not directly by this value.",
    },
    { key: "num_agent", label: "Number of Agents", type: "number", placeholder: "50" },
    { key: "seed", label: "Random Seed", type: "number", placeholder: "42" },
  ],
  web_report: [],
  manual_upload: [],
};

const TYPE_DEFAULTS: Record<string, Record<string, string>> = {
  ohlc_download: {
    client: "yfinance",
    symbol: "USDJPY=X",
    timeframe: "H1",
    from_ts: "2022-01-01",
    to_ts: "",
  },
  ddm_simulation: {
    timeframe: "M1",
    length: "1000",
    initial_price: "100",
    tick_time: "1",
    num_agent: "50",
    seed: "42",
  },
  web_report: {},
  manual_upload: {},
};

const TYPE_DESCRIPTIONS: Record<string, { label: string; description: string; status?: string }> = {
  ohlc_download: {
    label: "OHLC Download",
    description: "Download historical candle data from yfinance (free) or Alpha Vantage.",
  },
  ddm_simulation: {
    label: "DDM Simulation",
    description:
      "Generate synthetic OHLC data using a Deterministic Dealer Model. Tick data is stored and resampled to your chosen timeframe on demand.",
  },
  web_report: {
    label: "Web Report",
    description: "Scrape tabular data from a URL using a CSS selector.",
    status: "not-implemented",
  },
  manual_upload: {
    label: "Manual Upload",
    description:
      "Upload a CSV file as a dataset directly. After creating this datasource go to the Datasets page and use the Upload button.",
  },
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function NewDatasourcePage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [type, setType] = useState("ohlc_download");
  const [values, setValues] = useState<Record<string, string>>(TYPE_DEFAULTS["ohlc_download"]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleTypeChange(t: string) {
    setType(t);
    setValues(TYPE_DEFAULTS[t] ?? {});
  }

  function handleFieldChange(key: string, value: string) {
    setValues((prev) => ({ ...prev, [key]: value }));
  }

  function buildConfig(): Record<string, unknown> {
    const fields = TYPE_FIELD_DEFS[type] ?? [];
    const cfg: Record<string, unknown> = {};
    for (const f of fields) {
      const raw = values[f.key] ?? "";
      if (raw === "") continue;
      if (f.type === "number") {
        const n = Number(raw);
        if (!isNaN(n)) cfg[f.key] = n;
      } else {
        cfg[f.key] = raw;
      }
    }
    return cfg;
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/datasources", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, type, config: buildConfig() }),
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.error?.message ?? body.detail ?? "Failed to create datasource");
      }
      const body = await res.json();
      const ds = body.data ?? body;
      router.push(`/data/datasources/${ds.id}`);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  const typeInfo = TYPE_DESCRIPTIONS[type];
  const fields = TYPE_FIELD_DEFS[type] ?? [];

  return (
    <div className="max-w-lg space-y-6">
      <div>
        <a href="/data" className="text-xs text-gray-500 hover:text-white">← Data</a>
        <h1 className="mt-1 text-2xl font-semibold text-white">New Datasource</h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <Field label="Name">
          <input
            className="md-input w-full"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. USDJPY H1"
            required
          />
        </Field>

        <Field label="Type">
          <select className="md-input w-full" value={type} onChange={(e) => handleTypeChange(e.target.value)}>
            {Object.entries(TYPE_DESCRIPTIONS).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
        </Field>

        {/* Type description */}
        <div className={`rounded border px-3 py-2 text-xs ${typeInfo?.status === "not-implemented" ? "border-yellow-800 bg-yellow-900/20 text-yellow-300" : "border-gray-800 bg-gray-900 text-gray-400"}`}>
          {typeInfo?.status === "not-implemented" && <span className="font-semibold text-yellow-400">Not yet implemented. </span>}
          {typeInfo?.description}
        </div>

        {/* Type-specific config fields */}
        {fields.length > 0 && (
          <div className="rounded border border-gray-800 bg-gray-900 p-4 space-y-4">
            <p className="text-xs font-medium text-gray-400 uppercase">Configuration</p>
            {fields.map((f) => (
              <Field key={f.key} label={f.label} hint={f.hint}>
                {f.type === "select" ? (
                  <select
                    className="md-input w-full"
                    value={values[f.key] ?? ""}
                    onChange={(e) => handleFieldChange(f.key, e.target.value)}
                  >
                    {f.options?.map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                ) : f.type === "date" ? (
                  <input
                    type="date"
                    className="md-input w-full"
                    value={values[f.key] ?? ""}
                    onChange={(e) => handleFieldChange(f.key, e.target.value)}
                    placeholder={f.placeholder}
                  />
                ) : (
                  <input
                    type={f.type === "number" ? "number" : "text"}
                    step={f.type === "number" ? "any" : undefined}
                    className="md-input w-full"
                    value={values[f.key] ?? ""}
                    onChange={(e) => handleFieldChange(f.key, e.target.value)}
                    placeholder={f.placeholder}
                  />
                )}
              </Field>
            ))}
          </div>
        )}

        {error && <p className="text-sm text-red-400">{error}</p>}

        <button
          type="submit"
          disabled={saving || typeInfo?.status === "not-implemented"}
          className="w-full rounded bg-brand-500 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-40"
        >
          {saving ? "Creating…" : "Create Datasource"}
        </button>
      </form>

    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1">
      <label className="text-xs text-gray-400 uppercase">{label}</label>
      {children}
      {hint && <p className="text-xs text-gray-500">{hint}</p>}
    </div>
  );
}
