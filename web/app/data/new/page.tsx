"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const TYPE_CONFIGS: Record<string, Record<string, string>> = {
  ohlc_download: {
    client: "yfinance",
    symbol: "EURUSD=X",
    timeframe: "H1",
    from_ts: "2020-01-01",
    to_ts: "",
  },
  ddm_simulation: {
    num_agent: "50",
    initial_price: "100",
    length: "50000",
    tick_time: "1",
    timeframe: "M1",
    seed: "42",
  },
  web_report: {
    url: "",
    selector: "table",
    format: "table",
  },
  manual_upload: {},
};

export default function NewDatasourcePage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [type, setType] = useState("ohlc_download");
  const [config, setConfig] = useState(TYPE_CONFIGS["ohlc_download"]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function handleTypeChange(t: string) {
    setType(t);
    setConfig(TYPE_CONFIGS[t] ?? {});
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const res = await fetch("/api/v1/datasources", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, type, config }),
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

  return (
    <div className="max-w-lg space-y-6">
      <div>
        <a href="/data" className="text-xs text-gray-500 hover:text-white">← Data</a>
        <h1 className="mt-1 text-2xl font-semibold text-white">New Datasource</h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <Field label="Name">
          <input
            className="input w-full"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. EURUSD H1 2020–2024"
            required
          />
        </Field>

        <Field label="Type">
          <select className="input w-full" value={type} onChange={(e) => handleTypeChange(e.target.value)}>
            <option value="ohlc_download">OHLC Download (yfinance / Alpha Vantage)</option>
            <option value="ddm_simulation">DDM Simulation</option>
            <option value="web_report">Web Report</option>
            <option value="manual_upload">Manual Upload</option>
          </select>
        </Field>

        <Field label="Config (JSON)">
          <textarea
            className="input w-full font-mono text-xs h-48"
            value={JSON.stringify(config, null, 2)}
            onChange={(e) => {
              try { setConfig(JSON.parse(e.target.value)); } catch { /* ignore parse errors while typing */ }
            }}
          />
        </Field>

        {error && <p className="text-sm text-red-400">{error}</p>}

        <button
          type="submit"
          disabled={saving}
          className="w-full rounded bg-brand-500 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-40"
        >
          {saving ? "Creating…" : "Create Datasource"}
        </button>
      </form>

      <style>{`.input { @apply rounded border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white focus:border-brand-500 focus:outline-none; }`}</style>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <label className="text-xs text-gray-400 uppercase">{label}</label>
      {children}
    </div>
  );
}
