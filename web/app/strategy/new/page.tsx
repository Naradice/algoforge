"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const EXAMPLE_DEFINITIONS: Record<string, object> = {
  macd_rsi: {
    symbol: "AAPL",
    indicators: [
      { id: "macd", type: "macd", params: { fast: 12, slow: 26, signal_period: 9 } },
      { id: "rsi",  type: "rsi",  params: { period: 14 } },
      { id: "atr",  type: "atr",  params: { period: 14 } },
    ],
    entry: {
      direction: "buy",
      conditions: [
        { left: "macd_line", op: ">", right: "macd_signal" },
        { left: "rsi",       op: "<", right: 70 },
      ],
      logic: "and",
    },
    exit: {
      conditions: [
        { left: "macd_line", op: "<", right: "macd_signal" },
      ],
      logic: "or",
    },
    risk: { sl_pct: 0.02, tp_pct: 0.04, position_size: 1.0 },
  },

  ml_signal: {
    symbol: "AAPL",
    indicators: [
      { id: "macd", type: "macd", params: { fast: 12, slow: 26, signal_period: 9 } },
      { id: "rsi",  type: "rsi",  params: { period: 14 } },
    ],
    entry: {
      direction: "buy",
      conditions: [
        { type: "ml_signal", model_id: 1, direction: "buy", step: 1, min_confidence: 0.0 },
        { left: "rsi", op: "<", right: 75 },
      ],
      logic: "and",
    },
    exit: {
      conditions: [
        { type: "ml_signal", model_id: 1, direction: "sell", step: 1 },
      ],
      logic: "or",
    },
    risk: { sl_pct: 0.02, tp_pct: 0.05, position_size: 1.0 },
  },

  llm_signal: {
    symbol: "AAPL",
    indicators: [
      { id: "macd", type: "macd", params: { fast: 12, slow: 26, signal_period: 9 } },
      { id: "rsi",  type: "rsi",  params: { period: 14 } },
    ],
    entry: {
      direction: "buy",
      conditions: [
        {
          type: "llm_signal",
          direction: "buy",
          lookback: 10,
          model: "gemini-2.0-flash",
          columns: ["close", "volume", "macd_line", "macd_signal", "rsi"],
          cache: true,
        },
      ],
      logic: "and",
    },
    exit: {
      conditions: [
        {
          type: "llm_signal",
          direction: "sell",
          lookback: 10,
          model: "gemini-2.0-flash",
          columns: ["close", "volume", "macd_line", "macd_signal", "rsi"],
          cache: true,
        },
      ],
      logic: "or",
    },
    risk: { sl_pct: 0.025, tp_pct: 0.05, position_size: 1.0 },
  },
};

export default function NewStrategyPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [template, setTemplate] = useState("macd_rsi");
  const [defText, setDefText] = useState(
    JSON.stringify(EXAMPLE_DEFINITIONS["macd_rsi"], null, 2)
  );

  function handleTemplateChange(t: string) {
    setTemplate(t);
    setDefText(JSON.stringify(EXAMPLE_DEFINITIONS[t], null, 2));
  }
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    let definition: object;
    try {
      definition = JSON.parse(defText);
    } catch {
      setError("Definition is not valid JSON");
      return;
    }

    setSaving(true);
    try {
      const res = await fetch("/api/v1/strategies", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description, definition }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      const body = await res.json();
      const strategy = body.data ?? body;
      router.push(`/strategy/${strategy.id}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="max-w-2xl space-y-6">
      <h1 className="text-2xl font-semibold text-white">New Strategy</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="mb-1 block text-sm text-gray-400">Name</label>
          <input
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            placeholder="MACD + RSI Long"
          />
        </div>

        <div>
          <label className="mb-1 block text-sm text-gray-400">Description</label>
          <input
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            placeholder="Optional description"
          />
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between">
            <label className="text-sm text-gray-400">Strategy Definition (JSON)</label>
            <select
              value={template}
              onChange={(e) => handleTemplateChange(e.target.value)}
              className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-xs text-gray-300 focus:outline-none"
            >
              <option value="macd_rsi">Example: MACD + RSI</option>
              <option value="ml_signal">Example: ML Signal</option>
              <option value="llm_signal">Example: LLM Signal</option>
            </select>
          </div>
          <p className="mb-2 text-xs text-gray-500">
            Condition types: standard comparison{" "}
            <code className="text-gray-300">{"{ left, op, right }"}</code>, ML model{" "}
            <code className="text-gray-300">{"{ type: \"ml_signal\", model_id, direction }"}</code>,
            or LLM{" "}
            <code className="text-gray-300">{"{ type: \"llm_signal\", direction, lookback }"}</code>.
          </p>
          <textarea
            value={defText}
            onChange={(e) => setDefText(e.target.value)}
            rows={28}
            className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 font-mono text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
          />
        </div>

        {error && <p className="text-sm text-red-400">{error}</p>}

        <div className="flex gap-3">
          <button
            type="submit"
            disabled={saving}
            className="rounded bg-brand-500 px-4 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-50"
          >
            {saving ? "Creating…" : "Create Strategy"}
          </button>
          <a
            href="/strategy"
            className="rounded border border-gray-700 px-4 py-2 text-sm text-gray-400 hover:text-white"
          >
            Cancel
          </a>
        </div>
      </form>
    </div>
  );
}
