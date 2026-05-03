"use client";

import { useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { StrategyEditor } from "@/components/strategy-editor";
import { apiFetch } from "@/lib/fetcher";

// ---------------------------------------------------------------------------
// Strategy templates
// ---------------------------------------------------------------------------

type StrategyTemplate = {
  id: string;
  name: string;
  description: string;
  tags: string[];
  definition: object;
};

const defaultRisk = {
  risk_type: "percent_equity",
  position_size: 1.0,
  risk_pct: 0.01,
  atr_multiplier: 2.0,
  sl_pct: 0.02,
  tp_pct: 0.04,
  slippage_pct: 0.0005,
  commission_pct: 0.001,
  max_positions: 1,
  daily_loss_limit_pct: 0.0,
  cooldown_bars: 0,
  trailing_stop: false,
  trailing_atr_multiplier: 3.0,
  trailing_clip_with_price: false,
  trailing_only_in_profit: true,
};

const TEMPLATES: StrategyTemplate[] = [
  {
    id: "scratch",
    name: "",
    description: "Start with a blank canvas",
    tags: [],
    definition: {},
  },
  {
    id: "macd_crossover",
    name: "MACD Crossover",
    description:
      "Buy when MACD line crosses above the signal line; exit when it crosses below. Classic trend-following approach.",
    tags: ["trend", "momentum"],
    definition: {
      symbol: "AAPL",
      indicators: [
        { id: "macd", category: "technical", type: "macd", params: { fast: 12, slow: 26, signal_period: 9 } },
      ],
      long: {
        entry: {
          direction: "buy",
          conditions: [{ left: "macd_line", op: ">", right: "macd_signal" }],
          logic: "and",
        },
        exit: {
          direction: "buy",
          conditions: [{ left: "macd_line", op: "<", right: "macd_signal" }],
          logic: "and",
        },
      },
      risk: { ...defaultRisk },
    },
  },
  {
    id: "rsi_mean_reversion",
    name: "RSI Mean Reversion",
    description:
      "Buy oversold conditions (RSI < 30) and short overbought (RSI > 70). Exit when RSI returns to neutral (50).",
    tags: ["mean-reversion", "oscillator"],
    definition: {
      symbol: "AAPL",
      indicators: [
        { id: "rsi", category: "technical", type: "rsi", params: { period: 14 } },
      ],
      long: {
        entry: {
          direction: "buy",
          conditions: [{ left: "rsi", op: "<", right: 30 }],
          logic: "and",
        },
        exit: {
          direction: "buy",
          conditions: [{ left: "rsi", op: ">", right: 50 }],
          logic: "and",
        },
      },
      short: {
        entry: {
          direction: "sell",
          conditions: [{ left: "rsi", op: ">", right: 70 }],
          logic: "and",
        },
        exit: {
          direction: "sell",
          conditions: [{ left: "rsi", op: "<", right: 50 }],
          logic: "and",
        },
      },
      risk: { ...defaultRisk },
    },
  },
  {
    id: "ema_crossover",
    name: "EMA Golden / Death Cross",
    description:
      "Buy when fast EMA (20) crosses above slow EMA (50); exit when it crosses below. Standard trend-following strategy.",
    tags: ["trend", "crossover"],
    definition: {
      symbol: "AAPL",
      indicators: [
        { id: "ema_fast", category: "technical", type: "ema", params: { period: 20 } },
        { id: "ema_slow", category: "technical", type: "ema", params: { period: 50 } },
      ],
      long: {
        entry: {
          direction: "buy",
          conditions: [{ left: "ema_fast", op: ">", right: "ema_slow" }],
          logic: "and",
        },
        exit: {
          direction: "buy",
          conditions: [{ left: "ema_fast", op: "<", right: "ema_slow" }],
          logic: "and",
        },
      },
      short: {
        entry: {
          direction: "sell",
          conditions: [{ left: "ema_fast", op: "<", right: "ema_slow" }],
          logic: "and",
        },
        exit: {
          direction: "sell",
          conditions: [{ left: "ema_fast", op: ">", right: "ema_slow" }],
          logic: "and",
        },
      },
      risk: { ...defaultRisk },
    },
  },
  {
    id: "bollinger_band_reversion",
    name: "Bollinger Band Reversion",
    description:
      "Buy when price touches the lower band (oversold); short when it touches the upper band. Exit at the middle band (SMA).",
    tags: ["mean-reversion", "volatility"],
    definition: {
      symbol: "AAPL",
      indicators: [
        { id: "bb", category: "technical", type: "bb", params: { period: 20, std_dev: 2 } },
      ],
      long: {
        entry: {
          direction: "buy",
          conditions: [{ left: "close", op: "<", right: "bb_lower" }],
          logic: "and",
        },
        exit: {
          direction: "buy",
          conditions: [{ left: "close", op: ">", right: "bb_middle" }],
          logic: "and",
        },
      },
      short: {
        entry: {
          direction: "sell",
          conditions: [{ left: "close", op: ">", right: "bb_upper" }],
          logic: "and",
        },
        exit: {
          direction: "sell",
          conditions: [{ left: "close", op: "<", right: "bb_middle" }],
          logic: "and",
        },
      },
      risk: { ...defaultRisk },
    },
  },
  {
    id: "rsi_macd_combo",
    name: "RSI + MACD Combo",
    description:
      "Requires both RSI oversold (< 40) AND bullish MACD crossover to enter long. Reduces false signals.",
    tags: ["momentum", "filter"],
    definition: {
      symbol: "AAPL",
      indicators: [
        { id: "rsi", category: "technical", type: "rsi", params: { period: 14 } },
        { id: "macd", category: "technical", type: "macd", params: { fast: 12, slow: 26, signal_period: 9 } },
      ],
      long: {
        entry: {
          direction: "buy",
          conditions: [
            { left: "rsi", op: "<", right: 40 },
            { left: "macd_line", op: ">", right: "macd_signal" },
          ],
          logic: "and",
        },
        exit: {
          direction: "buy",
          conditions: [
            { left: "rsi", op: ">", right: 60 },
          ],
          logic: "and",
        },
      },
      risk: { ...defaultRisk },
    },
  },
  {
    id: "atr_breakout",
    name: "ATR Breakout",
    description:
      "Enter long when price breaks above the 20-bar high by more than 1× ATR. Uses ATR-based sizing for position risk.",
    tags: ["breakout", "volatility", "trend"],
    definition: {
      symbol: "AAPL",
      indicators: [
        { id: "atr", category: "technical", type: "atr", params: { period: 14 } },
        { id: "sma", category: "technical", type: "sma", params: { period: 20 } },
      ],
      long: {
        entry: {
          direction: "buy",
          conditions: [{ left: "close", op: ">", right: "sma" }],
          logic: "and",
        },
        exit: {
          direction: "buy",
          conditions: [{ left: "close", op: "<", right: "sma" }],
          logic: "and",
        },
      },
      risk: {
        ...defaultRisk,
        risk_type: "atr",
        atr_multiplier: 2.0,
      },
    },
  },

  // ── Range regime ────────────────────────────────────────────────────────────
  {
    id: "bb_rsi_dual_confirm",
    name: "BB + RSI Dual Confirm",
    description:
      "Price must pierce the BB band AND RSI must confirm oversold/overbought before entry. Reduces false touches in trending markets.",
    tags: ["range", "mean-reversion", "filter"],
    definition: {
      symbol: "AAPL",
      indicators: [
        { id: "bb",  category: "technical", type: "bb",  params: { period: 20, std_dev: 2 } },
        { id: "rsi", category: "technical", type: "rsi", params: { period: 14 } },
      ],
      long: {
        entry: {
          direction: "buy",
          conditions: [
            { left: "close", op: "<", right: "bb_lower" },
            { left: "rsi",   op: "<", right: 35 },
          ],
          logic: "and",
        },
        exit: {
          direction: "buy",
          conditions: [{ left: "close", op: ">", right: "bb_middle" }],
          logic: "and",
        },
      },
      short: {
        entry: {
          direction: "sell",
          conditions: [
            { left: "close", op: ">", right: "bb_upper" },
            { left: "rsi",   op: ">", right: 65 },
          ],
          logic: "and",
        },
        exit: {
          direction: "sell",
          conditions: [{ left: "close", op: "<", right: "bb_middle" }],
          logic: "and",
        },
      },
      risk: { ...defaultRisk },
    },
  },
  {
    id: "ema_channel_reversion",
    name: "EMA Channel Reversion",
    description:
      "Two EMAs form a channel. Buy below the fast EMA (mean-revert to it); short above it. Best in sideways, low-slope markets.",
    tags: ["range", "mean-reversion", "oscillator"],
    definition: {
      symbol: "AAPL",
      indicators: [
        { id: "ema_fast", category: "technical", type: "ema", params: { period: 10 } },
        { id: "ema_slow", category: "technical", type: "ema", params: { period: 30 } },
      ],
      long: {
        entry: {
          direction: "buy",
          conditions: [
            { left: "close",    op: "<", right: "ema_fast" },
            { left: "ema_fast", op: ">", right: "ema_slow" },
          ],
          logic: "and",
        },
        exit: {
          direction: "buy",
          conditions: [{ left: "close", op: ">", right: "ema_fast" }],
          logic: "and",
        },
      },
      short: {
        entry: {
          direction: "sell",
          conditions: [
            { left: "close",    op: ">", right: "ema_fast" },
            { left: "ema_fast", op: "<", right: "ema_slow" },
          ],
          logic: "and",
        },
        exit: {
          direction: "sell",
          conditions: [{ left: "close", op: "<", right: "ema_fast" }],
          logic: "and",
        },
      },
      risk: { ...defaultRisk },
    },
  },
  {
    id: "macd_histogram_oscillation",
    name: "MACD Histogram Oscillation",
    description:
      "Trade the MACD histogram crossing zero. In a range, the histogram oscillates predictably — buy on positive flip, short on negative flip.",
    tags: ["range", "momentum", "oscillator"],
    definition: {
      symbol: "AAPL",
      indicators: [
        { id: "macd", category: "technical", type: "macd", params: { fast: 8, slow: 17, signal_period: 9 } },
      ],
      long: {
        entry: {
          direction: "buy",
          conditions: [{ left: "macd_hist", op: ">", right: 0 }],
          logic: "and",
        },
        exit: {
          direction: "buy",
          conditions: [{ left: "macd_hist", op: "<", right: 0 }],
          logic: "and",
        },
      },
      short: {
        entry: {
          direction: "sell",
          conditions: [{ left: "macd_hist", op: "<", right: 0 }],
          logic: "and",
        },
        exit: {
          direction: "sell",
          conditions: [{ left: "macd_hist", op: ">", right: 0 }],
          logic: "and",
        },
      },
      risk: { ...defaultRisk, cooldown_bars: 2 },
    },
  },
  {
    id: "rsi_bb_slope_range",
    name: "Flat Slope Range Scalp",
    description:
      "Confirms a range regime by requiring a near-flat price slope, then scalps RSI extremes with tight BB exits. Avoids trend entries.",
    tags: ["range", "mean-reversion", "filter"],
    definition: {
      symbol: "AAPL",
      indicators: [
        { id: "rsi",   category: "technical", type: "rsi",   params: { period: 14 } },
        { id: "bb",    category: "technical", type: "bb",    params: { period: 20, std_dev: 1.5 } },
        { id: "slope", category: "technical", type: "slope", params: { period: 10, column: "close" } },
      ],
      long: {
        entry: {
          direction: "buy",
          conditions: [
            { left: "rsi",   op: "<", right: 35 },
            { left: "close", op: "<", right: "bb_lower" },
            { left: "slope", op: ">", right: -0.5 },
          ],
          logic: "and",
        },
        exit: {
          direction: "buy",
          conditions: [
            { left: "rsi", op: ">", right: 55 },
          ],
          logic: "and",
        },
      },
      short: {
        entry: {
          direction: "sell",
          conditions: [
            { left: "rsi",   op: ">", right: 65 },
            { left: "close", op: ">", right: "bb_upper" },
            { left: "slope", op: "<", right: 0.5 },
          ],
          logic: "and",
        },
        exit: {
          direction: "sell",
          conditions: [
            { left: "rsi", op: "<", right: 45 },
          ],
          logic: "and",
        },
      },
      risk: { ...defaultRisk, sl_pct: 0.015, tp_pct: 0.025, cooldown_bars: 3 },
    },
  },
];

// ---------------------------------------------------------------------------
// Tag badge
// ---------------------------------------------------------------------------

const TAG_COLORS: Record<string, string> = {
  trend:            "bg-blue-900 text-blue-300",
  momentum:         "bg-purple-900 text-purple-300",
  "mean-reversion": "bg-green-900 text-green-300",
  oscillator:       "bg-teal-900 text-teal-300",
  crossover:        "bg-sky-900 text-sky-300",
  volatility:       "bg-yellow-900 text-yellow-300",
  breakout:         "bg-orange-900 text-orange-300",
  filter:           "bg-pink-900 text-pink-300",
  range:            "bg-indigo-900 text-indigo-300",
};

type TemplateGroup = { label: string; ids: string[] };

const TEMPLATE_GROUPS: TemplateGroup[] = [
  {
    label: "Trend Following",
    ids: ["macd_crossover", "ema_crossover", "atr_breakout"],
  },
  {
    label: "Momentum & Mean Reversion",
    ids: ["rsi_mean_reversion", "bollinger_band_reversion", "rsi_macd_combo"],
  },
  {
    label: "Range Regime",
    ids: ["bb_rsi_dual_confirm", "ema_channel_reversion", "macd_histogram_oscillation", "rsi_bb_slope_range"],
  },
];

function TagBadge({ tag }: { tag: string }) {
  const cls = TAG_COLORS[tag] ?? "bg-gray-700 text-gray-300";
  return <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${cls}`}>{tag}</span>;
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function NewStrategyPage() {
  const router = useRouter();
  const [step, setStep] = useState<"pick" | "edit">("pick");
  const [templateId, setTemplateId] = useState<string | null>(null);
  const [editorKey, setEditorKey] = useState(0);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const definitionRef = useRef<object>({});

  function selectTemplate(tpl: StrategyTemplate) {
    setTemplateId(tpl.id);
    if (tpl.id !== "scratch" && !name) {
      setName(tpl.name);
      setDescription(tpl.description);
    }
    definitionRef.current = tpl.definition;
    setEditorKey((k) => k + 1);
    setStep("edit");
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSaving(true);
    try {
      const res = await apiFetch("/api/v1/strategies", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description, definition: definitionRef.current }),
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

  const selectedTemplate = TEMPLATES.find((t) => t.id === templateId);

  // ---- Step 1: template picker ----
  if (step === "pick") {
    return (
      <div className="max-w-4xl space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-white">New Strategy</h1>
          <p className="mt-1 text-sm text-gray-400">Choose a starting point — you can modify everything after.</p>
        </div>

        {/* Scratch card */}
        <div>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">From Scratch</h2>
          <button
            onClick={() => selectTemplate(TEMPLATES[0])}
            className="w-full rounded-lg border border-dashed border-gray-600 bg-gray-900 px-5 py-4 text-left transition hover:border-brand-500 hover:bg-gray-800"
          >
            <p className="font-medium text-white">Blank Strategy</p>
            <p className="mt-0.5 text-sm text-gray-400">Start with an empty editor and build your own logic.</p>
          </button>
        </div>

        {/* Template cards grouped by regime */}
        {TEMPLATE_GROUPS.map((group) => {
          const groupTemplates = group.ids.map((id) => TEMPLATES.find((t) => t.id === id)!).filter(Boolean);
          return (
            <div key={group.label}>
              <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">{group.label}</h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {groupTemplates.map((tpl) => (
                  <button
                    key={tpl.id}
                    onClick={() => selectTemplate(tpl)}
                    className="rounded-lg border border-gray-700 bg-gray-900 px-5 py-4 text-left transition hover:border-brand-500 hover:bg-gray-800"
                  >
                    <p className="font-medium text-white">{tpl.name}</p>
                    <p className="mt-1 text-sm text-gray-400">{tpl.description}</p>
                    {tpl.tags.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1">
                        {tpl.tags.map((tag) => <TagBadge key={tag} tag={tag} />)}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  // ---- Step 2: editor ----
  return (
    <div className="max-w-3xl space-y-6">
      <div className="flex items-center gap-3">
        <button
          onClick={() => setStep("pick")}
          className="text-sm text-gray-400 hover:text-white"
        >
          ← Back
        </button>
        <h1 className="text-2xl font-semibold text-white">
          {selectedTemplate && selectedTemplate.id !== "scratch" ? (
            <>New Strategy <span className="text-base font-normal text-gray-400">from {selectedTemplate.name}</span></>
          ) : (
            "New Strategy"
          )}
        </h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-4">
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
        </div>

        <div>
          <label className="mb-2 block text-sm text-gray-400">Strategy Definition</label>
          <div className="rounded border border-gray-700 bg-gray-950 p-4">
            <StrategyEditor
              key={editorKey}
              initialDefinition={selectedTemplate?.definition}
              onChange={(def) => { definitionRef.current = def; }}
            />
          </div>
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
          <button
            type="button"
            onClick={() => setStep("pick")}
            className="rounded border border-gray-700 px-4 py-2 text-sm text-gray-400 hover:text-white"
          >
            Change Template
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
